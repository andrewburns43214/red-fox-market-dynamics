import { createClient } from 'npm:@supabase/supabase-js@2';
import Stripe from 'npm:stripe@17.7.0';

const stripe = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
  apiVersion: '2023-10-16',
  httpClient: Stripe.createFetchHttpClient(),
});

const stripeStatus = (status: Stripe.Subscription.Status) => {
  if (status === 'active') return 'active';
  if (status === 'past_due') return 'past_due';
  if (status === 'unpaid') return 'unpaid';
  if (status === 'incomplete') return 'incomplete';
  if (status === 'incomplete_expired') return 'incomplete_expired';
  return 'canceled';
};

function periodDate(seconds: number | null | undefined) {
  return seconds ? new Date(seconds * 1000).toISOString() : null;
}

async function subscriptionOwner(
  supabase: ReturnType<typeof createClient>, subscription: Stripe.Subscription,
) {
  const metadataUser = subscription.metadata?.supabase_uid;
  if (metadataUser) return { userId: metadataUser, plan: subscription.metadata?.plan || null };
  const { data: profile } = await supabase
    .from('profiles')
    .select('id')
    .eq('stripe_customer_id', String(subscription.customer))
    .maybeSingle();
  if (!profile) return null;
  const annualPrice = Deno.env.get('STRIPE_PRICE_ANNUAL');
  const plan = subscription.items.data.some((item) => item.price.id === annualPrice) ? 'annual' : 'professional';
  return { userId: profile.id, plan };
}

async function applySubscription(
  supabase: ReturnType<typeof createClient>, event: Stripe.Event, subscription: Stripe.Subscription,
) {
  const owner = await subscriptionOwner(supabase, subscription);
  if (!owner) throw new Error(`Unable to resolve Red Fox user for subscription ${subscription.id}`);
  const { error } = await supabase.rpc('apply_stripe_subscription_event', {
    p_event_id: event.id,
    p_event_type: event.type,
    p_event_created_at: new Date(event.created * 1000).toISOString(),
    p_user_id: owner.userId,
    p_plan: owner.plan || 'professional',
    p_subscription_id: subscription.id,
    p_status: stripeStatus(subscription.status),
    p_period_start: periodDate(subscription.current_period_start),
    p_period_end: periodDate(subscription.current_period_end),
    p_cancel_at_period_end: subscription.cancel_at_period_end,
    p_canceled_at: periodDate(subscription.canceled_at),
  });
  if (error) throw error;
}

Deno.serve(async (req) => {
  const body = await req.text();
  const sig = req.headers.get('stripe-signature');

  if (!sig) {
    return new Response('Missing signature', { status: 400 });
  }

  let event: Stripe.Event;
  try {
    event = stripe.webhooks.constructEvent(
      body,
      sig,
      Deno.env.get('STRIPE_WEBHOOK_SECRET')!
    );
  } catch (err) {
    console.error('Webhook signature verification failed:', err.message);
    return new Response('Invalid signature', { status: 400 });
  }

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
  );

  switch (event.type) {
    case 'checkout.session.completed': {
      const session = event.data.object as Stripe.Checkout.Session;
      const userId = session.metadata?.supabase_uid;
      const plan = session.metadata?.plan;

      if (!userId || !plan) {
        console.error('Missing metadata on checkout session');
        break;
      }

      if (plan === 'day_pass') {
        // Expire at the next midnight in New York, including daylight-saving changes.
        const now = new Date();
        const etFormatter = new Intl.DateTimeFormat('en-US', {
          timeZone: 'America/New_York',
          year: 'numeric', month: '2-digit', day: '2-digit',
        });
        const tomorrow = new Date(now.getTime() + 24 * 60 * 60 * 1000);
        const dateParts = Object.fromEntries(
          etFormatter.formatToParts(tomorrow)
            .filter((part) => part.type !== 'literal')
            .map((part) => [part.type, part.value])
        );
        const midnightUtc = Date.UTC(
          Number(dateParts.year),
          Number(dateParts.month) - 1,
          Number(dateParts.day),
        );
        const offsetPart = new Intl.DateTimeFormat('en-US', {
          timeZone: 'America/New_York',
          timeZoneName: 'longOffset',
        }).formatToParts(new Date(midnightUtc)).find((part) => part.type === 'timeZoneName')?.value;
        const offsetMatch = offsetPart?.match(/^GMT([+-])(\d{2}):(\d{2})$/);
        if (!offsetMatch) throw new Error('Unable to calculate New York expiration time');
        const offsetMinutes = (Number(offsetMatch[2]) * 60 + Number(offsetMatch[3]))
          * (offsetMatch[1] === '+' ? 1 : -1);
        const midnightET = new Date(midnightUtc - offsetMinutes * 60 * 1000);

        const { error } = await supabase.rpc('apply_stripe_checkout_event', {
          p_event_id: event.id,
          p_event_type: event.type,
          p_event_created_at: new Date(event.created * 1000).toISOString(),
          p_user_id: userId,
          p_plan: 'day_pass',
          p_checkout_session_id: session.id,
          p_subscription_id: '',
          p_payment_intent_id: String(session.payment_intent || ''),
          p_period_start: now.toISOString(),
          p_period_end: midnightET.toISOString(),
        });
        if (error) throw error;
      } else if (plan === 'professional' || plan === 'annual') {
        const subscription = await stripe.subscriptions.retrieve(
          session.subscription as string
        );
        const { error } = await supabase.rpc('apply_stripe_checkout_event', {
          p_event_id: event.id,
          p_event_type: event.type,
          p_event_created_at: new Date(event.created * 1000).toISOString(),
          p_user_id: userId,
          p_plan: plan,
          p_checkout_session_id: session.id,
          p_subscription_id: subscription.id,
          p_payment_intent_id: String(session.payment_intent || ''),
          p_period_start: periodDate(subscription.current_period_start),
          p_period_end: periodDate(subscription.current_period_end),
        });
        if (error) throw error;
      }
      break;
    }

    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      const subscription = event.data.object as Stripe.Subscription;
      await applySubscription(supabase, event, subscription);
      break;
    }

    case 'invoice.payment_failed': {
      const invoice = event.data.object as Stripe.Invoice;
      if (invoice.subscription) {
        const subscription = await stripe.subscriptions.retrieve(String(invoice.subscription));
        await applySubscription(supabase, event, subscription);
      }
      break;
    }

    case 'charge.refunded': {
      const charge = event.data.object as Stripe.Charge;
      if (charge.amount_refunded >= charge.amount) {
        let subscriptionId: string | null = null;
        if (charge.invoice) {
          const invoice = await stripe.invoices.retrieve(String(charge.invoice));
          subscriptionId = invoice.subscription ? String(invoice.subscription) : null;
        }
        const { error } = await supabase.rpc('revoke_stripe_paid_access', {
          p_event_id: event.id,
          p_event_type: event.type,
          p_event_created_at: new Date(event.created * 1000).toISOString(),
          p_payment_intent_id: String(charge.payment_intent || ''),
          p_subscription_id: subscriptionId,
          p_reason: 'full_refund',
        });
        if (error) throw error;
      }
      break;
    }

    case 'charge.dispute.closed': {
      const dispute = event.data.object as Stripe.Dispute;
      // A lost dispute is the confirmed, reversed-payment state. Open or won
      // disputes do not alter access.
      if (dispute.status === 'lost') {
        const charge = await stripe.charges.retrieve(String(dispute.charge));
        let subscriptionId: string | null = null;
        if (charge.invoice) {
          const invoice = await stripe.invoices.retrieve(String(charge.invoice));
          subscriptionId = invoice.subscription ? String(invoice.subscription) : null;
        }
        const { error } = await supabase.rpc('revoke_stripe_paid_access', {
          p_event_id: event.id,
          p_event_type: event.type,
          p_event_created_at: new Date(event.created * 1000).toISOString(),
          p_payment_intent_id: String(charge.payment_intent || ''),
          p_subscription_id: subscriptionId,
          p_reason: 'chargeback_lost',
        });
        if (error) throw error;
      }
      break;
    }

    default:
      console.log('Unhandled event type:', event.type);
  }

  return new Response(JSON.stringify({ received: true }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
});
