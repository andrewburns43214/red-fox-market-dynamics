import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2';
import Stripe from 'https://esm.sh/stripe@13?target=deno';

const stripe = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
  apiVersion: '2023-10-16',
  httpClient: Stripe.createFetchHttpClient(),
});

serve(async (req) => {
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

        await supabase.from('subscriptions').insert({
          user_id: userId,
          plan: 'day_pass',
          status: 'active',
          stripe_checkout_session_id: session.id,
          current_period_start: now.toISOString(),
          current_period_end: midnightET.toISOString(),
        });
      } else if (plan === 'professional' || plan === 'annual') {
        const subscription = await stripe.subscriptions.retrieve(
          session.subscription as string
        );
        await supabase.from('subscriptions').insert({
          user_id: userId,
          plan,
          status: 'active',
          stripe_subscription_id: subscription.id,
          stripe_checkout_session_id: session.id,
          current_period_start: new Date(subscription.current_period_start * 1000).toISOString(),
          current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
        });
      }
      break;
    }

    case 'customer.subscription.updated':
    case 'customer.subscription.deleted': {
      const subscription = event.data.object as Stripe.Subscription;
      const status = subscription.status === 'active' ? 'active'
        : subscription.status === 'past_due' ? 'past_due'
        : 'canceled';

      await supabase
        .from('subscriptions')
        .update({
          status,
          current_period_end: new Date(subscription.current_period_end * 1000).toISOString(),
          updated_at: new Date().toISOString(),
        })
        .eq('stripe_subscription_id', subscription.id);
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
