import { createClient } from 'npm:@supabase/supabase-js@2';
import Stripe from 'npm:stripe@14.25.0';

const stripe = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
  apiVersion: '2023-10-16',
  httpClient: Stripe.createFetchHttpClient(),
});

const allowedOrigins = new Set(['https://redfoxmi.com', 'https://www.redfoxmi.com']);

function corsHeaders(req: Request) {
  const origin = req.headers.get('Origin') || '';
  return {
    // Checkout requires a valid Supabase JWT; permit both canonical site hosts.
    'Access-Control-Allow-Origin': allowedOrigins.has(origin) ? origin : 'https://www.redfoxmi.com',
    'Vary': 'Origin',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  };
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders(req) });
  }

  try {
    const { plan } = await req.json();
    if (!plan || !['day_pass', 'professional', 'annual'].includes(plan)) {
      return new Response(JSON.stringify({ error: 'Invalid plan' }), {
        status: 400,
        headers: { ...corsHeaders(req), 'Content-Type': 'application/json' },
      });
    }

    // Verify JWT
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_ANON_KEY')!,
      { global: { headers: { Authorization: req.headers.get('Authorization')! } } }
    );
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      return new Response(JSON.stringify({ error: 'Unauthorized' }), {
        status: 401,
        headers: { ...corsHeaders(req), 'Content-Type': 'application/json' },
      });
    }

    // Get or create Stripe customer
    const serviceClient = createClient(
      Deno.env.get('SUPABASE_URL')!,
      Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
    );
    const { data: profile } = await serviceClient
      .from('profiles')
      .select('stripe_customer_id')
      .eq('id', user.id)
      .single();

    let customerId: string | null = profile?.stripe_customer_id || null;
    // A customer ID created under an older Stripe account is not valid after a
    // key/account change. Verify it before using it, then replace it safely.
    if (customerId) {
      try {
        const customer = await stripe.customers.retrieve(customerId);
        if ('deleted' in customer && customer.deleted) customerId = null;
      } catch (err) {
        if ((err as { code?: string }).code === 'resource_missing') {
          customerId = null;
        } else {
          throw err;
        }
      }
    }
    if (!customerId) {
      const customer = await stripe.customers.create({
        email: user.email,
        metadata: { supabase_uid: user.id },
      });
      customerId = customer.id;
      await serviceClient
        .from('profiles')
        .update({ stripe_customer_id: customerId })
        .eq('id', user.id);
    }

    // Create Checkout Session
    const priceId = plan === 'day_pass'
      ? Deno.env.get('STRIPE_PRICE_DAY_PASS')!
      : plan === 'annual'
        ? Deno.env.get('STRIPE_PRICE_ANNUAL')!
        : Deno.env.get('STRIPE_PRICE_PROFESSIONAL')!;

    if (!priceId) throw new Error(`Stripe price is not configured for ${plan}`);

    const sessionConfig: Stripe.Checkout.SessionCreateParams = {
      customer: customerId,
      metadata: { supabase_uid: user.id, plan },
      success_url: Deno.env.get('SITE_URL')! + '/board.html?payment=success',
      cancel_url: Deno.env.get('SITE_URL')! + '/#pricing',
      line_items: [{ price: priceId, quantity: 1 }],
      mode: plan === 'day_pass' ? 'payment' : 'subscription',
      client_reference_id: user.id,
      ...(plan === 'day_pass' ? {} : {
        subscription_data: { metadata: { supabase_uid: user.id, plan } },
      }),
      custom_text: {
        submit: {
          message: 'You are purchasing Red Fox Market Intelligence access through secure Stripe Checkout. Stripe processes your payment; Red Fox never receives or stores your card details.',
        },
      },
    };

    const session = await stripe.checkout.sessions.create(sessionConfig);

    return new Response(JSON.stringify({ url: session.url }), {
      headers: { ...corsHeaders(req), 'Content-Type': 'application/json' },
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown checkout error';
    // Do not include secrets, but retain the provider error in function logs so
    // payment configuration problems can be fixed without exposing details to
    // customers.
    console.error('Checkout session creation failed:', message);
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { ...corsHeaders(req), 'Content-Type': 'application/json' },
    });
  }
});
