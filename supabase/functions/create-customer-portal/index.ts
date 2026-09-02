import { createClient } from 'npm:@supabase/supabase-js@2';
import Stripe from 'npm:stripe@17.7.0';

const stripe = new Stripe(Deno.env.get('STRIPE_SECRET_KEY')!, {
  apiVersion: '2023-10-16',
  httpClient: Stripe.createFetchHttpClient(),
});

const allowedOrigins = new Set(['https://redfoxmi.com', 'https://www.redfoxmi.com']);

function corsHeaders(req: Request) {
  const origin = req.headers.get('Origin') || '';
  return {
  'Access-Control-Allow-Origin': allowedOrigins.has(origin) ? origin : 'https://www.redfoxmi.com',
  'Vary': 'Origin',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  };
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders(req) });
  if (req.method !== 'POST') return new Response('Method not allowed', { status: 405, headers: corsHeaders(req) });

  const supabase = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_ANON_KEY')!,
    { global: { headers: { Authorization: req.headers.get('Authorization') || '' } } },
  );
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return new Response(JSON.stringify({ error: 'Unauthorized' }), { status: 401, headers: { ...corsHeaders(req), 'Content-Type': 'application/json' } });

  const admin = createClient(Deno.env.get('SUPABASE_URL')!, Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!);
  const { data: profile } = await admin.from('profiles').select('stripe_customer_id').eq('id', user.id).single();
  if (!profile?.stripe_customer_id) return new Response(JSON.stringify({ error: 'No billing profile found' }), { status: 404, headers: { ...corsHeaders(req), 'Content-Type': 'application/json' } });

  try {
    const session = await stripe.billingPortal.sessions.create({
      customer: profile.stripe_customer_id,
      return_url: Deno.env.get('SITE_URL')! + '/',
    });
    return new Response(JSON.stringify({ url: session.url }), { headers: { ...corsHeaders(req), 'Content-Type': 'application/json' } });
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unable to open billing portal';
    console.error('Billing portal creation failed:', message);
    return new Response(JSON.stringify({ error: message }), { status: 500, headers: { ...corsHeaders(req), 'Content-Type': 'application/json' } });
  }
});
