import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';

const corsHeaders = {
  'Access-Control-Allow-Origin': 'https://www.redfoxmi.com',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: corsHeaders });
  if (req.method !== 'POST') return new Response('Method not allowed', { status: 405, headers: corsHeaders });
  const origin = req.headers.get('Origin');
  if (origin !== 'https://www.redfoxmi.com' && origin !== 'https://redfoxmi.com') {
    return new Response('Forbidden', { status: 403, headers: corsHeaders });
  }

  try {
    const { name, email, category, message, website } = await req.json();
    if (website) return new Response(JSON.stringify({ ok: true }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
    if (!name || !email || !message || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(String(email)) || String(message).length > 4000) {
      throw new Error('Please complete the required fields.');
    }

    const response = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: { Authorization: `Bearer ${Deno.env.get('RESEND_API_KEY')}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({
        from: Deno.env.get('SUPPORT_FROM_EMAIL')!,
        to: [Deno.env.get('SUPPORT_TO_EMAIL')!],
        reply_to: String(email),
        subject: `[Red Fox Support] ${String(category || 'General')} - ${String(name).slice(0, 100)}`,
        text: `From: ${name}\nEmail: ${email}\nCategory: ${category || 'General'}\n\n${message}`,
      }),
    });
    if (!response.ok) throw new Error('Unable to send your request.');
    return new Response(JSON.stringify({ ok: true }), { headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message || 'Unable to send your request.' }), { status: 400, headers: { ...corsHeaders, 'Content-Type': 'application/json' } });
  }
});
