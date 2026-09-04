import { createClient } from 'npm:@supabase/supabase-js@2';

const allowedOrigins = new Set(['https://redfoxmi.com', 'https://www.redfoxmi.com']);

function cors(req: Request) {
  const origin = req.headers.get('origin') || '';
  return {
    'Access-Control-Allow-Origin': allowedOrigins.has(origin) ? origin : 'https://www.redfoxmi.com',
    'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
    'Vary': 'Origin',
  };
}

Deno.serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: cors(req) });
  if (req.method !== 'POST') return new Response('Method not allowed', { status: 405, headers: cors(req) });

  const authorization = req.headers.get('authorization');
  if (!authorization) return new Response('Unauthorized', { status: 401, headers: cors(req) });
  const userClient = createClient(
    Deno.env.get('SUPABASE_URL')!,
    Deno.env.get('SUPABASE_ANON_KEY')!,
    { global: { headers: { Authorization: authorization } } },
  );
  const { data: authData, error: authError } = await userClient.auth.getUser();
  if (authError || !authData.user) return new Response('Unauthorized', { status: 401, headers: cors(req) });
  const { data: admin, error: adminError } = await userClient.rpc('is_admin');
  if (adminError || admin !== true) return new Response('Forbidden', { status: 403, headers: cors(req) });

  let body: { search?: string; filter?: string; limit?: number; offset?: number } = {};
  try { body = await req.json(); } catch { /* defaults are safe */ }
  const limit = Math.min(Math.max(Number(body.limit) || 50, 1), 100);
  const offset = Math.max(Number(body.offset) || 0, 0);
  const { data, error } = await userClient.rpc('admin_list_users', {
    p_search: String(body.search || '').trim() || null,
    p_filter: String(body.filter || 'all'),
    p_limit: limit,
    p_offset: offset,
  });
  if (error) {
    console.error('Admin directory query failed:', error.message);
    return new Response('Unable to load users', { status: 500, headers: cors(req) });
  }
  return new Response(JSON.stringify({ users: data, limit, offset }), {
    headers: { ...cors(req), 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
  });
});
