-- Customer identity, roles, grants, and durable entitlement state.
-- Additive only: existing profiles, subscriptions, grants, and auth users remain valid.

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS first_name TEXT,
  ADD COLUMN IF NOT EXISTS last_name TEXT,
  ADD COLUMN IF NOT EXISTS normalized_email TEXT,
  ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS trial_started_at TIMESTAMPTZ;

UPDATE public.profiles
SET normalized_email = LOWER(BTRIM(email))
WHERE normalized_email IS NULL;

ALTER TABLE public.subscriptions
  ADD COLUMN IF NOT EXISTS stripe_payment_intent_id TEXT,
  ADD COLUMN IF NOT EXISTS last_stripe_event_created_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS last_stripe_event_id TEXT,
  ADD COLUMN IF NOT EXISTS cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS canceled_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS revocation_reason TEXT;

ALTER TABLE public.subscriptions DROP CONSTRAINT IF EXISTS subscriptions_status_check;
ALTER TABLE public.subscriptions
  ADD CONSTRAINT subscriptions_status_check
  CHECK (status IN ('active', 'canceled', 'past_due', 'unpaid', 'incomplete', 'incomplete_expired', 'expired', 'revoked'));

CREATE INDEX IF NOT EXISTS idx_profiles_normalized_email ON public.profiles(normalized_email);
CREATE INDEX IF NOT EXISTS idx_profiles_created_at ON public.profiles(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_subscriptions_access_lookup
  ON public.subscriptions(user_id, status, current_period_end DESC);
CREATE INDEX IF NOT EXISTS idx_subscriptions_payment_intent
  ON public.subscriptions(stripe_payment_intent_id) WHERE stripe_payment_intent_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS public.user_roles (
  user_id UUID PRIMARY KEY REFERENCES public.profiles(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('admin')),
  granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  granted_by UUID REFERENCES public.profiles(id) ON DELETE SET NULL
);

ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

CREATE TABLE IF NOT EXISTS public.admin_seed_emails (
  normalized_email TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO public.admin_seed_emails(normalized_email)
VALUES ('andrewburns43214@gmail.com')
ON CONFLICT (normalized_email) DO NOTHING;

INSERT INTO public.user_roles(user_id, role)
SELECT id, 'admin'
FROM public.profiles
WHERE normalized_email = 'andrewburns43214@gmail.com'
ON CONFLICT (user_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.access_grants (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  access_type TEXT NOT NULL DEFAULT 'complimentary' CHECK (access_type IN ('complimentary')),
  active BOOLEAN NOT NULL DEFAULT TRUE,
  granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  granted_by UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
  reason TEXT,
  expires_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ
);

ALTER TABLE public.access_grants ENABLE ROW LEVEL SECURITY;
CREATE INDEX IF NOT EXISTS idx_access_grants_active
  ON public.access_grants(user_id, active, expires_at);

-- Retain all existing boolean grants as auditable indefinite grants.
INSERT INTO public.access_grants(user_id, reason)
SELECT id, 'Migrated existing complimentary access'
FROM public.profiles
WHERE complimentary_access = TRUE
  AND NOT EXISTS (
    SELECT 1 FROM public.access_grants g
    WHERE g.user_id = profiles.id AND g.active = TRUE
  );

CREATE TABLE IF NOT EXISTS public.stripe_webhook_events (
  stripe_event_id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  event_created_at TIMESTAMPTZ NOT NULL,
  resource_id TEXT,
  processing_result TEXT NOT NULL DEFAULT 'applied' CHECK (processing_result IN ('applied', 'duplicate', 'stale', 'ignored')),
  processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stripe_webhook_events_resource
  ON public.stripe_webhook_events(resource_id, event_created_at DESC);

CREATE OR REPLACE FUNCTION public.is_admin()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = auth.uid() AND role = 'admin'
  )
$$;

CREATE OR REPLACE FUNCTION public.has_active_access()
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.profiles p
    WHERE p.id = auth.uid()
      AND (
        p.complimentary_access = TRUE
        OR (p.trial_ends_at IS NOT NULL AND p.trial_ends_at > NOW())
        OR EXISTS (
          SELECT 1 FROM public.access_grants g
          WHERE g.user_id = p.id
            AND g.active = TRUE
            AND g.revoked_at IS NULL
            AND (g.expires_at IS NULL OR g.expires_at > NOW())
        )
      )
  ) OR EXISTS (
    SELECT 1 FROM public.subscriptions s
    WHERE s.user_id = auth.uid()
      AND s.status IN ('active', 'canceled')
      AND s.revoked_at IS NULL
      AND s.current_period_end > NOW()
  )
$$;

CREATE OR REPLACE FUNCTION public.claim_free_day()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  normalized TEXT;
  granted BOOLEAN;
BEGIN
  SELECT normalized_email INTO normalized FROM public.profiles WHERE id = auth.uid() FOR UPDATE;
  IF normalized IS NULL THEN RETURN FALSE; END IF;

  INSERT INTO public.trial_claims(normalized_email)
  VALUES (normalized)
  ON CONFLICT (normalized_email) DO NOTHING
  RETURNING TRUE INTO granted;

  IF COALESCE(granted, FALSE) = FALSE THEN RETURN FALSE; END IF;

  UPDATE public.profiles
  SET trial_started_at = NOW(),
      trial_ends_at = NOW() + INTERVAL '24 hours',
      updated_at = NOW()
  WHERE id = auth.uid() AND trial_ends_at IS NULL;

  RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  given_first TEXT := NULLIF(BTRIM(COALESCE(NEW.raw_user_meta_data->>'first_name', '')), '');
  given_last TEXT := NULLIF(BTRIM(COALESCE(NEW.raw_user_meta_data->>'last_name', '')), '');
  normalized TEXT := LOWER(BTRIM(NEW.email));
BEGIN
  -- New signups must have real names; existing auth users are never reprocessed.
  IF given_first IS NULL OR given_last IS NULL OR LENGTH(given_first) > 100 OR LENGTH(given_last) > 100 THEN
    RAISE EXCEPTION 'First and last name are required';
  END IF;
  IF EXISTS (SELECT 1 FROM public.profiles WHERE normalized_email = normalized) THEN
    RAISE EXCEPTION 'An account already exists for this email';
  END IF;

  INSERT INTO public.profiles (id, email, normalized_email, first_name, last_name, trial_ends_at)
  VALUES (NEW.id, NEW.email, normalized, given_first, given_last, NULL);

  INSERT INTO public.user_roles(user_id, role)
  SELECT NEW.id, 'admin'
  WHERE EXISTS (SELECT 1 FROM public.admin_seed_emails WHERE normalized_email = normalized)
  ON CONFLICT (user_id) DO NOTHING;
  RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.touch_last_login()
RETURNS VOID
LANGUAGE sql
SECURITY DEFINER
SET search_path = public
AS $$
  UPDATE public.profiles SET last_login_at = NOW(), updated_at = NOW() WHERE id = auth.uid()
$$;

CREATE OR REPLACE FUNCTION public.admin_list_users(
  p_search TEXT DEFAULT NULL,
  p_filter TEXT DEFAULT 'all',
  p_limit INTEGER DEFAULT 50,
  p_offset INTEGER DEFAULT 0
)
RETURNS TABLE(
  user_id UUID, first_name TEXT, last_name TEXT, email TEXT, signup_date TIMESTAMPTZ,
  last_login TIMESTAMPTZ, access_type TEXT, access_active BOOLEAN, trial_ends TIMESTAMPTZ,
  plan TEXT, subscription_status TEXT, paid_through TIMESTAMPTZ, granted BOOLEAN,
  owner_guest_bypass BOOLEAN, stripe_customer_id TEXT
)
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF NOT public.is_admin() THEN RAISE EXCEPTION 'Forbidden'; END IF;
  IF p_limit < 1 OR p_limit > 100 OR p_offset < 0 THEN RAISE EXCEPTION 'Invalid pagination'; END IF;
  RETURN QUERY
  WITH latest_subscription AS (
    SELECT DISTINCT ON (s.user_id) s.* FROM public.subscriptions s
    ORDER BY s.user_id, s.updated_at DESC, s.created_at DESC
  ), rows AS (
    SELECT p.*, s.plan, s.status, s.current_period_end,
      EXISTS(SELECT 1 FROM public.access_grants g WHERE g.user_id=p.id AND g.active AND g.revoked_at IS NULL AND (g.expires_at IS NULL OR g.expires_at > NOW())) AS active_grant,
      EXISTS(SELECT 1 FROM public.user_roles r WHERE r.user_id=p.id AND r.role='admin') AS admin_role
    FROM public.profiles p LEFT JOIN latest_subscription s ON s.user_id=p.id
  )
  SELECT r.id, r.first_name, r.last_name, r.email, r.created_at, r.last_login_at,
    CASE WHEN r.complimentary_access OR r.active_grant THEN 'Granted'
         WHEN r.trial_ends_at > NOW() THEN 'Free Day'
         WHEN r.status IN ('active','canceled') AND r.current_period_end > NOW() THEN COALESCE(r.plan,'Paid')
         ELSE 'No access' END,
    (r.complimentary_access OR r.active_grant OR r.trial_ends_at > NOW() OR (r.status IN ('active','canceled') AND r.current_period_end > NOW())),
    r.trial_ends_at, r.plan, r.status, r.current_period_end,
    (r.complimentary_access OR r.active_grant),
    (r.normalized_email IN ('andrewburns43214@gmail.com','andrewburns43214+redfoxguests@gmail.com')),
    r.stripe_customer_id
  FROM rows r
  WHERE (p_search IS NULL OR p_search = '' OR concat_ws(' ',r.first_name,r.last_name,r.email) ILIKE '%' || p_search || '%')
    AND (p_filter='all'
      OR (p_filter='active' AND (r.complimentary_access OR r.active_grant OR r.trial_ends_at > NOW() OR (r.status IN ('active','canceled') AND r.current_period_end > NOW())))
      OR (p_filter='trial_active' AND r.trial_ends_at > NOW())
      OR (p_filter='trial_expired' AND r.trial_ends_at IS NOT NULL AND r.trial_ends_at <= NOW())
      OR (p_filter='monthly' AND r.plan='professional')
      OR (p_filter='annual' AND r.plan='annual')
      OR (p_filter='granted' AND (r.complimentary_access OR r.active_grant))
      OR (p_filter='bypass' AND r.normalized_email IN ('andrewburns43214@gmail.com','andrewburns43214+redfoxguests@gmail.com'))
      OR (p_filter='payment_issue' AND r.status IN ('past_due','unpaid','incomplete','incomplete_expired','revoked'))
      OR (p_filter='no_access' AND NOT (r.complimentary_access OR r.active_grant OR r.trial_ends_at > NOW() OR (r.status IN ('active','canceled') AND r.current_period_end > NOW())))
    )
  ORDER BY r.created_at DESC
  LIMIT p_limit OFFSET p_offset;
END;
$$;

REVOKE ALL ON FUNCTION public.admin_list_users(TEXT, TEXT, INTEGER, INTEGER) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.admin_list_users(TEXT, TEXT, INTEGER, INTEGER) TO authenticated;
