-- Repair the one-time trial claim RPC without changing paid, granted, or bypass access.
-- This is intentionally idempotent so it is safe to apply to a project where the
-- earlier entitlement migration was only partly materialized.

ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS normalized_email TEXT,
  ADD COLUMN IF NOT EXISTS trial_started_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMPTZ;

UPDATE public.profiles
SET normalized_email = LOWER(BTRIM(email))
WHERE normalized_email IS NULL;

CREATE TABLE IF NOT EXISTS public.trial_claims (
  normalized_email TEXT PRIMARY KEY,
  claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.trial_claims ENABLE ROW LEVEL SECURITY;

-- Preserve consumed historical trials if this table was absent in a partially
-- applied deployment. This does not alter any existing trial end timestamp.
INSERT INTO public.trial_claims (normalized_email, claimed_at)
SELECT p.normalized_email,
       COALESCE(p.trial_started_at, p.trial_ends_at - INTERVAL '24 hours', p.created_at)
FROM public.profiles p
WHERE p.normalized_email IS NOT NULL
  AND p.trial_ends_at IS NOT NULL
ON CONFLICT (normalized_email) DO NOTHING;

CREATE OR REPLACE FUNCTION public.claim_free_day()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  normalized TEXT;
  existing_trial_end TIMESTAMPTZ;
  granted BOOLEAN;
BEGIN
  -- Lock the account row before its one-time email claim is recorded.
  SELECT normalized_email, trial_ends_at
    INTO normalized, existing_trial_end
  FROM public.profiles
  WHERE id = auth.uid()
  FOR UPDATE;

  IF normalized IS NULL OR existing_trial_end IS NOT NULL THEN
    RETURN FALSE;
  END IF;

  INSERT INTO public.trial_claims(normalized_email)
  VALUES (normalized)
  ON CONFLICT (normalized_email) DO NOTHING
  RETURNING TRUE INTO granted;

  IF COALESCE(granted, FALSE) = FALSE THEN
    RETURN FALSE;
  END IF;

  UPDATE public.profiles
  SET trial_started_at = NOW(),
      trial_ends_at = NOW() + INTERVAL '24 hours',
      updated_at = NOW()
  WHERE id = auth.uid()
    AND trial_ends_at IS NULL;

  RETURN FOUND;
END;
$$;

REVOKE ALL ON FUNCTION public.claim_free_day() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.claim_free_day() TO authenticated;

NOTIFY pgrst, 'reload schema';
