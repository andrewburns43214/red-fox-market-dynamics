-- Preserve one no-card trial per verified email even if the account is later deleted.
CREATE TABLE IF NOT EXISTS public.trial_claims (
    normalized_email TEXT PRIMARY KEY,
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE public.trial_claims ENABLE ROW LEVEL SECURITY;

-- Users never need to write their own access fields. Stripe and the signup trigger use
-- privileged server-side paths, so removing this policy prevents self-granted trials.
DROP POLICY IF EXISTS "Users can update own profile" ON public.profiles;

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    newly_claimed TEXT;
BEGIN
    INSERT INTO public.trial_claims (normalized_email)
    VALUES (LOWER(NEW.email))
    ON CONFLICT (normalized_email) DO NOTHING
    RETURNING normalized_email INTO newly_claimed;

    INSERT INTO public.profiles (id, email, trial_ends_at)
    VALUES (
        NEW.id,
        NEW.email,
        CASE WHEN newly_claimed IS NOT NULL THEN NOW() + INTERVAL '24 hours' ELSE NULL END
    );
    RETURN NEW;
END;
$$;
