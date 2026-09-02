-- New accounts choose their access first. The one-time free day begins only
-- after the signed-in customer explicitly claims it from the pricing page.
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, trial_ends_at)
    VALUES (NEW.id, NEW.email, NULL);
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.claim_free_day()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    claimed BOOLEAN;
BEGIN
    UPDATE public.profiles
    SET trial_ends_at = NOW() + INTERVAL '24 hours',
        updated_at = NOW()
    WHERE id = auth.uid()
      AND trial_ends_at IS NULL
    RETURNING TRUE INTO claimed;

    RETURN COALESCE(claimed, FALSE);
END;
$$;
