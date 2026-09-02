-- One no-card-required trial per newly created account. It is never reset.
ALTER TABLE public.profiles
    ADD COLUMN IF NOT EXISTS trial_ends_at TIMESTAMPTZ;

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, trial_ends_at)
    VALUES (NEW.id, NEW.email, NOW() + INTERVAL '24 hours');
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION public.has_active_access()
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM public.profiles
        WHERE id = auth.uid()
          AND (complimentary_access = TRUE OR trial_ends_at > NOW())
    ) OR EXISTS (
        SELECT 1
        FROM public.subscriptions
        WHERE user_id = auth.uid()
          AND status = 'active'
          AND current_period_end > NOW()
    );
END;
$$;
