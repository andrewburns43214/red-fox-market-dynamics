-- Complimentary access is reserved for the owner and the controlled guest account.
ALTER TABLE public.profiles
    ADD COLUMN IF NOT EXISTS complimentary_access BOOLEAN NOT NULL DEFAULT FALSE;

UPDATE public.profiles
SET complimentary_access = TRUE
WHERE lower(email) IN (
    'andrewburns43214@gmail.com',
    'andrewburns43214+redfoxguests@gmail.com'
);

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
          AND complimentary_access = TRUE
    ) OR EXISTS (
        SELECT 1
        FROM public.subscriptions
        WHERE user_id = auth.uid()
          AND status = 'active'
          AND current_period_end > NOW()
    );
END;
$$;
