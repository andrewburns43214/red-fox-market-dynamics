-- Add the annual Stripe subscription plan without changing existing records.
ALTER TABLE public.subscriptions
    DROP CONSTRAINT IF EXISTS subscriptions_plan_check;

ALTER TABLE public.subscriptions
    ADD CONSTRAINT subscriptions_plan_check
    CHECK (plan IN ('day_pass', 'professional', 'annual'));
