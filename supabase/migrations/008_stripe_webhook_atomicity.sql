-- Atomic Stripe event application. Each mutation and its event receipt commit together.

CREATE OR REPLACE FUNCTION public.apply_stripe_checkout_event(
  p_event_id TEXT,
  p_event_type TEXT,
  p_event_created_at TIMESTAMPTZ,
  p_user_id UUID,
  p_plan TEXT,
  p_checkout_session_id TEXT,
  p_subscription_id TEXT,
  p_payment_intent_id TEXT,
  p_period_start TIMESTAMPTZ,
  p_period_end TIMESTAMPTZ
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE existing_id UUID;
BEGIN
  INSERT INTO public.stripe_webhook_events(stripe_event_id,event_type,event_created_at,resource_id)
  VALUES (p_event_id,p_event_type,p_event_created_at,COALESCE(p_subscription_id,p_checkout_session_id))
  ON CONFLICT (stripe_event_id) DO NOTHING;
  IF NOT FOUND THEN RETURN 'duplicate'; END IF;

  SELECT id INTO existing_id FROM public.subscriptions
  WHERE stripe_checkout_session_id=p_checkout_session_id
  ORDER BY created_at ASC LIMIT 1 FOR UPDATE;
  IF existing_id IS NULL THEN
    INSERT INTO public.subscriptions(
      user_id,plan,status,stripe_subscription_id,stripe_checkout_session_id,stripe_payment_intent_id,
      current_period_start,current_period_end,last_stripe_event_created_at,last_stripe_event_id
    ) VALUES (
      p_user_id,p_plan,'active',NULLIF(p_subscription_id,''),p_checkout_session_id,NULLIF(p_payment_intent_id,''),
      p_period_start,p_period_end,p_event_created_at,p_event_id
    );
  ELSE
    UPDATE public.subscriptions SET
      status='active', stripe_subscription_id=COALESCE(NULLIF(p_subscription_id,''),stripe_subscription_id),
      stripe_payment_intent_id=COALESCE(NULLIF(p_payment_intent_id,''),stripe_payment_intent_id),
      current_period_start=p_period_start,current_period_end=p_period_end,
      last_stripe_event_created_at=p_event_created_at,last_stripe_event_id=p_event_id,
      revoked_at=NULL,revocation_reason=NULL,updated_at=NOW()
    WHERE id=existing_id;
  END IF;
  RETURN 'applied';
END;
$$;

CREATE OR REPLACE FUNCTION public.apply_stripe_subscription_event(
  p_event_id TEXT,
  p_event_type TEXT,
  p_event_created_at TIMESTAMPTZ,
  p_user_id UUID,
  p_plan TEXT,
  p_subscription_id TEXT,
  p_status TEXT,
  p_period_start TIMESTAMPTZ,
  p_period_end TIMESTAMPTZ,
  p_cancel_at_period_end BOOLEAN DEFAULT FALSE,
  p_canceled_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE existing public.subscriptions%ROWTYPE;
BEGIN
  INSERT INTO public.stripe_webhook_events(stripe_event_id,event_type,event_created_at,resource_id)
  VALUES (p_event_id,p_event_type,p_event_created_at,p_subscription_id)
  ON CONFLICT (stripe_event_id) DO NOTHING;
  IF NOT FOUND THEN RETURN 'duplicate'; END IF;

  SELECT * INTO existing FROM public.subscriptions
  WHERE stripe_subscription_id=p_subscription_id
  ORDER BY created_at ASC LIMIT 1 FOR UPDATE;
  IF FOUND AND existing.last_stripe_event_created_at IS NOT NULL
    AND (existing.last_stripe_event_created_at > p_event_created_at
      OR (existing.last_stripe_event_created_at = p_event_created_at AND existing.last_stripe_event_id >= p_event_id)) THEN
    UPDATE public.stripe_webhook_events SET processing_result='stale' WHERE stripe_event_id=p_event_id;
    RETURN 'stale';
  END IF;

  IF NOT FOUND THEN
    INSERT INTO public.subscriptions(
      user_id,plan,status,stripe_subscription_id,current_period_start,current_period_end,
      last_stripe_event_created_at,last_stripe_event_id,cancel_at_period_end,canceled_at
    ) VALUES (
      p_user_id,p_plan,p_status,p_subscription_id,p_period_start,p_period_end,
      p_event_created_at,p_event_id,p_cancel_at_period_end,p_canceled_at
    );
  ELSE
    UPDATE public.subscriptions SET
      status=p_status,current_period_start=p_period_start,current_period_end=p_period_end,
      last_stripe_event_created_at=p_event_created_at,last_stripe_event_id=p_event_id,
      cancel_at_period_end=p_cancel_at_period_end,canceled_at=p_canceled_at,updated_at=NOW()
    WHERE id=existing.id;
  END IF;
  RETURN 'applied';
END;
$$;

CREATE OR REPLACE FUNCTION public.revoke_stripe_paid_access(
  p_event_id TEXT,
  p_event_type TEXT,
  p_event_created_at TIMESTAMPTZ,
  p_payment_intent_id TEXT DEFAULT NULL,
  p_subscription_id TEXT DEFAULT NULL,
  p_reason TEXT DEFAULT 'payment_reversed'
)
RETURNS TEXT
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  INSERT INTO public.stripe_webhook_events(stripe_event_id,event_type,event_created_at,resource_id)
  VALUES (p_event_id,p_event_type,p_event_created_at,COALESCE(p_subscription_id,p_payment_intent_id))
  ON CONFLICT (stripe_event_id) DO NOTHING;
  IF NOT FOUND THEN RETURN 'duplicate'; END IF;

  UPDATE public.subscriptions SET
    status='revoked',revoked_at=NOW(),revocation_reason=p_reason,
    current_period_end=LEAST(COALESCE(current_period_end,NOW()),NOW()),updated_at=NOW()
  WHERE (p_subscription_id IS NOT NULL AND stripe_subscription_id=p_subscription_id)
     OR (p_payment_intent_id IS NOT NULL AND stripe_payment_intent_id=p_payment_intent_id);
  RETURN 'applied';
END;
$$;

REVOKE ALL ON FUNCTION public.apply_stripe_checkout_event(TEXT,TEXT,TIMESTAMPTZ,UUID,TEXT,TEXT,TEXT,TEXT,TIMESTAMPTZ,TIMESTAMPTZ) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.apply_stripe_subscription_event(TEXT,TEXT,TIMESTAMPTZ,UUID,TEXT,TEXT,TEXT,TIMESTAMPTZ,TIMESTAMPTZ,BOOLEAN,TIMESTAMPTZ) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.revoke_stripe_paid_access(TEXT,TEXT,TIMESTAMPTZ,TEXT,TEXT,TEXT) FROM PUBLIC;
