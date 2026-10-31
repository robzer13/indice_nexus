-- Per-milestone authorization gate for OroTitan C8 (SHADOW ONLY).
-- A global kill-switch alone must not activate future J1-J8 approvals.
-- No gate is enabled, no milestone accepted, no run/stage/publish mutation.
BEGIN;
-- Retain the existing, stricter global kill switch unchanged.
CREATE TABLE IF NOT EXISTS public.orotitan_j0_c8_milestone_activation_v1(
 milestone_code text PRIMARY KEY REFERENCES
  public.orotitan_j0_milestones_v1(milestone_code) ON DELETE RESTRICT,
 enabled boolean NOT NULL DEFAULT false,
 authority_ref text,
 created_at timestamptz NOT NULL DEFAULT now(),
 CONSTRAINT j0_c8_milestone_authority_match CHECK(
  (enabled=false AND authority_ref IS NULL) OR
  (enabled=true AND authority_ref ~
   ('^GO_ACCEPT_'||milestone_code||'_[A-Z0-9_]{8,120}$'))
 )
);
ALTER TABLE public.orotitan_j0_c8_milestone_activation_v1 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.orotitan_j0_c8_milestone_activation_v1
 FROM PUBLIC,anon,authenticated,service_role;
GRANT SELECT ON public.orotitan_j0_c8_milestone_activation_v1 TO service_role;
INSERT INTO public.orotitan_j0_c8_milestone_activation_v1(milestone_code,enabled)
 SELECT milestone_code,false FROM public.orotitan_j0_milestones_v1
 ON CONFLICT(milestone_code) DO NOTHING;
-- Both global + exact milestone authorization, after immutable review readback.
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_accept_review_v1(
 p_decision_key text,p_review_id uuid,p_review_sha256 text,
 p_approver_label text,p_confirmation text,p_authority_ref text
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public,extensions AS $$
DECLARE v_lock public.orotitan_j0_c8_acceptance_control_v1%ROWTYPE;
 v_review public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 v_previous public.orotitan_j0_c8_decisions_v1%ROWTYPE;
 v_decision public.orotitan_j0_c8_decisions_v1%ROWTYPE;
 v_status jsonb;v_criteria jsonb;v_csha text;v_hash text;
 v_m_enabled boolean;v_m_authority text;
BEGIN
 IF p_decision_key IS NULL OR
 p_decision_key !~ '^orotitan:c8:decision:[a-zA-Z0-9:_-]{8,124}$'
 OR p_review_id IS NULL OR p_review_sha256 !~ '^[a-f0-9]{64}$'
 OR p_approver_label IS NULL OR length(btrim(p_approver_label)) NOT BETWEEN 4 AND 100
 OR p_confirmation IS NULL OR p_authority_ref IS NULL
 THEN RAISE EXCEPTION 'J0_C8_DECISION_INPUT_INVALID';END IF;
 SELECT * INTO v_lock FROM public.orotitan_j0_c8_acceptance_control_v1
 WHERE lock_name='GLOBAL_SHADOW' FOR UPDATE;
 IF NOT FOUND OR v_lock.enabled IS DISTINCT FROM true OR
    v_lock.activation_authority_ref IS NULL THEN
  RAISE EXCEPTION 'J0_C8_ACCEPTANCE_LOCKED';END IF;
 IF p_authority_ref<>v_lock.activation_authority_ref THEN
  RAISE EXCEPTION 'J0_C8_AUTHORITY_REF_INVALID';END IF;
 SELECT * INTO v_review FROM public.orotitan_j0_c8_reviews_v1
 WHERE review_id=p_review_id;
 IF NOT FOUND OR v_review.review_sha256<>p_review_sha256 THEN
  RAISE EXCEPTION 'J0_C8_REVIEW_NOT_FOUND_OR_HASH_MISMATCH';END IF;
 SELECT enabled,authority_ref INTO v_m_enabled,v_m_authority
 FROM public.orotitan_j0_c8_milestone_activation_v1
 WHERE milestone_code=v_review.milestone_code FOR UPDATE;
 IF NOT COALESCE(v_m_enabled,false) THEN
  RAISE EXCEPTION 'J0_C8_MILESTONE_ACCEPTANCE_LOCKED';END IF;
 IF p_authority_ref<>v_m_authority THEN
  RAISE EXCEPTION 'J0_C8_MILESTONE_AUTHORITY_REF_INVALID';END IF;
 IF p_confirmation<>('ACCEPTER '||v_review.milestone_code||' SHADOW') THEN
  RAISE EXCEPTION 'J0_C8_EXPLICIT_CONFIRMATION_REQUIRED';END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(
 pg_catalog.hashtextextended('orotitan:j0:gate:'||v_review.milestone_code,0));
 SELECT * INTO v_previous FROM public.orotitan_j0_c8_decisions_v1
 WHERE decision_key=p_decision_key;
 IF FOUND THEN
  IF v_previous.review_id<>p_review_id OR
     v_previous.review_sha256<>p_review_sha256 OR
     v_previous.approver_label<>btrim(p_approver_label) OR
     v_previous.consent_phrase<>p_confirmation OR
     v_previous.authority_ref<>p_authority_ref THEN
    RAISE EXCEPTION 'J0_C8_DECISION_IDEMPOTENCY_CONFLICT';END IF;
  RETURN jsonb_build_object('status','REPLAYED',
    'decisionId',v_previous.decision_id,'accepted',true);
 END IF;
 v_status:=public.orotitan_j0_c8_gate_check_v1(v_review.milestone_code);
 IF NOT (v_status->>'eligibleForReview')::boolean THEN
  RAISE EXCEPTION 'J0_C8_GATE_NOT_READY';END IF;
 IF (v_status->>'journalWatermark')::bigint<>v_review.journal_watermark THEN
  RAISE EXCEPTION 'J0_C8_REVIEW_STALE';END IF;
 PERFORM public.orotitan_j0_c8_validate_proofs_v1(
 v_review.milestone_code,v_review.evidence_event_ids);
 SELECT gate_criteria INTO v_criteria FROM public.orotitan_j0_milestones_v1
 WHERE milestone_code=v_review.milestone_code;
 v_csha:=encode(extensions.digest(v_criteria::text,'sha256'),'hex');
 IF v_csha<>v_review.criteria_sha256 THEN
  RAISE EXCEPTION 'J0_C8_CRITERIA_CHANGED';END IF;
 v_hash:=encode(extensions.digest(
 jsonb_build_object('milestone',v_review.milestone_code,
 'auditRef',v_review.audit_ref,'auditSha256',v_review.audit_sha256,
 'reviewer',v_review.reviewer_label,'proofIds',to_jsonb(v_review.evidence_event_ids),
 'criteriaSha256',v_review.criteria_sha256,
 'journalWatermark',v_review.journal_watermark)::text,'sha256'),'hex');
 IF v_hash<>v_review.review_sha256 THEN
  RAISE EXCEPTION 'J0_C8_REVIEW_HASH_INTEGRITY_FAILURE';END IF;
 INSERT INTO public.orotitan_j0_c8_decisions_v1(
 decision_key,milestone_code,review_id,review_sha256,approver_label,
 consent_phrase,authority_ref)
 VALUES(p_decision_key,v_review.milestone_code,p_review_id,p_review_sha256,
  btrim(p_approver_label),p_confirmation,p_authority_ref)
 RETURNING * INTO v_decision;
 RETURN jsonb_build_object('status','ACCEPTED','decisionId',v_decision.decision_id,
  'milestone',v_decision.milestone_code,'accepted',true,
  'stageStarted',false,'published',false);
END $$;

REVOKE ALL ON FUNCTION public.orotitan_j0_c8_accept_review_v1(
 text,uuid,text,text,text,text) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_accept_review_v1(
 text,uuid,text,text,text,text) TO service_role;
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_read_gate_v1(p_milestone text)
RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog,public AS $$
DECLARE v_status jsonb;v_lock boolean;v_mlock boolean;v_review_count bigint;
BEGIN
 v_status:=public.orotitan_j0_c8_gate_check_v1(p_milestone);
 SELECT enabled INTO v_lock FROM public.orotitan_j0_c8_acceptance_control_v1
 WHERE lock_name='GLOBAL_SHADOW';
 SELECT enabled INTO v_mlock
 FROM public.orotitan_j0_c8_milestone_activation_v1
 WHERE milestone_code=p_milestone;
 SELECT COUNT(*) INTO v_review_count FROM public.orotitan_j0_c8_reviews_v1
 WHERE milestone_code=p_milestone;
 RETURN v_status||jsonb_build_object('acceptanceEnabled',(COALESCE(v_lock,false) AND COALESCE(v_mlock,false)),
 'preparedReviews',v_review_count,
 'accepted',(v_status->>'alreadyAccepted')::boolean,
 'acceptanceAllowedNow',(COALESCE(v_lock,false)
  AND COALESCE(v_mlock,false)
  AND (v_status->>'eligibleForReview')::boolean));
END $$;

REVOKE ALL ON FUNCTION public.orotitan_j0_c8_read_gate_v1(text)
 FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_read_gate_v1(text)
 TO service_role;
COMMIT;
