-- OroTitan C8 SOLO_OWNER attestation candidate (SHADOW ONLY).
-- ADDITIVE, LOCAL CANDIDATE, NEVER execute on PRODUCTION.
-- This migration enables no lock, ratifies no amendment and accepts no milestone.
BEGIN;

DO $preflight$
BEGIN
 IF EXISTS(SELECT 1 FROM public.orotitan_j0_c8_acceptance_control_v1
           WHERE enabled OR activation_authority_ref IS NOT NULL)
 OR EXISTS(SELECT 1 FROM public.orotitan_j0_c8_milestone_activation_v1
           WHERE enabled OR authority_ref IS NOT NULL)
 OR EXISTS(SELECT 1 FROM public.orotitan_j0_c8_decisions_v1)
 THEN RAISE EXCEPTION 'C8_SOLO_MIGRATION_REQUIRES_CLOSED_UNUSED_GATES';END IF;
END $preflight$;

ALTER TABLE public.orotitan_j0_c8_acceptance_control_v1
 ADD COLUMN IF NOT EXISTS solo_owner_enabled boolean NOT NULL DEFAULT false,
 ADD COLUMN IF NOT EXISTS solo_amendment_ref text,
 ADD COLUMN IF NOT EXISTS solo_amendment_sha256 text;
ALTER TABLE public.orotitan_j0_c8_acceptance_control_v1
 DROP CONSTRAINT IF EXISTS j0_c8_solo_ratification_guard;
ALTER TABLE public.orotitan_j0_c8_acceptance_control_v1
 ADD CONSTRAINT j0_c8_solo_ratification_guard CHECK(
  (solo_owner_enabled=false AND solo_amendment_ref IS NULL AND solo_amendment_sha256 IS NULL)
  OR (solo_owner_enabled=true AND
   solo_amendment_ref IS NOT NULL AND solo_amendment_sha256 IS NOT NULL AND
   solo_amendment_ref ~ '^https://github[.]com/robzer13/indice_nexus/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$'
   AND solo_amendment_sha256 ~ '^[a-f0-9]{64}$'));

ALTER TABLE public.orotitan_j0_c8_reviews_v1
 ADD COLUMN IF NOT EXISTS governance_mode text NOT NULL DEFAULT 'STANDARD_TWO_PERSON',
 ADD COLUMN IF NOT EXISTS auth_provider text,
 ADD COLUMN IF NOT EXISTS owner_user_id uuid,
 ADD COLUMN IF NOT EXISTS amendment_ref text,
 ADD COLUMN IF NOT EXISTS amendment_sha256 text;
ALTER TABLE public.orotitan_j0_c8_reviews_v1
 ADD CONSTRAINT j0_c8_review_governance_guard CHECK(
  (governance_mode='STANDARD_TWO_PERSON' AND auth_provider IS NULL AND
   owner_user_id IS NULL AND amendment_ref IS NULL AND amendment_sha256 IS NULL)
  OR (governance_mode='SOLO_OWNER' AND milestone_code='J0' AND
   auth_provider IS NOT NULL AND auth_provider='SUPABASE_AUTH' AND
   owner_user_id IS NOT NULL AND amendment_ref IS NOT NULL AND amendment_sha256 IS NOT NULL AND
   amendment_ref ~ '^https://github[.]com/robzer13/indice_nexus/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$'
   AND amendment_sha256 ~ '^[a-f0-9]{64}$'));

ALTER TABLE public.orotitan_j0_c8_decisions_v1
 ADD COLUMN IF NOT EXISTS governance_mode text NOT NULL DEFAULT 'STANDARD_TWO_PERSON',
 ADD COLUMN IF NOT EXISTS auth_provider text,
 ADD COLUMN IF NOT EXISTS owner_user_id uuid,
 ADD COLUMN IF NOT EXISTS amendment_ref text,
 ADD COLUMN IF NOT EXISTS amendment_sha256 text,
 ADD COLUMN IF NOT EXISTS decision_sha256 text;
ALTER TABLE public.orotitan_j0_c8_decisions_v1
 ADD CONSTRAINT j0_c8_decision_governance_guard CHECK(
  (governance_mode='STANDARD_TWO_PERSON' AND auth_provider IS NULL AND
   owner_user_id IS NULL AND amendment_ref IS NULL AND amendment_sha256 IS NULL)
  OR (governance_mode='SOLO_OWNER' AND milestone_code='J0' AND
   auth_provider IS NOT NULL AND auth_provider='SUPABASE_AUTH' AND
   owner_user_id IS NOT NULL AND amendment_ref IS NOT NULL AND amendment_sha256 IS NOT NULL AND
   amendment_ref ~ '^https://github[.]com/robzer13/indice_nexus/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$'
   AND amendment_sha256 ~ '^[a-f0-9]{64}$' AND decision_sha256 IS NOT NULL
   AND decision_sha256 ~ '^[a-f0-9]{64}$'));

-- Overload used only by the SOLO route. The historic six-argument STANDARD
-- function remains byte-compatible and retains its distinct-person guard.
CREATE FUNCTION public.orotitan_j0_c8_prepare_review_v1(
 p_review_key text,p_milestone text,p_audit_ref text,p_audit_sha256 text,
 p_reviewer_label text,p_evidence_ids uuid[],p_governance_mode text,
 p_auth_provider text,p_owner_user_id uuid,p_amendment_ref text,
 p_amendment_sha256 text
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $$
DECLARE v_old public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 v_review public.orotitan_j0_c8_reviews_v1%ROWTYPE;v_status jsonb;
 v_watermark bigint;v_criteria jsonb;v_csha text;v_hash text;
 v_control public.orotitan_j0_c8_acceptance_control_v1%ROWTYPE;
BEGIN
 IF p_governance_mode IS DISTINCT FROM 'SOLO_OWNER' OR
  p_milestone IS DISTINCT FROM 'J0' OR
  p_auth_provider IS DISTINCT FROM 'SUPABASE_AUTH' OR p_owner_user_id IS NULL OR
  p_reviewer_label IS DISTINCT FROM ('supabase-auth:'||p_owner_user_id::text) OR
  p_review_key IS NULL OR p_review_key !~ '^orotitan:c8:review:[a-zA-Z0-9:_-]{8,124}$' OR
  p_audit_ref IS NULL OR p_audit_ref !~ '^https://github.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$' OR
  p_audit_sha256 IS NULL OR p_audit_sha256 !~ '^[a-f0-9]{64}$' OR
  p_amendment_ref IS NULL OR p_amendment_ref !~ '^https://github[.]com/robzer13/indice_nexus/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$' OR
  p_amendment_sha256 IS NULL OR p_amendment_sha256 !~ '^[a-f0-9]{64}$'
 THEN RAISE EXCEPTION 'J0_C8_SOLO_REVIEW_INPUT_INVALID';END IF;
 SELECT * INTO v_control FROM public.orotitan_j0_c8_acceptance_control_v1
 WHERE lock_name='GLOBAL_SHADOW';
 IF NOT FOUND OR v_control.solo_owner_enabled IS DISTINCT FROM true OR
  v_control.solo_amendment_ref IS DISTINCT FROM p_amendment_ref OR
  v_control.solo_amendment_sha256 IS DISTINCT FROM p_amendment_sha256
 THEN RAISE EXCEPTION 'J0_C8_SOLO_AMENDMENT_NOT_RATIFIED';END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('orotitan:j0:gate:J0',0));
 SELECT * INTO v_old FROM public.orotitan_j0_c8_reviews_v1 WHERE review_key=p_review_key;
 IF FOUND THEN
  IF v_old.milestone_code IS DISTINCT FROM p_milestone OR
   v_old.audit_ref IS DISTINCT FROM p_audit_ref OR
   v_old.audit_sha256 IS DISTINCT FROM p_audit_sha256 OR
   v_old.reviewer_label IS DISTINCT FROM p_reviewer_label OR
   v_old.evidence_event_ids IS DISTINCT FROM p_evidence_ids OR
   v_old.governance_mode IS DISTINCT FROM p_governance_mode OR
   v_old.auth_provider IS DISTINCT FROM p_auth_provider OR
   v_old.owner_user_id IS DISTINCT FROM p_owner_user_id OR
   v_old.amendment_ref IS DISTINCT FROM p_amendment_ref OR
   v_old.amendment_sha256 IS DISTINCT FROM p_amendment_sha256
  THEN RAISE EXCEPTION 'J0_C8_REVIEW_IDEMPOTENCY_CONFLICT';END IF;
  RETURN jsonb_build_object('status','REPLAYED','reviewId',v_old.review_id,
   'reviewSha256',v_old.review_sha256,'accepted',false,'governanceMode',v_old.governance_mode);
 END IF;
 v_status:=public.orotitan_j0_c8_gate_check_v1(p_milestone);
 IF NOT (v_status->>'eligibleForReview')::boolean THEN RAISE EXCEPTION 'J0_C8_GATE_NOT_READY';END IF;
 PERFORM public.orotitan_j0_c8_validate_proofs_v1(p_milestone,p_evidence_ids);
 SELECT gate_criteria INTO v_criteria FROM public.orotitan_j0_milestones_v1
 WHERE milestone_code=p_milestone;
 v_watermark:=(v_status->>'journalWatermark')::bigint;
 v_csha:=encode(extensions.digest(v_criteria::text,'sha256'),'hex');
 v_hash:=encode(extensions.digest(jsonb_build_object(
  'milestone',p_milestone,'auditRef',p_audit_ref,'auditSha256',p_audit_sha256,
  'reviewer',p_reviewer_label,'proofIds',to_jsonb(p_evidence_ids),
  'criteriaSha256',v_csha,'journalWatermark',v_watermark,
  'governanceMode',p_governance_mode,'authProvider',p_auth_provider,
  'ownerUserId',p_owner_user_id,'amendmentRef',p_amendment_ref,
  'amendmentSha256',p_amendment_sha256)::text,'sha256'),'hex');
 INSERT INTO public.orotitan_j0_c8_reviews_v1(
  review_key,milestone_code,audit_ref,audit_sha256,reviewer_label,
  evidence_event_ids,criteria_sha256,journal_watermark,review_sha256,
  governance_mode,auth_provider,owner_user_id,amendment_ref,amendment_sha256)
 VALUES(p_review_key,p_milestone,p_audit_ref,p_audit_sha256,p_reviewer_label,
  p_evidence_ids,v_csha,v_watermark,v_hash,p_governance_mode,p_auth_provider,
  p_owner_user_id,p_amendment_ref,p_amendment_sha256) RETURNING * INTO v_review;
 RETURN jsonb_build_object('status','PREPARED','reviewId',v_review.review_id,
  'reviewSha256',v_review.review_sha256,'journalWatermark',v_watermark,
  'accepted',false,'governanceMode',v_review.governance_mode);
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_prepare_review_v1(
 text,text,text,text,text,uuid[],text,text,uuid,text,text) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_prepare_review_v1(
 text,text,text,text,text,uuid[],text,text,uuid,text,text) TO service_role;

-- Replace the historic STANDARD overload as well: once the GLOBAL_SHADOW row
-- ratifies SOLO_OWNER, no six-argument caller may cross the governance boundary.
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_accept_review_v1(
 p_decision_key text,p_review_id uuid,p_review_sha256 text,
 p_approver_label text,p_confirmation text,p_authority_ref text
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $$
DECLARE v_lock public.orotitan_j0_c8_acceptance_control_v1%ROWTYPE;
 v_review public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 v_previous public.orotitan_j0_c8_decisions_v1%ROWTYPE;
 v_decision public.orotitan_j0_c8_decisions_v1%ROWTYPE;v_status jsonb;
 v_criteria jsonb;v_csha text;v_hash text;v_m_enabled boolean;v_m_authority text;
BEGIN
 IF p_decision_key IS NULL OR
  p_decision_key !~ '^orotitan:c8:decision:[a-zA-Z0-9:_-]{8,124}$' OR
  p_review_id IS NULL OR p_review_sha256 IS NULL OR p_review_sha256 !~ '^[a-f0-9]{64}$' OR
  p_approver_label IS NULL OR length(btrim(p_approver_label)) NOT BETWEEN 4 AND 100 OR
  p_confirmation IS NULL OR p_authority_ref IS NULL
 THEN RAISE EXCEPTION 'J0_C8_DECISION_INPUT_INVALID';END IF;
 SELECT * INTO v_lock FROM public.orotitan_j0_c8_acceptance_control_v1
 WHERE lock_name='GLOBAL_SHADOW' FOR UPDATE;
 IF NOT FOUND OR v_lock.enabled IS DISTINCT FROM true OR
  v_lock.activation_authority_ref IS NULL
 THEN RAISE EXCEPTION 'J0_C8_ACCEPTANCE_LOCKED';END IF;
 IF v_lock.solo_owner_enabled IS DISTINCT FROM false OR
  p_authority_ref ~ '^GO_ACCEPT_J0_SOLO_'
 THEN RAISE EXCEPTION 'J0_C8_GOVERNANCE_MODE_MISMATCH';END IF;
 IF p_authority_ref IS DISTINCT FROM v_lock.activation_authority_ref
 THEN RAISE EXCEPTION 'J0_C8_AUTHORITY_REF_INVALID';END IF;
 SELECT * INTO v_review FROM public.orotitan_j0_c8_reviews_v1 WHERE review_id=p_review_id;
 IF NOT FOUND OR v_review.review_sha256 IS DISTINCT FROM p_review_sha256
 THEN RAISE EXCEPTION 'J0_C8_REVIEW_NOT_FOUND_OR_HASH_MISMATCH';END IF;
 IF v_review.governance_mode IS DISTINCT FROM 'STANDARD_TWO_PERSON'
 THEN RAISE EXCEPTION 'J0_C8_GOVERNANCE_MODE_MISMATCH';END IF;
 IF lower(btrim(v_review.reviewer_label))=lower(btrim(p_approver_label))
 THEN RAISE EXCEPTION 'J0_C8_REVIEWER_APPROVER_NOT_DISTINCT';END IF;
 SELECT enabled,authority_ref INTO v_m_enabled,v_m_authority
 FROM public.orotitan_j0_c8_milestone_activation_v1
 WHERE milestone_code=v_review.milestone_code FOR UPDATE;
 IF NOT COALESCE(v_m_enabled,false)
 THEN RAISE EXCEPTION 'J0_C8_MILESTONE_ACCEPTANCE_LOCKED';END IF;
 IF p_authority_ref IS DISTINCT FROM v_m_authority
 THEN RAISE EXCEPTION 'J0_C8_MILESTONE_AUTHORITY_REF_INVALID';END IF;
 IF p_confirmation IS DISTINCT FROM ('ACCEPTER '||v_review.milestone_code||' SHADOW')
 THEN RAISE EXCEPTION 'J0_C8_EXPLICIT_CONFIRMATION_REQUIRED';END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('orotitan:j0:gate:'||v_review.milestone_code,0));
 SELECT * INTO v_previous FROM public.orotitan_j0_c8_decisions_v1
 WHERE decision_key=p_decision_key;
 IF FOUND THEN
  IF v_previous.review_id IS DISTINCT FROM p_review_id OR
   v_previous.review_sha256 IS DISTINCT FROM p_review_sha256 OR
   v_previous.approver_label IS DISTINCT FROM btrim(p_approver_label) OR
   v_previous.consent_phrase IS DISTINCT FROM p_confirmation OR
   v_previous.authority_ref IS DISTINCT FROM p_authority_ref OR
   v_previous.governance_mode IS DISTINCT FROM 'STANDARD_TWO_PERSON'
  THEN RAISE EXCEPTION 'J0_C8_DECISION_IDEMPOTENCY_CONFLICT';END IF;
  RETURN jsonb_build_object('status','REPLAYED','decisionId',v_previous.decision_id,
   'accepted',true);
 END IF;
 v_status:=public.orotitan_j0_c8_gate_check_v1(v_review.milestone_code);
 IF NOT (v_status->>'eligibleForReview')::boolean
 THEN RAISE EXCEPTION 'J0_C8_GATE_NOT_READY';END IF;
 IF (v_status->>'journalWatermark')::bigint IS DISTINCT FROM v_review.journal_watermark
 THEN RAISE EXCEPTION 'J0_C8_REVIEW_STALE';END IF;
 PERFORM public.orotitan_j0_c8_validate_proofs_v1(
  v_review.milestone_code,v_review.evidence_event_ids);
 SELECT gate_criteria INTO v_criteria FROM public.orotitan_j0_milestones_v1
 WHERE milestone_code=v_review.milestone_code;
 v_csha:=encode(extensions.digest(v_criteria::text,'sha256'),'hex');
 IF v_csha IS DISTINCT FROM v_review.criteria_sha256
 THEN RAISE EXCEPTION 'J0_C8_CRITERIA_CHANGED';END IF;
 v_hash:=encode(extensions.digest(jsonb_build_object(
  'milestone',v_review.milestone_code,'auditRef',v_review.audit_ref,
  'auditSha256',v_review.audit_sha256,'reviewer',v_review.reviewer_label,
  'proofIds',to_jsonb(v_review.evidence_event_ids),
  'criteriaSha256',v_review.criteria_sha256,
  'journalWatermark',v_review.journal_watermark)::text,'sha256'),'hex');
 IF v_hash IS DISTINCT FROM v_review.review_sha256
 THEN RAISE EXCEPTION 'J0_C8_REVIEW_HASH_INTEGRITY_FAILURE';END IF;
 INSERT INTO public.orotitan_j0_c8_decisions_v1(
  decision_key,milestone_code,review_id,review_sha256,approver_label,
  consent_phrase,authority_ref,governance_mode)
 VALUES(p_decision_key,v_review.milestone_code,p_review_id,p_review_sha256,
  btrim(p_approver_label),p_confirmation,p_authority_ref,'STANDARD_TWO_PERSON')
 RETURNING * INTO v_decision;
 RETURN jsonb_build_object('status','ACCEPTED','decisionId',v_decision.decision_id,
  'milestone',v_decision.milestone_code,'accepted',true,
  'stageStarted',false,'published',false);
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_accept_review_v1(
 text,uuid,text,text,text,text) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_accept_review_v1(
 text,uuid,text,text,text,text) TO service_role;

CREATE FUNCTION public.orotitan_j0_c8_accept_review_v1(
 p_decision_key text,p_review_id uuid,p_review_sha256 text,
 p_approver_label text,p_confirmation text,p_authority_ref text,
 p_governance_mode text,p_auth_provider text,p_owner_user_id uuid,
 p_amendment_ref text,p_amendment_sha256 text
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $$
DECLARE v_lock public.orotitan_j0_c8_acceptance_control_v1%ROWTYPE;
 v_review public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 v_previous public.orotitan_j0_c8_decisions_v1%ROWTYPE;
 v_decision public.orotitan_j0_c8_decisions_v1%ROWTYPE;v_status jsonb;
 v_criteria jsonb;v_csha text;v_review_hash text;v_decision_hash text;
 v_m_enabled boolean;v_m_authority text;
BEGIN
 IF p_governance_mode IS DISTINCT FROM 'SOLO_OWNER' OR
  p_auth_provider IS DISTINCT FROM 'SUPABASE_AUTH' OR p_owner_user_id IS NULL OR
  p_approver_label IS DISTINCT FROM ('supabase-auth:'||p_owner_user_id::text) OR
  p_decision_key IS NULL OR p_decision_key !~ '^orotitan:c8:decision:[a-zA-Z0-9:_-]{8,124}$' OR
  p_review_id IS NULL OR p_review_sha256 IS NULL OR p_review_sha256 !~ '^[a-f0-9]{64}$' OR
  p_confirmation IS DISTINCT FROM 'ACCEPTER J0 SHADOW EN MODE SOLO' OR
  p_authority_ref IS NULL OR p_authority_ref !~ '^GO_ACCEPT_J0_SOLO_[A-Z0-9_]{8,115}$' OR
  p_amendment_ref IS NULL OR p_amendment_ref !~ '^https://github[.]com/robzer13/indice_nexus/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$' OR
  p_amendment_sha256 IS NULL OR p_amendment_sha256 !~ '^[a-f0-9]{64}$'
 THEN RAISE EXCEPTION 'J0_C8_SOLO_DECISION_INPUT_INVALID';END IF;
 SELECT * INTO v_lock FROM public.orotitan_j0_c8_acceptance_control_v1
 WHERE lock_name='GLOBAL_SHADOW' FOR UPDATE;
 IF NOT FOUND OR v_lock.enabled IS DISTINCT FROM true OR
  v_lock.activation_authority_ref IS DISTINCT FROM p_authority_ref THEN RAISE EXCEPTION 'J0_C8_ACCEPTANCE_LOCKED';END IF;
 IF v_lock.solo_owner_enabled IS DISTINCT FROM true OR
  v_lock.solo_amendment_ref IS DISTINCT FROM p_amendment_ref OR
  v_lock.solo_amendment_sha256 IS DISTINCT FROM p_amendment_sha256
 THEN RAISE EXCEPTION 'J0_C8_SOLO_AMENDMENT_NOT_RATIFIED';END IF;
 SELECT * INTO v_review FROM public.orotitan_j0_c8_reviews_v1 WHERE review_id=p_review_id;
 IF NOT FOUND OR v_review.review_sha256 IS DISTINCT FROM p_review_sha256 THEN
  RAISE EXCEPTION 'J0_C8_REVIEW_NOT_FOUND_OR_HASH_MISMATCH';END IF;
 IF v_review.milestone_code IS DISTINCT FROM 'J0' OR
  v_review.governance_mode IS DISTINCT FROM p_governance_mode OR
  v_review.auth_provider IS DISTINCT FROM p_auth_provider OR
  v_review.owner_user_id IS DISTINCT FROM p_owner_user_id OR
  v_review.amendment_ref IS DISTINCT FROM p_amendment_ref OR
  v_review.amendment_sha256 IS DISTINCT FROM p_amendment_sha256 OR
  v_review.reviewer_label IS DISTINCT FROM p_approver_label
 THEN RAISE EXCEPTION 'J0_C8_SOLO_OWNER_ATTESTATION_MISMATCH';END IF;
 SELECT enabled,authority_ref INTO v_m_enabled,v_m_authority
 FROM public.orotitan_j0_c8_milestone_activation_v1 WHERE milestone_code='J0' FOR UPDATE;
 IF NOT COALESCE(v_m_enabled,false) THEN RAISE EXCEPTION 'J0_C8_MILESTONE_ACCEPTANCE_LOCKED';END IF;
 IF v_m_authority IS DISTINCT FROM p_authority_ref THEN RAISE EXCEPTION 'J0_C8_MILESTONE_AUTHORITY_REF_INVALID';END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('orotitan:j0:gate:J0',0));
 SELECT * INTO v_previous FROM public.orotitan_j0_c8_decisions_v1 WHERE decision_key=p_decision_key;
 IF FOUND THEN
  IF v_previous.review_id IS DISTINCT FROM p_review_id OR
   v_previous.review_sha256 IS DISTINCT FROM p_review_sha256 OR
   v_previous.approver_label IS DISTINCT FROM p_approver_label OR
   v_previous.consent_phrase IS DISTINCT FROM p_confirmation OR
   v_previous.authority_ref IS DISTINCT FROM p_authority_ref OR
   v_previous.governance_mode IS DISTINCT FROM p_governance_mode OR
   v_previous.owner_user_id IS DISTINCT FROM p_owner_user_id OR
   v_previous.amendment_sha256 IS DISTINCT FROM p_amendment_sha256
  THEN RAISE EXCEPTION 'J0_C8_DECISION_IDEMPOTENCY_CONFLICT';END IF;
  RETURN jsonb_build_object('status','REPLAYED','decisionId',v_previous.decision_id,
   'milestone','J0','accepted',true,'governanceMode',v_previous.governance_mode,
   'decisionSha256',v_previous.decision_sha256);
 END IF;
 v_status:=public.orotitan_j0_c8_gate_check_v1('J0');
 IF NOT (v_status->>'eligibleForReview')::boolean THEN RAISE EXCEPTION 'J0_C8_GATE_NOT_READY';END IF;
 IF (v_status->>'journalWatermark')::bigint<>v_review.journal_watermark THEN RAISE EXCEPTION 'J0_C8_REVIEW_STALE';END IF;
 PERFORM public.orotitan_j0_c8_validate_proofs_v1('J0',v_review.evidence_event_ids);
 SELECT gate_criteria INTO v_criteria FROM public.orotitan_j0_milestones_v1 WHERE milestone_code='J0';
 v_csha:=encode(extensions.digest(v_criteria::text,'sha256'),'hex');
 IF v_csha<>v_review.criteria_sha256 THEN RAISE EXCEPTION 'J0_C8_CRITERIA_CHANGED';END IF;
 v_review_hash:=encode(extensions.digest(jsonb_build_object(
  'milestone','J0','auditRef',v_review.audit_ref,'auditSha256',v_review.audit_sha256,
  'reviewer',v_review.reviewer_label,'proofIds',to_jsonb(v_review.evidence_event_ids),
  'criteriaSha256',v_review.criteria_sha256,'journalWatermark',v_review.journal_watermark,
  'governanceMode',v_review.governance_mode,'authProvider',v_review.auth_provider,
  'ownerUserId',v_review.owner_user_id,'amendmentRef',v_review.amendment_ref,
  'amendmentSha256',v_review.amendment_sha256)::text,'sha256'),'hex');
 IF v_review_hash<>v_review.review_sha256 THEN RAISE EXCEPTION 'J0_C8_REVIEW_HASH_INTEGRITY_FAILURE';END IF;
 v_decision_hash:=encode(extensions.digest(jsonb_build_object(
  'reviewId',p_review_id,'reviewSha256',p_review_sha256,'approver',p_approver_label,
  'consent',p_confirmation,'authorityRef',p_authority_ref,
  'governanceMode',p_governance_mode,'authProvider',p_auth_provider,
  'ownerUserId',p_owner_user_id,'amendmentRef',p_amendment_ref,
  'amendmentSha256',p_amendment_sha256)::text,'sha256'),'hex');
 INSERT INTO public.orotitan_j0_c8_decisions_v1(
  decision_key,milestone_code,review_id,review_sha256,approver_label,consent_phrase,
  authority_ref,governance_mode,auth_provider,owner_user_id,amendment_ref,
  amendment_sha256,decision_sha256)
 VALUES(p_decision_key,'J0',p_review_id,p_review_sha256,p_approver_label,p_confirmation,
  p_authority_ref,p_governance_mode,p_auth_provider,p_owner_user_id,p_amendment_ref,
  p_amendment_sha256,v_decision_hash) RETURNING * INTO v_decision;
 RETURN jsonb_build_object('status','ACCEPTED','decisionId',v_decision.decision_id,
  'milestone','J0','accepted',true,'governanceMode',v_decision.governance_mode,
  'decisionSha256',v_decision.decision_sha256,'stageStarted',false,'published',false);
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_accept_review_v1(
 text,uuid,text,text,text,text,text,text,uuid,text,text) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_accept_review_v1(
 text,uuid,text,text,text,text,text,text,uuid,text,text) TO service_role;

COMMIT;
