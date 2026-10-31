-- OroTitan Run Factory J0 C8 human acceptance gate (SHADOW ONLY).
-- Acceptance is HARD LOCKED until a separate operator GO and privileged
-- migration explicitly activate the control row. This migration NEVER
-- accepts any milestone and never touches run/stage/snapshot/publication.
BEGIN;

CREATE TABLE IF NOT EXISTS public.orotitan_j0_c8_acceptance_control_v1 (
  lock_name text PRIMARY KEY CHECK(lock_name='GLOBAL_SHADOW'),
  enabled boolean NOT NULL DEFAULT false,
  activation_authority_ref text,
  created_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT j0_c8_disabled_without_authority CHECK(
    (enabled=false AND activation_authority_ref IS NULL) OR
    (enabled=true AND activation_authority_ref ~ '^GO_ACCEPT_J0_[A-Z0-9_]{8,120}$')
  )
);
INSERT INTO public.orotitan_j0_c8_acceptance_control_v1(lock_name,enabled)
VALUES ('GLOBAL_SHADOW',false) ON CONFLICT(lock_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS public.orotitan_j0_c8_reviews_v1 (
  review_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  review_key text NOT NULL UNIQUE CHECK(review_key ~ '^orotitan:c8:review:[a-zA-Z0-9:_-]{8,124}$'),
  milestone_code text NOT NULL REFERENCES public.orotitan_j0_milestones_v1(milestone_code) ON DELETE RESTRICT,
  audit_ref text NOT NULL CHECK(audit_ref ~ '^https://github.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$'),
  audit_sha256 text NOT NULL CHECK(audit_sha256 ~ '^[a-f0-9]{64}$'),
  reviewer_label text NOT NULL CHECK(length(btrim(reviewer_label)) BETWEEN 4 AND 100),
  evidence_event_ids uuid[] NOT NULL CHECK(array_length(evidence_event_ids,1)=7),
  criteria_sha256 text NOT NULL CHECK(criteria_sha256 ~ '^[a-f0-9]{64}$'),
  journal_watermark bigint NOT NULL CHECK(journal_watermark>=1),
  review_sha256 text NOT NULL UNIQUE CHECK(review_sha256 ~ '^[a-f0-9]{64}$'),
  created_at timestamptz NOT NULL DEFAULT now(),
  scope text NOT NULL DEFAULT 'SHADOW_ONLY' CHECK(scope='SHADOW_ONLY')
);

CREATE TABLE IF NOT EXISTS public.orotitan_j0_c8_decisions_v1 (
  decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  decision_key text NOT NULL UNIQUE CHECK(decision_key ~ '^orotitan:c8:decision:[a-zA-Z0-9:_-]{8,124}$'),
  milestone_code text NOT NULL UNIQUE REFERENCES public.orotitan_j0_milestones_v1(milestone_code) ON DELETE RESTRICT,
  review_id uuid NOT NULL UNIQUE REFERENCES public.orotitan_j0_c8_reviews_v1(review_id) ON DELETE RESTRICT,
  review_sha256 text NOT NULL CHECK(review_sha256 ~ '^[a-f0-9]{64}$'),
  approver_label text NOT NULL CHECK(length(btrim(approver_label)) BETWEEN 4 AND 100),
  consent_phrase text NOT NULL,
  authority_ref text NOT NULL,
  decision_status text NOT NULL DEFAULT 'ACCEPTED' CHECK(decision_status='ACCEPTED'),
  scope text NOT NULL DEFAULT 'SHADOW_ONLY' CHECK(scope='SHADOW_ONLY'),
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.orotitan_j0_c8_acceptance_control_v1 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.orotitan_j0_c8_reviews_v1 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.orotitan_j0_c8_decisions_v1 ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.orotitan_j0_c8_acceptance_control_v1 FROM PUBLIC,anon,authenticated,service_role;
REVOKE ALL ON public.orotitan_j0_c8_reviews_v1 FROM PUBLIC,anon,authenticated,service_role;
REVOKE ALL ON public.orotitan_j0_c8_decisions_v1 FROM PUBLIC,anon,authenticated,service_role;
GRANT SELECT ON public.orotitan_j0_c8_acceptance_control_v1 TO service_role;
GRANT SELECT ON public.orotitan_j0_c8_reviews_v1 TO service_role;
GRANT SELECT ON public.orotitan_j0_c8_decisions_v1 TO service_role;

DROP TRIGGER IF EXISTS orotitan_j0_c8_reviews_append_only ON public.orotitan_j0_c8_reviews_v1;
CREATE TRIGGER orotitan_j0_c8_reviews_append_only BEFORE UPDATE OR DELETE
ON public.orotitan_j0_c8_reviews_v1 FOR EACH ROW
EXECUTE FUNCTION public.orotitan_j0_deny_history_change_v1();
DROP TRIGGER IF EXISTS orotitan_j0_c8_decisions_append_only ON public.orotitan_j0_c8_decisions_v1;
CREATE TRIGGER orotitan_j0_c8_decisions_append_only BEFORE UPDATE OR DELETE
ON public.orotitan_j0_c8_decisions_v1 FOR EACH ROW
EXECUTE FUNCTION public.orotitan_j0_deny_history_change_v1();

-- The check never mutates state; C8 is represented by the distinct,
-- explicit acceptance authorization (not a self-referential 8th proof).
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_gate_check_v1(p_milestone text)
RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog,public AS $$
DECLARE v_criteria jsonb;v_i integer;v_missing integer[]:='{}';v_watermark bigint;
 v_open bigint;v_accepted bigint;v_count bigint;v_qual text;
BEGIN
 SELECT gate_criteria INTO v_criteria FROM public.orotitan_j0_milestones_v1
 WHERE milestone_code=p_milestone AND scope='SHADOW_ONLY';
 IF v_criteria IS NULL OR jsonb_typeof(v_criteria)<>'array'
 OR jsonb_array_length(v_criteria)<>8 THEN RAISE EXCEPTION 'J0_C8_GATE_SCHEMA_INVALID';END IF;
 FOR v_i IN 1..7 LOOP
  v_qual:=v_i::text;
  SELECT COUNT(*) INTO v_count FROM public.orotitan_j0_events_v1
  WHERE milestone_code=p_milestone AND event_type='EVIDENCE_RECORDED'
    AND operator_details->>'criterionIndex'=v_qual
    AND (p_milestone<>'J0' OR v_i NOT IN (1,6,7) OR
         verification_state='VERIFIED_DIRECT_REGISTRY_READBACK');
  IF v_count=0 THEN v_missing:=array_append(v_missing,v_i);END IF;
 END LOOP;
 SELECT COALESCE(MAX(event_seq),0) INTO v_watermark
 FROM public.orotitan_j0_events_v1 WHERE milestone_code=p_milestone;
 SELECT COUNT(*) INTO v_open FROM (
   SELECT DISTINCT ON(evidence_ref) evidence_ref,event_type
   FROM public.orotitan_j0_events_v1
   WHERE milestone_code=p_milestone AND event_type IN ('BLOCKER_OPENED','BLOCKER_RESOLVED')
   ORDER BY evidence_ref,event_seq DESC
 ) recent WHERE event_type='BLOCKER_OPENED';
 SELECT COUNT(*) INTO v_accepted FROM public.orotitan_j0_c8_decisions_v1
 WHERE milestone_code=p_milestone AND decision_status='ACCEPTED';
 RETURN jsonb_build_object(
   'milestone',p_milestone,'scope','SHADOW_ONLY',
   'criteriaCount',8,'evidenceCriteriaChecked',7,
   'missingEvidenceIndices',to_jsonb(v_missing),
   'openBlockers',v_open,'journalWatermark',v_watermark,
   'alreadyAccepted',v_accepted>0,
   'eligibleForReview',(cardinality(v_missing)=0 AND v_open=0 AND v_accepted=0),
   'humanAcceptanceRequired',true
 );
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_gate_check_v1(text)
FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_gate_check_v1(text) TO service_role;

-- Verify specific pinned evidence, its exact criterion index and,
-- for J0's C1/C6/C7, the actual Registry-revalidated proof class.
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_validate_proofs_v1(
 p_milestone text,p_ids uuid[]
) RETURNS void LANGUAGE plpgsql SECURITY INVOKER
 SET search_path=pg_catalog,public AS $$
DECLARE i integer;v_ok bigint;
BEGIN
 IF p_ids IS NULL OR array_ndims(p_ids)<>1 OR array_length(p_ids,1)<>7 THEN
  RAISE EXCEPTION 'J0_C8_PROOF_VECTOR_INVALID';END IF;
 FOR i IN 1..7 LOOP
  SELECT COUNT(*) INTO v_ok FROM public.orotitan_j0_events_v1
  WHERE event_id=p_ids[i] AND milestone_code=p_milestone
    AND event_type='EVIDENCE_RECORDED' AND
    operator_details->>'criterionIndex'=i::text
    AND (p_milestone<>'J0' OR i NOT IN (1,6,7) OR
      (verification_state='VERIFIED_DIRECT_REGISTRY_READBACK'
       AND verified_snapshot IS NOT NULL));
  IF v_ok<>1 THEN RAISE EXCEPTION 'J0_C8_PROOF_MISSING_OR_INVALID_C%',i;END IF;
 END LOOP;
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_validate_proofs_v1(text,uuid[])
FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_validate_proofs_v1(text,uuid[])
TO service_role;

-- Prepares a frozen review, NEVER a decision. Service caller may not insert
-- directly: privileged insert is restricted to this validated RPC.
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_prepare_review_v1(
 p_review_key text,p_milestone text,p_audit_ref text,p_audit_sha256 text,
 p_reviewer_label text,p_evidence_ids uuid[]
) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public,extensions AS $$
DECLARE v_old public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 v_review public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 v_status jsonb;v_watermark bigint;v_criteria jsonb;
 v_csha text;v_hash text;
BEGIN
 IF p_review_key IS NULL OR
 p_review_key !~ '^orotitan:c8:review:[a-zA-Z0-9:_-]{8,124}$' OR
 p_milestone !~ '^J[0-8]$' OR
 p_audit_ref !~ '^https://github.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+/blob/[a-f0-9]{40}/docs/audits/[a-zA-Z0-9_.-]+[.]md$' OR
 p_audit_sha256 !~ '^[a-f0-9]{64}$' OR
 p_reviewer_label IS NULL OR length(btrim(p_reviewer_label)) NOT BETWEEN 4 AND 100
 THEN RAISE EXCEPTION 'J0_C8_REVIEW_INPUT_INVALID';END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(
  pg_catalog.hashtextextended('orotitan:j0:gate:'||p_milestone,0));
 SELECT * INTO v_old FROM public.orotitan_j0_c8_reviews_v1 WHERE review_key=p_review_key;
 IF FOUND THEN
  IF v_old.milestone_code<>p_milestone OR v_old.audit_ref<>p_audit_ref
    OR v_old.audit_sha256<>p_audit_sha256 OR
    v_old.reviewer_label<>btrim(p_reviewer_label) OR
    v_old.evidence_event_ids<>p_evidence_ids THEN
    RAISE EXCEPTION 'J0_C8_REVIEW_IDEMPOTENCY_CONFLICT';END IF;
  RETURN jsonb_build_object('status','REPLAYED','reviewId',v_old.review_id,
    'reviewSha256',v_old.review_sha256,'accepted',false);
 END IF;
 v_status:=public.orotitan_j0_c8_gate_check_v1(p_milestone);
 IF NOT (v_status->>'eligibleForReview')::boolean THEN
  RAISE EXCEPTION 'J0_C8_GATE_NOT_READY';END IF;
 PERFORM public.orotitan_j0_c8_validate_proofs_v1(p_milestone,p_evidence_ids);
 SELECT gate_criteria INTO v_criteria FROM public.orotitan_j0_milestones_v1
 WHERE milestone_code=p_milestone;
 v_watermark:=(v_status->>'journalWatermark')::bigint;
 v_csha:=encode(extensions.digest(v_criteria::text,'sha256'),'hex');
 v_hash:=encode(extensions.digest(
 jsonb_build_object('milestone',p_milestone,'auditRef',p_audit_ref,
 'auditSha256',p_audit_sha256,'reviewer',btrim(p_reviewer_label),
 'proofIds',to_jsonb(p_evidence_ids),'criteriaSha256',v_csha,
 'journalWatermark',v_watermark)::text,'sha256'),'hex');
 INSERT INTO public.orotitan_j0_c8_reviews_v1(
  review_key,milestone_code,audit_ref,audit_sha256,reviewer_label,
  evidence_event_ids,criteria_sha256,journal_watermark,review_sha256)
 VALUES(p_review_key,p_milestone,p_audit_ref,p_audit_sha256,btrim(p_reviewer_label),
  p_evidence_ids,v_csha,v_watermark,v_hash)
 RETURNING * INTO v_review;
 RETURN jsonb_build_object('status','PREPARED','reviewId',v_review.review_id,
  'reviewSha256',v_review.review_sha256,'journalWatermark',v_watermark,
  'accepted',false);
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_prepare_review_v1(
 text,text,text,text,text,uuid[]) FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_prepare_review_v1(
 text,text,text,text,text,uuid[]) TO service_role;

-- This is the future acceptance capability. SHADOW control.enabled=false.
-- It requires a DISTINCT later GO encoded by a privileged migration:
-- service_role has no UPDATE on control, so cannot turn the lock on.
-- Never update milestones.initial_status or start downstream stages.
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

-- Generic J0..J8 decision projection, never the static initial_status.
-- A later live decision is only meaningful after explicit separate GO.
CREATE OR REPLACE FUNCTION public.orotitan_j0_c8_read_gate_v1(p_milestone text)
RETURNS jsonb LANGUAGE plpgsql SECURITY INVOKER SET search_path=pg_catalog,public AS $$
DECLARE v_status jsonb;v_lock boolean;v_review_count bigint;
BEGIN
 v_status:=public.orotitan_j0_c8_gate_check_v1(p_milestone);
 SELECT enabled INTO v_lock FROM public.orotitan_j0_c8_acceptance_control_v1
 WHERE lock_name='GLOBAL_SHADOW';
 SELECT COUNT(*) INTO v_review_count FROM public.orotitan_j0_c8_reviews_v1
 WHERE milestone_code=p_milestone;
 RETURN v_status||jsonb_build_object('acceptanceEnabled',COALESCE(v_lock,false),
 'preparedReviews',v_review_count,
 'accepted',(v_status->>'alreadyAccepted')::boolean,
 'acceptanceAllowedNow',(COALESCE(v_lock,false)
  AND (v_status->>'eligibleForReview')::boolean));
END $$;
REVOKE ALL ON FUNCTION public.orotitan_j0_c8_read_gate_v1(text)
FROM PUBLIC,anon,authenticated;
GRANT EXECUTE ON FUNCTION public.orotitan_j0_c8_read_gate_v1(text)
TO service_role;

COMMIT;