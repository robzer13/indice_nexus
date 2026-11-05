#!/usr/bin/env bash
# SOLO_OWNER acceptance simulation in disposable LOCAL PostgreSQL only.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
unset PGSERVICE PGSERVICEFILE PGHOSTADDR
case "$PGHOST" in 127.0.0.1|localhost|::1|/var/run/postgresql) ;; *) echo "C8 SOLO REFUSED NON-LOCAL POSTGRES" >&2; exit 2;; esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_c8_solo_$$"
trap 'dropdb --if-exists --force "$db" >/dev/null 2>&1 || true' EXIT
createdb "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create schema if not exists extensions; alter extension pgcrypto set schema extensions; grant usage on schema extensions to service_role' >/dev/null
for file in 20260909_orotitan_equity_i1_identity.sql 20260911_orotitan_equity_i3_snapshot_persistence.sql \
  20260914_orotitan_registry_v1_1_core.sql 20260914_orotitan_registry_v1_2_guards_rls.sql \
  20260914_orotitan_registry_v1_3_rpcs.sql 20260914_orotitan_registry_v1_4_manifest_authority.sql \
  20260914_orotitan_registry_v1_5_contract_pin_guards.sql 20260920_orotitan_registry_v1_6_checkpoint_output_revalidation.sql \
  20260920_orotitan_registry_v1_7_final_output_rebinding.sql 20260920_orotitan_registry_v1_8_manifest_persistence_receipt_integrity.sql \
  20260920203621_orotitan_registry_bundle_persistence_integrity.sql 20260920_orotitan_registry_v1_9_attestation_idempotency_fix.sql \
  20260921_orotitan_registry_v1_10_methodology_successor_cas.sql 20260921_orotitan_registry_v1_11_persistence_attestation_authority_repair.sql \
  20260923_orotitan_registry_v1_12_valuation_date_alignment_successor.sql \
  20261031_orotitan_runfactory_p0e_shadow_initial_run.sql 20261031_orotitan_runfactory_p0e_shadow_helper_grants.sql \
  20261031_orotitan_runfactory_j0_control_tower_shadow.sql 20261031_orotitan_j0_c8_human_acceptance_shadow.sql \
  20261031_orotitan_j0_c8_gate_event_serialization_shadow.sql 20261031_orotitan_j0_c8_per_milestone_authority_shadow.sql \
  20261031211508_orotitan_j0_c8_shadow_independent_reviewer_generic_authority_v1_1.sql \
  20261105_orotitan_j0_c8_solo_owner_attestation_shadow.sql; do
 psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/$file" >/dev/null
done
pack="$(cat "$root/contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json")"
psql -X -v ON_ERROR_STOP=1 -v pack="$pack" -d "$db" <<'SQL'
INSERT INTO public.orotitan_runs(creation_idempotency_key,entry_path,canonical_mode,run_type,
 data_cutoff,process_version,pilotage_contract_version,contract_pins,contract_set_sha256)
SELECT 'c8-solo-disposable-run','IMPOSED_COMPANY','ANALYZE','INITIAL','2026-11-05',
 p->'contract_pins'->'process'->>'version',p->'contract_pins'->'pilotage'->>'version',
 p->'contract_pins',p->>'contract_set_sha256' FROM (SELECT :'pack'::jsonb p) s;
DO $fixture$
DECLARE i integer;x jsonb;ids uuid[]:='{}';run_id uuid;owner_id uuid:='11111111-1111-4111-8111-111111111111';
 amendment_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/SOLO_TEST.md';
 audit_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('b',40)||'/docs/audits/J0_TEST.md';
BEGIN
 SELECT r.run_id INTO STRICT run_id FROM public.orotitan_runs r WHERE creation_idempotency_key='c8-solo-disposable-run';
 FOR i IN 1..7 LOOP
  x:=public.orotitan_j0_record_event_v1('orotitan:j0:c8-solo-proof-'||i,'J0','EVIDENCE_RECORDED',
   'Fictional local SOLO proof C'||i,CASE WHEN i IN(1,6,7) THEN 'registry:run:'||run_id ELSE 'test:solo-proof-'||i END,
   CASE WHEN i IN(1,6,7) THEN run_id ELSE NULL END,jsonb_build_object('criterionIndex',i,'source','LOCAL_ONLY'));
  ids:=array_append(ids,(x->>'event_id')::uuid);
 END LOOP;
 PERFORM set_config('orotitan.solo.ids',array_to_string(ids,','),false);
 x:=public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:standard-under-solo-go','J0',audit_ref,repeat('c',64),
  'Independent Standard Reviewer',ids);
 PERFORM set_config('orotitan.solo.standard_review_id',x->>'reviewId',false);
 PERFORM set_config('orotitan.solo.standard_review_hash',x->>'reviewSha256',false);
 BEGIN
  PERFORM public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:solo-before-ratification','J0',audit_ref,repeat('c',64),
   'supabase-auth:'||owner_id,ids,'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
  RAISE EXCEPTION 'SOLO_DRAFT_AMENDMENT_BYPASS';
 EXCEPTION WHEN OTHERS THEN IF SQLERRM<>'J0_C8_SOLO_AMENDMENT_NOT_RATIFIED' THEN RAISE;END IF;END;
 BEGIN
  PERFORM public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:solo-j1-denied','J1',audit_ref,repeat('c',64),
   'supabase-auth:'||owner_id,ids,'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
  RAISE EXCEPTION 'SOLO_J1_BYPASS';
 EXCEPTION WHEN OTHERS THEN IF SQLERRM<>'J0_C8_SOLO_REVIEW_INPUT_INVALID' THEN RAISE;END IF;END;
 RAISE NOTICE 'C8_SOLO_DRAFT_AND_J1_DENIAL_PASS';
END $fixture$;

UPDATE public.orotitan_j0_c8_acceptance_control_v1 SET solo_owner_enabled=true,
 solo_amendment_ref='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/SOLO_TEST.md',
 solo_amendment_sha256=repeat('d',64) WHERE lock_name='GLOBAL_SHADOW';
DO $null_review_constraints$
DECLARE i integer;owner_id uuid:='11111111-1111-4111-8111-111111111111';
 v_amendment_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/SOLO_TEST.md';
BEGIN
 FOR i IN 1..4 LOOP
  BEGIN
   INSERT INTO public.orotitan_j0_c8_reviews_v1(
    review_key,milestone_code,audit_ref,audit_sha256,reviewer_label,evidence_event_ids,
    criteria_sha256,journal_watermark,review_sha256,governance_mode,auth_provider,
    owner_user_id,amendment_ref,amendment_sha256)
   SELECT 'orotitan:c8:review:direct-null-'||i,s.milestone_code,s.audit_ref,s.audit_sha256,
    'supabase-auth:'||owner_id,s.evidence_event_ids,s.criteria_sha256,s.journal_watermark,
    s.review_sha256,'SOLO_OWNER',CASE WHEN i=3 THEN NULL ELSE 'SUPABASE_AUTH' END,
    CASE WHEN i=4 THEN NULL ELSE owner_id END,CASE WHEN i=1 THEN NULL ELSE v_amendment_ref END,
    CASE WHEN i=2 THEN NULL ELSE repeat('d',64) END
   FROM public.orotitan_j0_c8_reviews_v1 s WHERE s.review_key='orotitan:c8:review:standard-under-solo-go';
   RAISE EXCEPTION 'SOLO_DIRECT_NULL_REVIEW_CONSTRAINT_BYPASS_%',i;
  EXCEPTION WHEN check_violation THEN NULL;END;
 END LOOP;
END $null_review_constraints$;
SET ROLE service_role;
DO $prepare$
DECLARE ids uuid[]:=string_to_array(current_setting('orotitan.solo.ids'),',')::uuid[];
 owner_id uuid:='11111111-1111-4111-8111-111111111111';p jsonb;r jsonb;
 amendment_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/SOLO_TEST.md';
 audit_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('b',40)||'/docs/audits/J0_TEST.md';
BEGIN
 FOR i IN 1..4 LOOP
  BEGIN
   PERFORM public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:solo-null-'||i,'J0',audit_ref,repeat('c',64),
    'supabase-auth:'||owner_id,ids,'SOLO_OWNER',
    CASE WHEN i=3 THEN NULL ELSE 'SUPABASE_AUTH' END,
    CASE WHEN i=4 THEN NULL ELSE owner_id END,
    CASE WHEN i=1 THEN NULL ELSE amendment_ref END,
    CASE WHEN i=2 THEN NULL ELSE repeat('d',64) END);
   RAISE EXCEPTION 'SOLO_NULL_REVIEW_INPUT_BYPASS_%',i;
  EXCEPTION WHEN OTHERS THEN
   IF SQLERRM<>'J0_C8_SOLO_REVIEW_INPUT_INVALID' THEN RAISE;END IF;
  END;
 END LOOP;
 IF (SELECT count(*) FROM public.orotitan_j0_c8_reviews_v1
     WHERE governance_mode='SOLO_OWNER')<>0
 THEN RAISE EXCEPTION 'SOLO_NULL_REVIEW_PERSISTED';END IF;
 p:=public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:solo-nominal-20261105','J0',audit_ref,repeat('c',64),
  'supabase-auth:'||owner_id,ids,'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
 r:=public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:solo-nominal-20261105','J0',audit_ref,repeat('c',64),
  'supabase-auth:'||owner_id,ids,'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
 IF p->>'status'<>'PREPARED' OR r->>'status'<>'REPLAYED' OR r->>'reviewId'<>p->>'reviewId' THEN RAISE EXCEPTION 'SOLO_PREPARE_REPLAY_FAILED';END IF;
 BEGIN
  PERFORM public.orotitan_j0_c8_prepare_review_v1('orotitan:c8:review:solo-nominal-20261105','J0',audit_ref,repeat('c',64),
   'supabase-auth:'||owner_id,ids,'SOLO_OWNER','SUPABASE_AUTH','22222222-2222-4222-8222-222222222222',amendment_ref,repeat('d',64));
  RAISE EXCEPTION 'SOLO_REVIEW_CONFLICT_BYPASS';
 EXCEPTION WHEN OTHERS THEN IF SQLERRM<>'J0_C8_SOLO_REVIEW_INPUT_INVALID' AND SQLERRM<>'J0_C8_REVIEW_IDEMPOTENCY_CONFLICT' THEN RAISE;END IF;END;
 PERFORM set_config('orotitan.solo.review_id',p->>'reviewId',false);
 PERFORM set_config('orotitan.solo.review_hash',p->>'reviewSha256',false);
END $prepare$;
RESET ROLE;

DO $null_decision_constraints$
DECLARE i integer;owner_id uuid:='11111111-1111-4111-8111-111111111111';r public.orotitan_j0_c8_reviews_v1%ROWTYPE;
 amendment_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/SOLO_TEST.md';
BEGIN
 SELECT * INTO STRICT r FROM public.orotitan_j0_c8_reviews_v1 WHERE review_key='orotitan:c8:review:solo-nominal-20261105';
 FOR i IN 1..5 LOOP
  BEGIN
   INSERT INTO public.orotitan_j0_c8_decisions_v1(
    decision_key,milestone_code,review_id,review_sha256,approver_label,consent_phrase,
    authority_ref,governance_mode,auth_provider,owner_user_id,amendment_ref,
    amendment_sha256,decision_sha256)
   VALUES('orotitan:c8:decision:direct-null-'||i,'J0',r.review_id,r.review_sha256,
    'supabase-auth:'||owner_id,'ACCEPTER J0 SHADOW EN MODE SOLO','GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
    'SOLO_OWNER',CASE WHEN i=3 THEN NULL ELSE 'SUPABASE_AUTH' END,
    CASE WHEN i=4 THEN NULL ELSE owner_id END,CASE WHEN i=1 THEN NULL ELSE amendment_ref END,
    CASE WHEN i=2 THEN NULL ELSE repeat('d',64) END,
    CASE WHEN i=5 THEN NULL ELSE repeat('e',64) END);
   RAISE EXCEPTION 'SOLO_DIRECT_NULL_DECISION_CONSTRAINT_BYPASS_%',i;
  EXCEPTION WHEN check_violation THEN NULL;END;
 END LOOP;
END $null_decision_constraints$;

UPDATE public.orotitan_j0_c8_acceptance_control_v1 SET enabled=true,
 activation_authority_ref='GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105' WHERE lock_name='GLOBAL_SHADOW';
UPDATE public.orotitan_j0_c8_milestone_activation_v1 SET enabled=true,
 authority_ref='GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105' WHERE milestone_code='J0';
SET ROLE service_role;
DO $accept$
DECLARE review_id uuid:=current_setting('orotitan.solo.review_id')::uuid;
 review_hash text:=current_setting('orotitan.solo.review_hash');owner_id uuid:='11111111-1111-4111-8111-111111111111';
 other_id uuid:='22222222-2222-4222-8222-222222222222';d jsonb;r jsonb;
 standard_review_id uuid:=current_setting('orotitan.solo.standard_review_id')::uuid;
 standard_review_hash text:=current_setting('orotitan.solo.standard_review_hash');i integer;
 amendment_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/SOLO_TEST.md';
BEGIN
 BEGIN
  PERFORM public.orotitan_j0_c8_accept_review_v1('orotitan:c8:decision:standard-under-solo-go',
   standard_review_id,standard_review_hash,'Independent Standard Approver','ACCEPTER J0 SHADOW',
   'GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105');
  RAISE EXCEPTION 'STANDARD_UNDER_SOLO_GO_BYPASS';
 EXCEPTION WHEN OTHERS THEN
  IF SQLERRM<>'J0_C8_GOVERNANCE_MODE_MISMATCH' THEN RAISE;END IF;
 END;
 IF (SELECT count(*) FROM public.orotitan_j0_c8_decisions_v1)<>0
 THEN RAISE EXCEPTION 'STANDARD_UNDER_SOLO_GO_DECISION_PERSISTED';END IF;
 RAISE NOTICE 'C8_STANDARD_UNDER_SOLO_GO_DENIAL_PASS';
 FOR i IN 1..4 LOOP
  BEGIN
   PERFORM public.orotitan_j0_c8_accept_review_v1('orotitan:c8:decision:solo-null-'||i,review_id,review_hash,
    'supabase-auth:'||owner_id,'ACCEPTER J0 SHADOW EN MODE SOLO','GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
    'SOLO_OWNER',CASE WHEN i=3 THEN NULL ELSE 'SUPABASE_AUTH' END,
    CASE WHEN i=4 THEN NULL ELSE owner_id END,
    CASE WHEN i=1 THEN NULL ELSE amendment_ref END,
    CASE WHEN i=2 THEN NULL ELSE repeat('d',64) END);
   RAISE EXCEPTION 'SOLO_NULL_DECISION_INPUT_BYPASS_%',i;
  EXCEPTION WHEN OTHERS THEN
   IF SQLERRM<>'J0_C8_SOLO_DECISION_INPUT_INVALID' THEN RAISE;END IF;
  END;
 END LOOP;
 IF (SELECT count(*) FROM public.orotitan_j0_c8_decisions_v1)<>0
 THEN RAISE EXCEPTION 'SOLO_NULL_DECISION_PERSISTED';END IF;
 BEGIN
  PERFORM public.orotitan_j0_c8_accept_review_v1('orotitan:c8:decision:solo-wrong-owner',review_id,review_hash,
   'supabase-auth:'||other_id,'ACCEPTER J0 SHADOW EN MODE SOLO','GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
   'SOLO_OWNER','SUPABASE_AUTH',other_id,amendment_ref,repeat('d',64));
  RAISE EXCEPTION 'SOLO_WRONG_OWNER_BYPASS';
 EXCEPTION WHEN OTHERS THEN IF SQLERRM<>'J0_C8_SOLO_OWNER_ATTESTATION_MISMATCH' THEN RAISE;END IF;END;
 BEGIN
  PERFORM public.orotitan_j0_c8_accept_review_v1('orotitan:c8:decision:solo-wrong-consent',review_id,review_hash,
   'supabase-auth:'||owner_id,'ACCEPTER J0 SHADOW','GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
   'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
  RAISE EXCEPTION 'SOLO_WRONG_CONSENT_BYPASS';
 EXCEPTION WHEN OTHERS THEN IF SQLERRM<>'J0_C8_SOLO_DECISION_INPUT_INVALID' THEN RAISE;END IF;END;
 d:=public.orotitan_j0_c8_accept_review_v1('orotitan:c8:decision:solo-nominal-20261105',review_id,review_hash,
  'supabase-auth:'||owner_id,'ACCEPTER J0 SHADOW EN MODE SOLO','GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
  'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
 r:=public.orotitan_j0_c8_accept_review_v1('orotitan:c8:decision:solo-nominal-20261105',review_id,review_hash,
  'supabase-auth:'||owner_id,'ACCEPTER J0 SHADOW EN MODE SOLO','GO_ACCEPT_J0_SOLO_LOCAL_TEST_20261105',
  'SOLO_OWNER','SUPABASE_AUTH',owner_id,amendment_ref,repeat('d',64));
 IF d->>'status'<>'ACCEPTED' OR r->>'status'<>'REPLAYED' OR
  r->>'decisionSha256'<>d->>'decisionSha256' OR d->>'decisionSha256' !~ '^[a-f0-9]{64}$' OR
  d->>'stageStarted'<>'false' OR d->>'published'<>'false'
  OR (SELECT count(*) FROM public.orotitan_j0_c8_decisions_v1 WHERE governance_mode='SOLO_OWNER' AND milestone_code='J0')<>1
  OR (SELECT count(*) FROM public.orotitan_j0_c8_decisions_v1 WHERE milestone_code='J1')<>0
  OR (SELECT count(*) FROM public.orotitan_run_stages)<>0 OR (SELECT count(*) FROM public.orotitan_artifacts)<>0
  OR (SELECT count(*) FROM public.orotitan_run_events)<>0
 THEN RAISE EXCEPTION 'SOLO_NOMINAL_OR_NO_CASCADE_FAILED %',d;END IF;
 RAISE NOTICE 'C8_SOLO_NOMINAL_REPLAY_NO_J1_NO_RESEARCH_NO_PUBLICATION_PASS';
END $accept$;
RESET ROLE;
SQL
echo "J0 C8 SOLO_OWNER LOCAL PG PASS: STANDARD-under-SOLO and NULL identity/amendment denied; attested fictional owner nominal J0 replayed; no Research/J1/publication"
