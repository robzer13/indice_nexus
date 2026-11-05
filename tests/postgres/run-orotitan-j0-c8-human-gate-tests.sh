#!/usr/bin/env bash
# J0 C8 human approval in disposable LOCAL PostgreSQL only; never accept J0 on SHADOW.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
# Prevent libpq service/hostaddr overrides from redirecting this local test.
unset PGSERVICE PGSERVICEFILE PGHOSTADDR
case "$PGHOST" in 127.0.0.1|localhost|::1|/var/run/postgresql) ;;
 *) echo "J0 REFUSED NON-LOCAL POSTGRES" >&2; exit 2;; esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_j0_$$"
trap 'dropdb --if-exists --force "$db" >/dev/null 2>&1 || true' EXIT
createdb "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create schema if not exists extensions; alter extension pgcrypto set schema extensions; grant usage on schema extensions to service_role' >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20260909_orotitan_equity_i1_identity.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql" >/dev/null
for file in \
  20260914_orotitan_registry_v1_1_core.sql \
  20260914_orotitan_registry_v1_2_guards_rls.sql \
  20260914_orotitan_registry_v1_3_rpcs.sql \
  20260914_orotitan_registry_v1_4_manifest_authority.sql \
  20260914_orotitan_registry_v1_5_contract_pin_guards.sql \
  20260920_orotitan_registry_v1_6_checkpoint_output_revalidation.sql \
  20260920_orotitan_registry_v1_7_final_output_rebinding.sql \
  20260920_orotitan_registry_v1_8_manifest_persistence_receipt_integrity.sql \
  20260920203621_orotitan_registry_bundle_persistence_integrity.sql \
  20260920_orotitan_registry_v1_9_attestation_idempotency_fix.sql \
  20260921_orotitan_registry_v1_10_methodology_successor_cas.sql \
  20260921_orotitan_registry_v1_11_persistence_attestation_authority_repair.sql \
  20260923_orotitan_registry_v1_12_valuation_date_alignment_successor.sql; do
 psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/$file" >/dev/null
done
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_p0e_shadow_initial_run.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_p0e_shadow_helper_grants.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_j0_control_tower_shadow.sql" >/dev/null
# Preserve historic migrations; apply and replay the complete C8 chain in order.
# v1.1 intentionally requires all gates closed and no decisions. Replaying a
# predecessor alone is not a valid idempotence test of the installed frontier.
c8_chain=(
  20261031_orotitan_j0_c8_human_acceptance_shadow.sql
  20261031_orotitan_j0_c8_gate_event_serialization_shadow.sql
  20261031_orotitan_j0_c8_per_milestone_authority_shadow.sql
  20261031211508_orotitan_j0_c8_shadow_independent_reviewer_generic_authority_v1_1.sql
)
apply_c8_chain() {
 local pass="$1"
 if [[ "$pass" = replay ]]; then
  # Reuse v1.1's exact preflight BEFORE any predecessor can replace a function.
  # In conditional calls, bash disables errexit: explicitly propagate failures.
  sed -n '/^DO \$preflight\$/,/^END \$preflight\$;/p' "$root/migrations/${c8_chain[3]}" |
   psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null || return 1
 fi
 for file in "${c8_chain[@]}"; do
  echo "C8 $pass: $file"
  psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/$file" >/dev/null || return 1
 done
}
for pass in install replay; do
 apply_c8_chain "$pass"
 fingerprint="$(psql -XqAt -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/orotitan-j0-c8-protections-verify.sql")"
 if [[ "$pass" = install ]]; then installed_fingerprint="$fingerprint";
 else test "$fingerprint" = "$installed_fingerprint"; fi
done
echo "J0_C8_FULL_CHAIN_REPLAY_PROTECTIONS_PASS $installed_fingerprint"
psql -X -v ON_ERROR_STOP=1 -d "$db" \
 -f "$root/migrations/20261105_orotitan_j0_c8_solo_owner_attestation_shadow.sql" >/dev/null
installed_fingerprint="$(psql -XqAt -v ON_ERROR_STOP=1 -d "$db" \
 -f "$root/tests/postgres/orotitan-j0-c8-protections-verify.sql")"
echo "J0_C8_SOLO_CANDIDATE_STANDARD_COMPATIBILITY_INSTALLED $installed_fingerprint"
# Reuse the existing frozen thirteen-pin pack, including its real fingerprint.
# This deliberately unbound local run never starts an analytical stage.
pack="$(cat "$root/contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json")"
psql -X -v ON_ERROR_STOP=1 -v pack="$pack" -d "$db" <<'SQL'
 INSERT INTO public.orotitan_runs(
  creation_idempotency_key,entry_path,canonical_mode,run_type,
  data_cutoff,process_version,pilotage_contract_version,
  contract_pins,contract_set_sha256)
 SELECT 'c8-disposable-only-run','IMPOSED_COMPANY','ANALYZE','INITIAL',
  '2026-10-30',p->'contract_pins'->'process'->>'version',
  p->'contract_pins'->'pilotage'->>'version',
  p->'contract_pins',p->>'contract_set_sha256'
 FROM (SELECT :'pack'::jsonb AS p) pack;
 DO $pins$
 DECLARE r public.orotitan_runs%ROWTYPE; computed text; rejected_constraint text;
 BEGIN
  SELECT * INTO STRICT r FROM public.orotitan_runs
  WHERE creation_idempotency_key='c8-disposable-only-run';
  SELECT encode(extensions.digest(string_agg(
    key||'|'||(value->>'version')||'|'||(value->>'content_sha256'),
    E'\n' ORDER BY key COLLATE "C")||E'\n','sha256'),'hex') INTO computed
  FROM jsonb_each(r.contract_pins);
  IF (SELECT count(*) FROM jsonb_object_keys(r.contract_pins))<>13
     OR NOT public.orotitan_contract_pins_complete(r.contract_pins)
     OR computed<>r.contract_set_sha256
  THEN RAISE EXCEPTION 'J0_C8_CONTRACT_PIN_FIXTURE_INVALID';END IF;
  BEGIN
   INSERT INTO public.orotitan_runs(
    creation_idempotency_key,entry_path,canonical_mode,run_type,data_cutoff,
    process_version,pilotage_contract_version,contract_pins,contract_set_sha256)
   VALUES('c8-invalid-pins-local-only','IMPOSED_COMPANY','ANALYZE','INITIAL',
    '2026-10-30','1.0','1.0.1','{}'::jsonb,repeat('a',64));
   RAISE EXCEPTION 'J0_C8_EMPTY_PINS_ACCEPTED';
  EXCEPTION WHEN check_violation THEN
   GET STACKED DIAGNOSTICS rejected_constraint=CONSTRAINT_NAME;
   IF rejected_constraint<>'orotitan_runs_contract_pins_complete_check' THEN RAISE;END IF;
  END;
  RAISE NOTICE 'J0_C8_THIRTEEN_REAL_PINS_AND_EMPTY_PINS_REJECTED_PASS';
 END $pins$;
SQL
psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL'
set role service_role;
do $matrix$
declare
 i integer; ids uuid[]:='{}'::uuid[];
 x jsonb; prepared jsonb; replay jsonb; rc jsonb;
 v_hash text:=repeat('b',64);
 v_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/J0_SANDBOX_TEST.md';
 v_review_key text:='orotitan:c8:review:local-j0-matrix-20261031';
 v_id uuid;
 v_local_run uuid;
 other_ids uuid[]:='{}'::uuid[]; other_review jsonb;
begin
 if (SELECT count(*) FROM pg_trigger
     WHERE tgrelid='public.orotitan_j0_events_v1'::regclass
       AND tgname='orotitan_j0_c8_serialize_event_insert'
       AND tgenabled='O')<>1
 then raise exception 'J0_C8_SERIALIZATION_TRIGGER_REQUIRED';end if;
 if (select count(*) from public.orotitan_j0_c8_acceptance_control_v1 where enabled)<>0
  or (select count(*) from public.orotitan_j0_c8_reviews_v1)<>0
  or (select count(*) from public.orotitan_j0_c8_decisions_v1)<>0
 then raise exception 'J0_C8_INITIAL_UNEXPECTED_WRITES';end if;
 if has_table_privilege('service_role','public.orotitan_j0_c8_decisions_v1','INSERT')
  or has_table_privilege('service_role','public.orotitan_j0_c8_reviews_v1','INSERT')
  or has_table_privilege('service_role','public.orotitan_j0_c8_acceptance_control_v1','UPDATE')
  or has_function_privilege('anon','public.orotitan_j0_c8_accept_review_v1(text,uuid,text,text,text,text)','EXECUTE')
  or has_function_privilege('authenticated','public.orotitan_j0_c8_prepare_review_v1(text,text,text,text,text,uuid[])','EXECUTE')
 then raise exception 'J0_C8_ROLE_GUARDS_FAILED';end if;
 -- Actual SHADOW J0 blocker evidence is simulated ONLY with no proof here;
 -- check J0 cannot prepare without 7 real criteria proofs.
 x:=public.orotitan_j0_c8_read_gate_v1('J0');
 IF (SELECT COUNT(*) FROM public.orotitan_j0_c8_milestone_activation_v1 WHERE enabled)<>0
 THEN RAISE EXCEPTION 'J0_C8_MILESTONE_LOCKS_NOT_DISABLED';END IF;
 if x->>'eligibleForReview'<>'false' OR x->>'acceptanceEnabled'<>'false' then
   raise exception 'J0_C8_PREMATURE_J0_READY';end if;
 begin
   perform public.orotitan_j0_c8_prepare_review_v1(
     'orotitan:c8:review:rejected-j0-20261031',
     'J0',v_ref,v_hash,'Reviewer Local',array[
      gen_random_uuid(),gen_random_uuid(),gen_random_uuid(),
      gen_random_uuid(),gen_random_uuid(),gen_random_uuid(),gen_random_uuid()
     ]);
   raise exception 'J0_C8_EXPECTED_J0_MISSING_PROOFS';
 exception when others then
   if sqlerrm<>'J0_C8_GATE_NOT_READY' then raise;end if;
 end;
 SELECT run_id INTO STRICT v_local_run FROM public.orotitan_runs
 WHERE creation_idempotency_key='c8-disposable-only-run';
 -- Make a fictional J0 with seven local evidence events; no other issuer or
 -- production run is ever started.
 for i in 1..7 loop
   x:=public.orotitan_j0_record_event_v1(
    'orotitan:j0:c8-local-j0-proof-'||i::text,
    'J0','EVIDENCE_RECORDED','Disposable isolated proof C'||i::text,
    CASE WHEN i IN (1,6,7) THEN 'registry:run:'||v_local_run::text
     ELSE 'test:j0-sandbox-c8-proof-'||i::text END,
    CASE WHEN i IN (1,6,7) THEN v_local_run ELSE NULL END,
    jsonb_build_object('criterionIndex',i,'source','LOCAL_ONLY'));
   if x->>'status'<>'CREATED' then raise exception 'J0_C8_FIXTURE_WRITE_FAILED';end if;
   ids:=array_append(ids,(x->>'event_id')::uuid);
 end loop;
 x:=public.orotitan_j0_c8_gate_check_v1('J0');
 if x->>'eligibleForReview'<>'true' OR x->>'openBlockers'<>'0' then
  raise exception 'J0_C8_SEVEN_PROOFS_NOT_READY %',x;end if;
 prepared:=public.orotitan_j0_c8_prepare_review_v1(
  v_review_key,'J0',v_ref,v_hash,'Reviewer Local',ids);
 if prepared->>'status'<>'PREPARED' OR prepared->>'accepted'<>'false'
 or (select count(*) from public.orotitan_j0_c8_reviews_v1)<>1
 or (select count(*) from public.orotitan_j0_c8_decisions_v1)<>0
 then raise exception 'J0_C8_PREPARE_RECEIPT_FAILED %',prepared;end if;
 replay:=public.orotitan_j0_c8_prepare_review_v1(
  v_review_key,'J0',v_ref,v_hash,'Reviewer Local',ids);
 if replay->>'status'<>'REPLAYED' or replay->>'reviewId'<>prepared->>'reviewId'
 then raise exception 'J0_C8_REVIEW_REPLAY_FAILED';end if;
 begin
  perform public.orotitan_j0_c8_prepare_review_v1(
   v_review_key,'J0',v_ref,repeat('c',64),'Reviewer Local',ids);
  raise exception 'J0_C8_REVIEW_CONFLICT_NOT_DETECTED';
 exception when others then
  if sqlerrm<>'J0_C8_REVIEW_IDEMPOTENCY_CONFLICT' then raise;end if;
 end;
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   'orotitan:c8:decision:local-j0-20261031',(prepared->>'reviewId')::uuid,
   prepared->>'reviewSha256','Approver Local',
   'ACCEPTER J0 SHADOW','GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_ACCEPTANCE_DISABLED_BYPASS';
 exception when others then
  if sqlerrm<>'J0_C8_ACCEPTANCE_LOCKED' then raise;end if;
 end;
 -- Prepare a local J1 review without starting J1. It must remain unaccepted
 -- even when J0 is explicitly enabled later by the disposable database owner.
 for i in 1..7 loop
  x:=public.orotitan_j0_record_event_v1(
   'orotitan:j0:c8-local-j1-proof-'||i::text,'J1','EVIDENCE_RECORDED',
   'Disposable other-milestone proof C'||i::text,'test:local-j1-proof-'||i::text,
   NULL,jsonb_build_object('criterionIndex',i,'source','LOCAL_ONLY'));
  other_ids:=array_append(other_ids,(x->>'event_id')::uuid);
 end loop;
 other_review:=public.orotitan_j0_c8_prepare_review_v1(
  'orotitan:c8:review:local-j1-locked-only','J1',v_ref,v_hash,'Reviewer Local',other_ids);
 perform set_config('orotitan.c8_other_review',other_review->>'reviewId',false);
 perform set_config('orotitan.c8_other_hash',other_review->>'reviewSha256',false);
 perform set_config('orotitan.c8_fixture_review',prepared->>'reviewId',false);
 perform set_config('orotitan.c8_fixture_hash',prepared->>'reviewSha256',false);
 perform set_config('orotitan.c8_fixture_proofs',array_to_string(ids,','),false);
 raise notice 'J0_C8_TESTS_PREPARE_REPLAY_CONFLICT_AND_LOCK_PASS';
end $matrix$;
reset role;
-- Only the disposable PG owner can activate the lock for testing.
update public.orotitan_j0_c8_acceptance_control_v1
set enabled=true,activation_authority_ref='GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST'
where lock_name='GLOBAL_SHADOW';
UPDATE public.orotitan_j0_c8_milestone_activation_v1
SET enabled=true,authority_ref='GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST'
WHERE milestone_code='J0';
DO $guard$
BEGIN
 IF (SELECT COUNT(*) FROM public.orotitan_j0_c8_milestone_activation_v1 WHERE enabled)<>1
 OR (SELECT COUNT(*) FROM public.orotitan_j0_c8_milestone_activation_v1 WHERE enabled AND milestone_code='J1')<>0
 THEN RAISE EXCEPTION 'J0_C8_CROSS_MILESTONE_GO_LEAK';END IF;
END $guard$;
set role service_role;
do $accept$
declare
 old_id uuid:=current_setting('orotitan.c8_fixture_review')::uuid;
 old_hash text:=current_setting('orotitan.c8_fixture_hash');
 ids uuid[]:=string_to_array(current_setting('orotitan.c8_fixture_proofs'),',')::uuid[];
 newer jsonb; v jsonb; replay jsonb; x jsonb;
 v_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/J0_SANDBOX_TEST.md';
 v_decision_key text:='orotitan:c8:decision:local-j0-test-20261031';
begin
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,old_id,old_hash,'Approver Local','WRONG',
   'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_CONFIRMATION_BYPASS';
 exception when others then
  if sqlerrm<>'J0_C8_EXPLICIT_CONFIRMATION_REQUIRED' then raise;end if;
 end;
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,old_id,old_hash,'Approver Local','ACCEPTER J0 SHADOW',
   'GO_ACCEPT_J0_WRONG_AUTHORITY');
  raise exception 'J0_C8_AUTHORITY_BYPASS';
 exception when others then
  if sqlerrm<>'J0_C8_AUTHORITY_REF_INVALID' then raise;end if;
 end;
 -- v1.1 compares declared labels after lower(btrim(...)).
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,old_id,old_hash,'  rEvIeWeR lOcAl  ','ACCEPTER J0 SHADOW',
   'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_IDENTICAL_NORMALIZED_LABELS_ACCEPTED';
 exception when others then
  if sqlerrm<>'J0_C8_REVIEWER_APPROVER_NOT_DISTINCT' then raise;end if;
 end;
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   'orotitan:c8:decision:local-j1-denied-only',
   current_setting('orotitan.c8_other_review')::uuid,
   current_setting('orotitan.c8_other_hash'),'Approver Local','ACCEPTER J1 SHADOW',
   'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_OTHER_MILESTONE_ACCEPTED_WITH_J0_GO';
 exception when others then
  if sqlerrm<>'J0_C8_MILESTONE_ACCEPTANCE_LOCKED' then raise;end if;
 end;
 raise notice 'J0_C8_NORMALIZED_LABEL_AND_OTHER_MILESTONE_DENIAL_PASS';
 -- New event makes the earlier review stale.
 v:=public.orotitan_j0_record_event_v1(
  'orotitan:j0:c8-new-j0-proof-for-staleness',
  'J0','EVIDENCE_RECORDED',
  'New evidence invalidates previous review',
  'test:j0-sandbox-c8-staleness',NULL,
  '{"criterionIndex":4}'::jsonb);
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,old_id,old_hash,'Approver Local','ACCEPTER J0 SHADOW',
   'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_STALE_REVIEW_ACCEPTED';
 exception when others then
  if sqlerrm<>'J0_C8_REVIEW_STALE' then raise;end if;
 end;
 -- New sealed review under the new watermark.
 newer:=public.orotitan_j0_c8_prepare_review_v1(
  'orotitan:c8:review:local-j0-after-staleness-20261031',
  'J0',v_ref,repeat('b',64),'Reviewer Local',ids);
 v:=public.orotitan_j0_c8_accept_review_v1(
  v_decision_key,(newer->>'reviewId')::uuid,newer->>'reviewSha256',
  'Approver Local','ACCEPTER J0 SHADOW',
  'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
 if v->>'status'<>'ACCEPTED' or v->>'accepted'<>'true'
  or v->>'stageStarted'<>'false' or v->>'published'<>'false'
  or (select count(*) from public.orotitan_j0_c8_decisions_v1)<>1
  or (select count(*) from public.orotitan_run_stages)<>0
 then raise exception 'J0_C8_SIMULATED_ACCEPT_FAILED %',v;end if;
 replay:=public.orotitan_j0_c8_accept_review_v1(
  v_decision_key,(newer->>'reviewId')::uuid,newer->>'reviewSha256',
  'Approver Local','ACCEPTER J0 SHADOW',
  'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
 if replay->>'status'<>'REPLAYED' or replay->>'decisionId'<>v->>'decisionId'
 then raise exception 'J0_C8_DECISION_REPLAY_FAILED';end if;
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,(newer->>'reviewId')::uuid,newer->>'reviewSha256',
   'Approver Forged','ACCEPTER J0 SHADOW',
   'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_DECISION_CONFLICT_NOT_DENIED';
 exception when others then
  if sqlerrm<>'J0_C8_DECISION_IDEMPOTENCY_CONFLICT' then raise;end if;
 end;
 begin
  update public.orotitan_j0_c8_decisions_v1 set decision_status='DENIED'
   where decision_id=(v->>'decisionId')::uuid;
  raise exception 'J0_C8_DECISION_UPDATE_ALLOWED';
 exception when insufficient_privilege then null;
 end;
 x:=public.orotitan_j0_c8_read_gate_v1('J0');
 if x->>'accepted'<>'true' or x->>'acceptanceAllowedNow'<>'false' then
  raise exception 'J0_C8_ACCEPTED_PROJECTION_INVALID %',x;end if;
 if (select count(*) from public.orotitan_j0_c8_decisions_v1
     where milestone_code='J1')<>0 then raise exception 'J1_UNAUTHORIZED_IN_LOCAL_TEST';end if;
 raise notice 'J0_C8_TESTS_STALE_WRONG_GO_ACCEPT_REPLAY_IMMUTABLE_PASS';
end $accept$;
reset role;
SQL
# Full-chain replay must refuse BEFORE replacing any predecessor definition.
if reapply_output="$(apply_c8_chain replay 2>&1)"; then
 echo "C8 chain unexpectedly replayed with an active local decision" >&2; exit 1
fi
[[ "$reapply_output" == *C8_SHADOW_HARDENING_REQUIRES_ALL_GATES_DISABLED* ]]
[[ "$reapply_output" != *"C8 replay:"* ]]
echo "J0_C8_ACTIVE_CHAIN_REPLAY_REFUSED_BEFORE_PREDECESSORS_PASS"
# An active/accepted database must reject v1.1 at preflight, before mutation.
if reapply_output="$(psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/${c8_chain[3]}" 2>&1)"; then
 echo "C8 v1.1 unexpectedly reapplied with an active local decision" >&2; exit 1
fi
[[ "$reapply_output" == *C8_SHADOW_HARDENING_REQUIRES_ALL_GATES_DISABLED* ]]
echo "J0_C8_ACTIVE_V1_1_REAPPLY_REFUSED_WITHOUT_MUTATION_PASS"
# Verify actual final constraints, function bodies, ACLs/RLS and triggers, not
# merely the decision count. No obsolete predecessor is reapplied after v1.1.
test "$(psql -XqAt -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/orotitan-j0-c8-protections-verify.sql")" = "$installed_fingerprint"
psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL'
DO $final$
BEGIN
 IF (SELECT count(*) FROM public.orotitan_j0_c8_decisions_v1)<>1
 OR (SELECT count(*) FROM public.orotitan_j0_c8_decisions_v1 WHERE milestone_code<>'J0')<>0
 OR (SELECT count(*) FROM public.orotitan_run_stages)<>0
 OR (SELECT count(*) FROM public.orotitan_artifacts)<>0
 OR (SELECT count(*) FROM public.orotitan_run_events)<>0
 OR (SELECT count(*) FROM public.snapshots)<>(SELECT snapshots FROM public.fixture_legacy_before)
 OR EXISTS (SELECT 1 FROM public.orotitan_runs
            WHERE run_status<>'CREATED' OR current_stage IS NOT NULL)
 OR (SELECT count(*) FROM public.orotitan_j0_c8_milestone_activation_v1 WHERE enabled)<>1
 THEN RAISE EXCEPTION 'J0_C8_FINAL_STATE_REGRESSION';END IF;
 RAISE NOTICE 'J0_C8_FINAL_V1_1_NO_STAGE_NO_PUBLICATION_PASS';
END $final$;
SQL
echo "J0 C8 LOCAL PG PASS: full chain install/replay, 13 pins, closed lock, review replay/conflict, normalized labels, wrong GO, stale review, other-milestone denial, LOCAL J0-only approval/replay, immutable decision, unchanged v1.1 protections, no stage or publication"
