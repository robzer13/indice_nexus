#!/usr/bin/env bash
# J0 C8 human approval in disposable LOCAL PostgreSQL only; never accept J0 on SHADOW.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
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
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_j0_c8_human_acceptance_shadow.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL'
set role service_role;
do $matrix$
declare
 i integer; ids uuid[]:='{}'::uuid[];
 x jsonb; prepared jsonb; replay jsonb; rc jsonb;
 v_hash text:=repeat('b',64);
 v_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/J1_SANDBOX_TEST.md';
 v_review_key text:='orotitan:c8:review:local-j1-matrix-20261031';
 v_id uuid;
begin
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
 -- Make a fictional J1 with seven local evidence events; no other issuer or
 -- production run is ever started.
 for i in 1..7 loop
   x:=public.orotitan_j0_record_event_v1(
    'orotitan:j0:c8-local-j1-proof-'||i::text,
    'J1','EVIDENCE_RECORDED','Disposable isolated proof C'||i::text,
    'test:j1-sandbox-c8-proof-'||i::text,NULL,
    jsonb_build_object('criterionIndex',i,'source','LOCAL_ONLY'));
   if x->>'status'<>'CREATED' then raise exception 'J0_C8_FIXTURE_WRITE_FAILED';end if;
   ids:=array_append(ids,(x->>'event_id')::uuid);
 end loop;
 x:=public.orotitan_j0_c8_gate_check_v1('J1');
 if x->>'eligibleForReview'<>'true' OR x->>'openBlockers'<>'0' then
  raise exception 'J0_C8_SEVEN_PROOFS_NOT_READY %',x;end if;
 prepared:=public.orotitan_j0_c8_prepare_review_v1(
  v_review_key,'J1',v_ref,v_hash,'Reviewer Local',ids);
 if prepared->>'status'<>'PREPARED' OR prepared->>'accepted'<>'false'
 or (select count(*) from public.orotitan_j0_c8_reviews_v1)<>1
 or (select count(*) from public.orotitan_j0_c8_decisions_v1)<>0
 then raise exception 'J0_C8_PREPARE_RECEIPT_FAILED %',prepared;end if;
 replay:=public.orotitan_j0_c8_prepare_review_v1(
  v_review_key,'J1',v_ref,v_hash,'Reviewer Local',ids);
 if replay->>'status'<>'REPLAYED' or replay->>'reviewId'<>prepared->>'reviewId'
 then raise exception 'J0_C8_REVIEW_REPLAY_FAILED';end if;
 begin
  perform public.orotitan_j0_c8_prepare_review_v1(
   v_review_key,'J1',v_ref,repeat('c',64),'Reviewer Local',ids);
  raise exception 'J0_C8_REVIEW_CONFLICT_NOT_DETECTED';
 exception when others then
  if sqlerrm<>'J0_C8_REVIEW_IDEMPOTENCY_CONFLICT' then raise;end if;
 end;
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   'orotitan:c8:decision:local-j1-20261031',(prepared->>'reviewId')::uuid,
   prepared->>'reviewSha256','Approver Local',
   'ACCEPTER J1 SHADOW','GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_ACCEPTANCE_DISABLED_BYPASS';
 exception when others then
  if sqlerrm<>'J0_C8_ACCEPTANCE_LOCKED' then raise;end if;
 end;
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
set role service_role;
do $accept$
declare
 old_id uuid:=current_setting('orotitan.c8_fixture_review')::uuid;
 old_hash text:=current_setting('orotitan.c8_fixture_hash');
 ids uuid[]:=string_to_array(current_setting('orotitan.c8_fixture_proofs'),',')::uuid[];
 newer jsonb; v jsonb; replay jsonb; x jsonb;
 v_ref text:='https://github.com/robzer13/indice_nexus/blob/'||repeat('a',40)||'/docs/audits/J1_SANDBOX_TEST.md';
 v_decision_key text:='orotitan:c8:decision:local-j1-test-20261031';
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
   v_decision_key,old_id,old_hash,'Approver Local','ACCEPTER J1 SHADOW',
   'GO_ACCEPT_J0_WRONG_AUTHORITY');
  raise exception 'J0_C8_AUTHORITY_BYPASS';
 exception when others then
  if sqlerrm<>'J0_C8_AUTHORITY_REF_INVALID' then raise;end if;
 end;
 -- New event makes the earlier review stale.
 v:=public.orotitan_j0_record_event_v1(
  'orotitan:j0:c8-new-j1-proof-for-staleness',
  'J1','EVIDENCE_RECORDED',
  'New evidence invalidates previous review',
  'test:j1-sandbox-c8-staleness',NULL,
  '{"criterionIndex":4}'::jsonb);
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,old_id,old_hash,'Approver Local','ACCEPTER J1 SHADOW',
   'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
  raise exception 'J0_C8_STALE_REVIEW_ACCEPTED';
 exception when others then
  if sqlerrm<>'J0_C8_REVIEW_STALE' then raise;end if;
 end;
 -- New sealed review under the new watermark.
 newer:=public.orotitan_j0_c8_prepare_review_v1(
  'orotitan:c8:review:local-j1-after-staleness-20261031',
  'J1',v_ref,repeat('b',64),'Reviewer Local',ids);
 v:=public.orotitan_j0_c8_accept_review_v1(
  v_decision_key,(newer->>'reviewId')::uuid,newer->>'reviewSha256',
  'Approver Local','ACCEPTER J1 SHADOW',
  'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
 if v->>'status'<>'ACCEPTED' or v->>'accepted'<>'true'
  or v->>'stageStarted'<>'false' or v->>'published'<>'false'
  or (select count(*) from public.orotitan_j0_c8_decisions_v1)<>1
  or (select count(*) from public.orotitan_run_stages)<>0
 then raise exception 'J0_C8_SIMULATED_ACCEPT_FAILED %',v;end if;
 replay:=public.orotitan_j0_c8_accept_review_v1(
  v_decision_key,(newer->>'reviewId')::uuid,newer->>'reviewSha256',
  'Approver Local','ACCEPTER J1 SHADOW',
  'GO_ACCEPT_J0_SANDBOX_ISOLATED_TEST');
 if replay->>'status'<>'REPLAYED' or replay->>'decisionId'<>v->>'decisionId'
 then raise exception 'J0_C8_DECISION_REPLAY_FAILED';end if;
 begin
  perform public.orotitan_j0_c8_accept_review_v1(
   v_decision_key,(newer->>'reviewId')::uuid,newer->>'reviewSha256',
   'Approver Forged','ACCEPTER J1 SHADOW',
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
 x:=public.orotitan_j0_c8_read_gate_v1('J1');
 if x->>'accepted'<>'true' or x->>'acceptanceAllowedNow'<>'false' then
  raise exception 'J0_C8_ACCEPTED_PROJECTION_INVALID %',x;end if;
 if (select count(*) from public.orotitan_j0_c8_decisions_v1
     where milestone_code='J0')<>0 then raise exception 'J0_REAL_ACCEPTANCE_IN_LOCAL_TEST';end if;
 raise notice 'J0_C8_TESTS_STALE_WRONG_GO_ACCEPT_REPLAY_IMMUTABLE_PASS';
end $accept$;
reset role;
SQL
# Re-entrant migration does not clear the isolated fixture decision,
# nor does it enable any new real-world environment.
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_j0_c8_human_acceptance_shadow.sql" >/dev/null
test "$(psql -XAt -d "$db" -c 'select count(*) from public.orotitan_j0_c8_decisions_v1')" = "1"
test "$(psql -XAt -d "$db" -c "select count(*) from public.orotitan_j0_c8_decisions_v1 where milestone_code='J0'")" = "0"
echo "J0 C8 LOCAL PG PASS: locked-by-default, 7-proof review, replay/conflict, stale, explicit GO, simulated J1 approval, immutable decision and migration idempotence"
