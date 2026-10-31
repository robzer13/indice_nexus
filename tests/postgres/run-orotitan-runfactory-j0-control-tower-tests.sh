#!/usr/bin/env bash
# J0 Control Tower PostgreSQL functional matrix on disposable LOCAL PG only.
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
psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL'
set role service_role;
do $matrix$
declare
 first jsonb; again jsonb; opened jsonb; closed jsonb; attempt jsonb;
begin
 if (select count(*) from public.orotitan_j0_milestones_v1)<>9 or
    (select count(*) from public.orotitan_j0_events_v1)<>2 or
    (select count(*) from public.orotitan_j0_milestones_v1 where initial_status='ACCEPTE')<>0
 then raise exception 'J0_SEED_INTEGRITY_FAILED';end if;
 first:=public.orotitan_j0_record_event_v1(
  'orotitan:j0:unit-proof-1','J0','EVIDENCE_RECORDED',
  'Evidence submitted by operator','https://github.com/example/commit',
  null,'{"trace":"manual-only"}'::jsonb);
 again:=public.orotitan_j0_record_event_v1(
  'orotitan:j0:unit-proof-1','J0','EVIDENCE_RECORDED',
  'Evidence submitted by operator','https://github.com/example/commit',
  null,'{"trace":"manual-only"}'::jsonb);
 if first->>'status'<>'CREATED' or again->>'status'<>'REPLAYED' or
    first->>'event_id'<>again->>'event_id' or
    first->>'verification_state'<>'REPORTED' then
  raise exception 'J0_EVIDENCE_IDEMPOTENCY_FAILED';end if;
 begin
  perform public.orotitan_j0_record_event_v1(
   'orotitan:j0:unit-proof-1','J0','EVIDENCE_RECORDED',
   'Different evidence title','https://github.com/example/commit',
   null,'{"trace":"manual-only"}'::jsonb);
  raise exception 'J0_CONFLICT_EXPECTED';
 exception when others then
  if sqlerrm<>'J0_IDEMPOTENCY_CONFLICT' then raise;end if;
 end;
 opened:=public.orotitan_j0_record_event_v1(
  'orotitan:j0:unit-issue-open','J0','BLOCKER_OPENED',
  'HTTP 403 browser not tested','issue:j0-http-check',
  null,'{}'::jsonb);
 closed:=public.orotitan_j0_record_event_v1(
  'orotitan:j0:unit-issue-close','J0','BLOCKER_RESOLVED',
  'HTTP 403 browser verified','issue:j0-http-check',
  null,'{}'::jsonb);
 if opened->>'status'<>'CREATED' or closed->>'status'<>'CREATED' then
  raise exception 'J0_ISSUE_MATRIX_FAILED';end if;
 begin
  perform public.orotitan_j0_record_event_v1(
   'orotitan:j0:unit-issue-close-2','J0','BLOCKER_RESOLVED',
   'Duplicate closure denied','issue:j0-http-check',null,'{}'::jsonb);
  raise exception 'J0_DUPLICATE_ISSUE_CLOSE_EXPECTED';
 exception when others then
  if sqlerrm<>'J0_ISSUE_STATE_CONFLICT' then raise;end if;
 end;
 begin
  perform public.orotitan_j0_record_event_v1(
   'orotitan:j0:unit-accept','J0','MILESTONE_ACCEPTED',
   'Fake acceptance blocked','https://example.org/not-authorized',
   null,'{}'::jsonb);
  raise exception 'J0_ACCEPTANCE_BYPASS_EXPECTED';
 exception when others then
  if sqlerrm<>'J0_EVENT_INVALID' then raise;end if;
 end;
 if (select count(*) from public.orotitan_j0_events_v1)<>5 or
    (select count(*) from public.orotitan_j0_milestones_v1 where initial_status='ACCEPTE')<>0 then
  raise exception 'J0_UNINTENDED_WRITE';end if;
 if has_table_privilege('anon','public.orotitan_j0_events_v1','SELECT')
  or has_table_privilege('authenticated','public.orotitan_j0_milestones_v1','SELECT')
  or has_table_privilege('service_role','public.orotitan_j0_events_v1','UPDATE')
  or has_table_privilege('service_role','public.orotitan_j0_milestones_v1','UPDATE')
  or has_function_privilege('anon',
   'public.orotitan_j0_record_event_v1(text,text,text,text,text,uuid,jsonb)','EXECUTE')
 then raise exception 'J0_ACL_EXPOSURE';end if;
 raise notice 'J0_LOCAL_PG_MATRIX_PASSED';
end $matrix$;
reset role;
SQL
# Rerunning an immutable schema/seed must not duplicate event receipts.
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_j0_control_tower_shadow.sql" >/dev/null
test "$(psql -XAt -d "$db" -c 'select count(*) from public.orotitan_j0_milestones_v1')" = "9"
test "$(psql -XAt -d "$db" -c 'select count(*) from public.orotitan_j0_events_v1')" = "5"
echo "J0 POSTGRES PASS: 9 milestones, immutable approval/start, event receipt, idempotency, conflict, incident closure, denied acceptance, ACLs, reentrant migration"
