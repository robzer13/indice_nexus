#!/usr/bin/env bash
# P0-E real PostgreSQL local integration. Test data and forged byte proofs
# are strictly contained in a disposable PG17 database, NEVER SHADOW/PROD.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1|/var/run/postgresql) ;;
 *) echo "REFUSED NON-LOCAL DATABASE" >&2; exit 2;; esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_p0e_$$"
trap 'dropdb --if-exists --force "$db" >/dev/null 2>&1 || true' EXIT
createdb "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create schema if not exists extensions; alter extension pgcrypto set schema extensions' >/dev/null
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
psql -X -v ON_ERROR_STOP=1 -d "$db" -c "update public.securities set isin='FR0000120321',primary_listing=true where ticker='BET' and exchange='XPAR'" >/dev/null
pins="$(node -e "process.stdout.write(JSON.stringify(require('$root/contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json').contract_pins))")"
psql -X -v ON_ERROR_STOP=1 -v pins="$pins" -d "$db" <<'SQL'
select set_config('orotitan.p0e_test_pins', :'pins',false);
set role service_role;
do $matrix$
declare
 v_issuer uuid;
 v_security uuid;
 v_dossier uuid;
 v_pins jsonb:=current_setting('orotitan.p0e_test_pins')::jsonb;
 v_proof jsonb;
 v_sha text:='3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2';
 v_fingerprint text:=repeat('a',64);
 v_result jsonb;
 v_replay jsonb;
 v_run uuid;
 v_count integer;
begin
 select i.issuer_id,s.security_id,d.dossier_id
  into strict v_issuer,v_security,v_dossier
 from public.issuers i join public.securities s on s.issuer_id=i.issuer_id
  join public.research_dossiers d on d.issuer_id=i.issuer_id and d.active
 where s.ticker='BET' and s.exchange='XPAR';
 select jsonb_build_object(
  'profileId','OROTITAN_V3_1','contractSetSha256',v_sha,
  'pinCount',15,'passedPinCount',15,'allOriginalBytesVerified',true,
  'registryMutated',false,'checkedAt',now(),
  'checks',jsonb_agg(jsonb_build_object('logicalName',key,
    'sha256',value->>'content_sha256','byteLength',100,
    'status','ORIGINAL_CONTRACT_BYTES_VERIFIED'))) into v_proof
 from jsonb_each(v_pins);
 -- T1: full initial creation and binding, no stage/publish/snapshot.
 v_result:=public.orotitan_shadow_create_initial_run_v3_1(
  v_issuer,v_security,v_dossier,'FR0000120321','XPAR','2026-10-30',
  v_pins,v_sha,'orotitan:v3:create:'||v_fingerprint,v_fingerprint,
  v_proof,'GO P0-E SHADOW FR0000120321 2026-10-30');
 v_run:=(v_result->>'run_id')::uuid;
 if v_result->>'status'<>'CREATED' or v_result->>'atomic_creation_and_binding'<>'true'
 or v_result->>'run_stage_started'<>'false' or
 (select state_version from public.orotitan_runs where run_id=v_run)<>2 or
 (select count(*) from public.orotitan_run_events where run_id=v_run)<>2 or
 (select count(*) from public.orotitan_run_stages where run_id=v_run)<>0 or
 (select count(*) from public.orotitan_p0e_run_admission_receipts_v1 where run_id=v_run)<>1
 then raise exception 'P0E_T1_FAILED %',v_result; end if;
 -- T2: repeat is idempotent.
 v_replay:=public.orotitan_shadow_create_initial_run_v3_1(
  v_issuer,v_security,v_dossier,'FR0000120321','XPAR','2026-10-30',
  v_pins,v_sha,'orotitan:v3:create:'||v_fingerprint,v_fingerprint,
  v_proof,'GO P0-E SHADOW FR0000120321 2026-10-30');
 if v_replay->>'status'<>'REPLAYED' or v_replay->>'run_id'<>v_run::text or
    (select count(*) from public.orotitan_runs where issuer_id=v_issuer)<>1 or
    (select count(*) from public.orotitan_run_events where run_id=v_run)<>2
 then raise exception 'P0E_T2_REPLAY_FAILED %',v_replay; end if;
 -- T3: conflicting same-cutoff idempotency key fails closed.
 begin
  perform public.orotitan_shadow_create_initial_run_v3_1(
   v_issuer,v_security,v_dossier,'FR0000120321','XPAR','2026-10-30',
   v_pins,v_sha,'orotitan:v3:create:'||repeat('b',64),repeat('b',64),
   v_proof,'GO P0-E SHADOW FR0000120321 2026-10-30');
  raise exception 'P0E_T3_EXPECTED_FAIL_MISSING';
 exception when others then
  if sqlerrm <> 'P0E_EXISTING_RUN_CONFLICT' then raise; end if;
 end;
 -- T4: incomplete/forged byte proof fails closed.
 begin
  perform public.orotitan_shadow_create_initial_run_v3_1(
   v_issuer,v_security,v_dossier,'FR0000120321','XPAR','2026-10-30',
   v_pins,v_sha,'orotitan:v3:create:'||v_fingerprint,v_fingerprint,
   v_proof||jsonb_build_object('passedPinCount',14),
   'GO P0-E SHADOW FR0000120321 2026-10-30');
  raise exception 'P0E_T4_EXPECTED_FAIL_MISSING';
 exception when others then
  if sqlerrm <> 'P0E_ORIGINAL_BYTES_RECEIPT_REQUIRED' then raise; end if;
 end;
 -- T5: GO bound to ISIN/cutoff.
 begin
  perform public.orotitan_shadow_create_initial_run_v3_1(
   v_issuer,v_security,v_dossier,'FR0000120321','XPAR','2026-10-30',
   v_pins,v_sha,'orotitan:v3:create:'||v_fingerprint,v_fingerprint,
   v_proof,'GO WRONG');
  raise exception 'P0E_T5_EXPECTED_FAIL_MISSING';
 exception when others then
  if sqlerrm <> 'P0E_EXPLICIT_GO_REQUIRED' then raise; end if;
 end;
 if (select count(*) from public.orotitan_runs where issuer_id=v_issuer)<>1 or
   (select count(*) from public.orotitan_p0e_run_admission_receipts_v1 where issuer_id=v_issuer)<>1 then
   raise exception 'P0E_T6_FAILED_UNINTENDED_WRITES';
 end if;
 if has_function_privilege('anon',
    'public.orotitan_shadow_create_initial_run_v3_1(uuid,uuid,uuid,text,text,date,jsonb,text,text,text,jsonb,text)','EXECUTE')
 or has_function_privilege('authenticated',
    'public.orotitan_shadow_create_initial_run_v3_1(uuid,uuid,uuid,text,text,date,jsonb,text,text,text,jsonb,text)','EXECUTE')
 or not has_function_privilege('service_role',
    'public.orotitan_shadow_create_initial_run_v3_1(uuid,uuid,uuid,text,text,date,jsonb,text,text,text,jsonb,text)','EXECUTE')
 then raise exception 'P0E_T7_ACCESS_CONTROL_FAILED';end if;
 raise notice 'P0E_T1_TO_T7_PASSED';
end $matrix$;
reset role;
SQL
before="$(psql -XAt -d "$db" -c 'select count(*) from public.orotitan_p0e_run_admission_receipts_v1')"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_p0e_shadow_initial_run.sql" >/dev/null
test "$before" = "$(psql -XAt -d "$db" -c 'select count(*) from public.orotitan_p0e_run_admission_receipts_v1')"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_p0e_shadow_helper_grants.sql" >/dev/null
echo "P0E POSTGRES PASS: atomic initial run + bind + events + readback + receipt + no stage + replay + conflict + denied GO + denied bytes + privileges + idempotent migration"
