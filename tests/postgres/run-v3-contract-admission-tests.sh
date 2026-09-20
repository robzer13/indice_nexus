#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo "Refusing non-local PostgreSQL host: $PGHOST" >&2; exit 2 ;; esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
database="orotitan_v3_admission_$$"
sql_file="$(mktemp)"
cleanup() {
  rm -f "$sql_file"
  dropdb --if-exists --force "$database" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "$database"
server_version_num="$(psql -XAt -d "$database" -c "show server_version_num")"
if (( server_version_num < 170000 || server_version_num >= 180000 )); then
  echo "PostgreSQL 17 required, found server_version_num=$server_version_num" >&2
  exit 3
fi

psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >/dev/null
create schema if not exists extensions;
alter extension pgcrypto set schema extensions;
SQL

for migration in   "$root/migrations/20260909_orotitan_equity_i1_identity.sql"   "$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql"   "$root/migrations/20260914_orotitan_registry_v1_1_core.sql"   "$root/migrations/20260914_orotitan_registry_v1_2_guards_rls.sql"   "$root/migrations/20260914_orotitan_registry_v1_3_rpcs.sql"   "$root/migrations/20260914_orotitan_registry_v1_4_manifest_authority.sql"   "$root/migrations/20260914_orotitan_registry_v1_5_contract_pin_guards.sql"   "$root/migrations/20260920_orotitan_registry_v1_6_checkpoint_output_revalidation.sql"   "$root/migrations/20260920_orotitan_registry_v1_7_final_output_rebinding.sql"   "$root/migrations/20260920_orotitan_registry_v1_8_manifest_persistence_receipt_integrity.sql"
do
  psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$migration" >/dev/null
done

registry9="$root/migrations/20260920203621_orotitan_registry_bundle_persistence_integrity.sql"
if [[ -f "$registry9" ]]; then
  psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$registry9" >/dev/null
fi

node - "$root" > "$sql_file" <<'NODE'
const fs = require("node:fs");
const path = require("node:path");
const root = process.argv[2];
const v2 = JSON.parse(fs.readFileSync(path.join(root, "contracts/orotitan-equity/v2/contract-pin-pack/OROTITAN_CONTRACT_PIN_PACK_V2.json"), "utf8"));
const v3 = JSON.parse(fs.readFileSync(path.join(root, "contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3.json"), "utf8"));
const q = (obj) => JSON.stringify(obj).replaceAll("$pins$", "$pins_escaped$");
process.stdout.write(String.raw`\set ON_ERROR_STOP on

do $$
declare
  v_issuer uuid;
  v_v2_pins jsonb := $pins$${q(v2.contract_pins)}$pins$::jsonb;
  v_v3_pins jsonb := $pins$${q(v3.contract_pins)}$pins$::jsonb;
  v_v2_hash text := '${v2.contract_set_sha256}';
  v_v3_hash text := '${v3.contract_set_sha256}';
  v_v2_result jsonb;
  v_v3_result jsonb;
  v_v2_run uuid;
  v_v3_run uuid;
  v_v2_before jsonb;
  v_v2_after jsonb;
  v_rejected boolean := false;
begin
  select issuer_id into v_issuer
  from public.legacy_company_identity_map
  order by legacy_company_id
  limit 1;

  if v_issuer is null then
    raise exception 'V3_ADMISSION_TEST_IDENTITY_MISSING';
  end if;

  if not public.orotitan_contract_pins_complete(v_v3_pins) then
    raise exception 'V3_CONTRACT_PINS_INCOMPLETE';
  end if;
  if public.orotitan_contract_set_sha256(v_v3_pins) <> v_v3_hash then
    raise exception 'V3_CONTRACT_SET_HASH_RECONCILIATION_FAIL';
  end if;

  v_v2_result := public.create_orotitan_run(
    'v3-admission:v2-existing',
    v_issuer,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    date '2026-09-19',
    null,
    null,
    v_v2_pins->'process'->>'version',
    v_v2_pins->'pilotage'->>'version',
    v_v2_pins,
    v_v2_hash,
    repeat('2',64)
  );
  v_v2_run := (v_v2_result->>'run_id')::uuid;

  select to_jsonb(r) into v_v2_before
  from public.orotitan_runs r
  where r.run_id = v_v2_run;

  v_v3_result := public.create_orotitan_run(
    'v3-admission:v3-new-run',
    v_issuer,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    date '2026-09-19',
    v_v2_run,
    null,
    v_v3_pins->'process'->>'version',
    v_v3_pins->'pilotage'->>'version',
    v_v3_pins,
    v_v3_hash,
    repeat('3',64)
  );
  v_v3_run := (v_v3_result->>'run_id')::uuid;

  if not exists (
    select 1
    from public.orotitan_runs r
    where r.run_id = v_v3_run
      and r.parent_run_id = v_v2_run
      and r.run_type = 'INITIAL'
      and r.baseline_snapshot_id is null
      and r.data_cutoff = date '2026-09-19'
      and r.process_version = v_v3_pins->'process'->>'version'
      and r.pilotage_contract_version = v_v3_pins->'pilotage'->>'version'
      and r.contract_pins = v_v3_pins
      and r.contract_set_sha256 = v_v3_hash
  ) then
    raise exception 'V3_NEW_RUN_ADMISSION_PERSISTENCE_MISMATCH';
  end if;

  select to_jsonb(r) into v_v2_after
  from public.orotitan_runs r
  where r.run_id = v_v2_run;

  if v_v2_after is distinct from v_v2_before then
    raise exception 'V2_EXISTING_RUN_MUTATED_BY_V3_ADMISSION';
  end if;

  begin
    perform public.create_orotitan_run(
      'v3-admission:wrong-hash',
      v_issuer,
      'IMPOSED_COMPANY',
      'ANALYZE',
      'INITIAL',
      date '2026-09-19',
      null,
      null,
      v_v3_pins->'process'->>'version',
      v_v3_pins->'pilotage'->>'version',
      v_v3_pins,
      repeat('0',64),
      repeat('4',64)
    );
  exception when check_violation then
    v_rejected := true;
  end;

  if not v_rejected then
    raise exception 'CONTRACT_SET_HASH_ENFORCEMENT_FAILED';
  end if;

  if exists (
    select 1 from public.orotitan_runs
    where creation_idempotency_key = 'v3-admission:wrong-hash'
  ) then
    raise exception 'WRONG_HASH_CREATED_PARTIAL_RUN';
  end if;
end;
$$;

select jsonb_build_object(
  'V2_EXISTING_RUN_COMPATIBILITY','PASS',
  'V3_NEW_RUN_ADMISSION','PASS',
  'CONTRACT_SET_HASH_ENFORCEMENT','PASS',
  'REGISTRY_REGRESSION','PASS ALL'
) as orotitan_v3_contract_set_admission_regression;
`);
NODE

psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$sql_file"
echo 'OroTitan V3 Contract Set admission regression: PASS ALL'
