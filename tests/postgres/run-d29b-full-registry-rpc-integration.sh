#!/usr/bin/env bash
# D29b: Full real Registry V3.1 migrations/RPC + synthetic Research quarantine.
# DISPOSABLE PG17 ONLY. Never deploy SQL proposals or use Supabase production.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo 'D29b full Registry: refusing remote PGHOST' >&2; exit 2 ;;
esac
[[ "$PGUSER" == postgres && "$PGPORT" == 5432 ]] || {
  echo 'D29b full Registry: refusing non-fixture PGUSER/PGPORT' >&2; exit 2;
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d29b_real_registry_$$"
created=0
roles_created=0
cleanup(){
  if [[ "$created" == 1 ]]; then
    dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
  fi
  if [[ "$roles_created" == 1 ]]; then
    psql -X -v ON_ERROR_STOP=1 -d postgres -c \
      'DROP ROLE IF EXISTS orotitan_research_intake_writer,orotitan_research_intake_reader,orotitan_research_intake_receipt_executor;' \
      >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
other="$(psql -XAt -d postgres -c "SELECT count(*) FROM pg_roles WHERE rolname IN (
  'orotitan_research_intake_writer','orotitan_research_intake_reader',
  'orotitan_research_intake_receipt_executor')")"
[[ "$other" == 0 ]] || { echo 'D29b: preexisting private roles, aborting' >&2; exit 2; }
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'SHOW server_version_num')"
((version >= 170000 && version < 180000)) || {
  echo 'D29b: PG17 required' >&2; exit 3;
}
psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL' > /dev/null
CREATE SCHEMA IF NOT EXISTS extensions;
ALTER EXTENSION pgcrypto SET SCHEMA extensions;
SQL
for f in \
  migrations/20260909_orotitan_equity_i1_identity.sql \
  migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql \
  migrations/20260914_orotitan_registry_v1_1_core.sql \
  migrations/20260914_orotitan_registry_v1_2_guards_rls.sql \
  migrations/20260914_orotitan_registry_v1_3_rpcs.sql \
  migrations/20260914_orotitan_registry_v1_4_manifest_authority.sql \
  migrations/20260914_orotitan_registry_v1_5_contract_pin_guards.sql \
  migrations/20260920_orotitan_registry_v1_6_checkpoint_output_revalidation.sql \
  migrations/20260920_orotitan_registry_v1_7_final_output_rebinding.sql \
  migrations/20260920_orotitan_registry_v1_8_manifest_persistence_receipt_integrity.sql \
  migrations/20260920203621_orotitan_registry_bundle_persistence_integrity.sql \
  migrations/20260920_orotitan_registry_v1_9_attestation_idempotency_fix.sql \
  migrations/20260921_orotitan_registry_v1_10_methodology_successor_cas.sql \
  migrations/20260921_orotitan_registry_v1_11_persistence_attestation_authority_repair.sql \
  migrations/20260923_orotitan_registry_v1_12_valuation_date_alignment_successor.sql; do
  psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/$f" >/dev/null
done
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
  "$root/docs/orotitan-equity/proposals/D11_V31_RESEARCH_FINAL_REGISTRY_V2_SQL_PROPOSED.sql" >/dev/null
# This produces a genuine CHECKPOINT, FINAL and downstream Deep Dive RPC
# history on a separate synthetic run. Do not create its source bytes.
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
  "$root/tests/postgres/research-v31-handoff-registry-e2e.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
  "$root/tests/postgres/d29b-full-registry-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
  "$root/docs/orotitan-equity/proposals/D29B_RESEARCH_PRIVATE_RECEIPT_SQL_REVIEW_ONLY.sql" >/dev/null
roles_created=1
bash "$root/tests/postgres/d29b-real-registry-rpc-race.sh" "$db"
echo 'D29b actual V3.1 Registry RPC integration passed in disposable PostgreSQL 17.'
