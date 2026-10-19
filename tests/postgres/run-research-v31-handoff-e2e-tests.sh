#!/usr/bin/env bash
# D11 real V3.1 Registry Research -> Deep Dive transaction, isolated PG17 ONLY.
# Refuses any remote DB or non-fixture user. Never deploys the SQL proposal.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo "D11 E2E: refusing remote PGHOST $PGHOST" >&2; exit 2 ;;
esac
if [[ "$PGUSER" != "postgres" ]]; then
  echo "D11 E2E: refusing non-fixture PGUSER $PGUSER" >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d11_registry_e2e_$$"
created=0
cleanup() {
  if [[ "$created" -eq 1 ]]; then
    dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'show server_version_num')"
if (( version < 170000 || version >= 180000 )); then
  echo "D11 E2E: PostgreSQL 17 required, got $version" >&2; exit 3
fi

psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL' >/dev/null
create schema if not exists extensions;
alter extension pgcrypto set schema extensions;
SQL

for file in \
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
  psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/$file" >/dev/null
done

# Only the test fixture DB sees this proposed authority successor.
psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/docs/orotitan-equity/proposals/D11_V31_RESEARCH_FINAL_REGISTRY_V2_SQL_PROPOSED.sql" >/dev/null

# Includes V1 E2E unchanged AND V3.1 V2 Research E2E with real RPCs.
psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/tests/postgres/research-v31-handoff-registry-e2e.sql"

echo 'D11 V1 regression + real V3.1 Research->Deep Dive Registry E2E PASS in isolated PostgreSQL 17; production UNCHANGED.'
