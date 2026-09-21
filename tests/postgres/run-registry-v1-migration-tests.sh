#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo "Refusing non-local PostgreSQL host: $PGHOST" >&2; exit 2 ;; esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
i1="$root/migrations/20260909_orotitan_equity_i1_identity.sql"
i3="$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql"
registry1="$root/migrations/20260914_orotitan_registry_v1_1_core.sql"
registry2="$root/migrations/20260914_orotitan_registry_v1_2_guards_rls.sql"
registry3="$root/migrations/20260914_orotitan_registry_v1_3_rpcs.sql"
registry4="$root/migrations/20260914_orotitan_registry_v1_4_manifest_authority.sql"
registry5="$root/migrations/20260914_orotitan_registry_v1_5_contract_pin_guards.sql"
registry6="$root/migrations/20260920_orotitan_registry_v1_6_checkpoint_output_revalidation.sql"
registry7="$root/migrations/20260920_orotitan_registry_v1_7_final_output_rebinding.sql"
registry8="$root/migrations/20260920_orotitan_registry_v1_8_manifest_persistence_receipt_integrity.sql"
registry9="$root/migrations/20260920203621_orotitan_registry_bundle_persistence_integrity.sql"
registry10="$root/migrations/20260920_orotitan_registry_v1_9_attestation_idempotency_fix.sql"
registry11="$root/migrations/20260921_orotitan_registry_v1_10_methodology_successor_cas.sql"
registry12="$root/migrations/20260921_orotitan_registry_v1_11_persistence_attestation_authority_repair.sql"
verify="$root/tests/postgres/registry-v1-verify.sql"
revalidation_verify="$root/tests/postgres/registry-v1-checkpoint-revalidation-verify.sql"
successor_rebinding_verify="$root/tests/postgres/registry-v1-successor-rebinding-verify.sql"
manifest_receipt_verify="$root/tests/postgres/registry-v1-manifest-persistence-receipt-verify.sql"
bundle_receipt_verify="$root/tests/postgres/registry-v1-bundle-persistence-integrity-verify.sql"
successor_cas_verify="$root/tests/postgres/registry-v1-methodology-successor-cas-verify.sql"
attestation_authority_verify="$root/tests/postgres/registry-v1-persistence-attestation-authority-verify.sql"
database="orotitan_registry_v1_$$"
cleanup() { dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; }
trap cleanup EXIT

createdb "$database"
server_version_num="$(psql -XAt -d "$database" -c "show server_version_num")"
if (( server_version_num < 170000 || server_version_num >= 180000 )); then
  echo "PostgreSQL 17 required, found server_version_num=$server_version_num" >&2
  exit 3
fi

# Deployed predecessor migration byte identities are frozen.
test "$(git hash-object "$registry6")" = "e64f635d043482f7f9a1dd695bd3a412f0087ffa"
test "$(git hash-object "$registry7")" = "4d61d4765e27681b4bec3583133c10d08a8e0684"
test "$(git hash-object "$registry8")" = "a5ff07afe47a5f9582069bb88c067109eeb6a61b"

psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$fixture" >/dev/null

# Supabase production installs pgcrypto in the extensions schema. Mirror that
# namespace locally before applying the registry RPCs.
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >/dev/null
create schema if not exists extensions;
alter extension pgcrypto set schema extensions;
SQL

psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i1" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i3" >/dev/null

legacy_before="$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
identity_before="$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"

for migration in "$registry1" "$registry2" "$registry3" "$registry4" "$registry5" "$registry6" "$registry7" "$registry8"; do
  psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$migration" >/dev/null
done

# Registry migrations must not reinterpret or mutate pre-existing legacy or
# canonical identity data.
test "$legacy_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
test "$identity_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"

psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$verify"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$revalidation_verify"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$successor_rebinding_verify"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$manifest_receipt_verify"

# Preserve the historical V1.8 semantic frontier above, then apply exactly one
# generated forward migration and run the bundle-wide receipt closure matrix.
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$registry9" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$registry10" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$registry11" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$registry12" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$attestation_authority_verify"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$bundle_receipt_verify"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$successor_cas_verify"

# The operational matrix is registry-only and must still leave legacy and
# canonical identity rows byte-equivalent at the row-json level.
test "$legacy_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
test "$identity_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"

echo 'Registry V1.6/V1.7/V1.8 + forward bundle persistence + V1.9 + V1.10 + V1.11 attestation authority PostgreSQL 17 regression: PASS ALL'
