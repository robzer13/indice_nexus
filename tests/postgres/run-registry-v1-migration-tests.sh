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
verify="$root/tests/postgres/registry-v1-verify.sql"
database="orotitan_registry_v1_$$"
cleanup() { dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; }
trap cleanup EXIT

createdb "$database"
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

for migration in "$registry1" "$registry2" "$registry3" "$registry4" "$registry5"; do
  psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$migration" >/dev/null
done

# Registry migrations must not reinterpret or mutate pre-existing legacy or
# canonical identity data.
test "$legacy_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
test "$identity_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"

psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$verify"

# The operational matrix is registry-only and must still leave legacy and
# canonical identity rows byte-equivalent at the row-json level.
test "$legacy_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
test "$identity_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"

echo 'Registry V1 PostgreSQL integration matrix passed'
