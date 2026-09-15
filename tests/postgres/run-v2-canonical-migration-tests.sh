#!/usr/bin/env bash
set -euo pipefail

: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *) echo "Refusing non-local PostgreSQL host: $PGHOST" >&2; exit 2 ;; esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
i1="$root/migrations/20260909_orotitan_equity_i1_identity.sql"
i3="$root/migrations/20260911_orotitan_equity_i3_snapshot_persistence.sql"
i3b="$root/migrations/20260911_orotitan_equity_i3b_validated_snapshot_writer.sql"
v2="$root/migrations/20260915_orotitan_v2_canonical_snapshot_writer.sql"
verify="$root/tests/postgres/v2-canonical-verify.sql"
database="orotitan_v2_$$"
cleanup() { dropdb --if-exists --force "$database" >/dev/null 2>&1 || true; }
trap cleanup EXIT

createdb "$database"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$fixture" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i1" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i3" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$i3b" >/dev/null
legacy_before="$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
identity_before="$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$v2" >/dev/null
test "$legacy_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(c)::text, ',' order by c.id) from companies c)||(select string_agg(row_to_json(s)::text, ',' order by s.id) from snapshots s)||(select string_agg(row_to_json(p)::text, ',' order by p.id) from market_prices p)||(select string_agg(row_to_json(r)::text, ',' order by r.id) from market_sync_runs r))")"
test "$identity_before" = "$(psql -XAt -d "$database" -c "select md5((select string_agg(row_to_json(i)::text, ',' order by i.issuer_id) from issuers i)||(select string_agg(row_to_json(s)::text, ',' order by s.security_id) from securities s)||(select string_agg(row_to_json(d)::text, ',' order by d.dossier_id) from research_dossiers d)||(select string_agg(row_to_json(m)::text, ',' order by m.legacy_company_id) from legacy_company_identity_map m))")"
psql -X -v ON_ERROR_STOP=1 -d "$database" -f "$verify"
echo 'OroTitan V2 canonical PostgreSQL integration matrix passed'
