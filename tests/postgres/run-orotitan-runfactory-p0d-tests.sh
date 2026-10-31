#!/usr/bin/env bash
# LOCAL / DISPOSABLE PostgreSQL ONLY. Never run against SHADOW/production.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in
  127.0.0.1|localhost|::1|/var/run/postgresql) ;;
  *) echo "P0-D REFUSED NON-LOCAL POSTGRES: $PGHOST" >&2; exit 2 ;;
esac
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
prefix="orotitan_p0d_$$_"
dbs=()
cleanup(){
 for db in "${dbs[@]}"; do dropdb --if-exists --force "$db" >/dev/null 2>&1 || true; done
}
trap cleanup EXIT
new_db(){
 local db="${prefix}$1"
 createdb "$db"; dbs+=("$db")
 psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/tests/postgres/i1-fixture.sql" >/dev/null
 psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20260909_orotitan_equity_i1_identity.sql" >/dev/null
 psql -X -v ON_ERROR_STOP=1 -d "$db" -c 'create schema if not exists extensions; alter extension pgcrypto set schema extensions' >/dev/null
 psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$root/migrations/20261031_orotitan_runfactory_p0d_atomic_identity_onboarding.sql" >/dev/null
 printf '%s' "$db"
}
main="$(new_db main)"
psql -X -v ON_ERROR_STOP=1 -d "$main" -f "$root/tests/postgres/runfactory-p0d-onboarding-verify.sql" >/dev/null
before="$(psql -XAt -d "$main" -c 'select count(*) from public.orotitan_identity_onboarding_receipts_v1')"
psql -X -v ON_ERROR_STOP=1 -d "$main" -f "$root/migrations/20261031_orotitan_runfactory_p0d_atomic_identity_onboarding.sql" >/dev/null
after="$(psql -XAt -d "$main" -c 'select count(*) from public.orotitan_identity_onboarding_receipts_v1')"
test "$before" = "$after"

# Two simultaneous requests with different idempotency keys for the SAME
# ISIN/listing must converge to one issuer, one security and one dossier.
race="$(new_db race)"
call(){
 local db="$1" key="$2" isin="$3"
 psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "
 select public.orotitan_onboard_identity_shadow_v1(
  '$key','$isin','CON','XPAR',
  'Concurrent Fixture','Concurrent Fixture Inc.','US','USD','USD',
  true,'MAJOR',2,null,null,
  jsonb_build_object('issuer_source_url','https://example.com/issuer',
   'listing_source_url','https://example.org/listing',
   'method','OPERATOR_PRIMARY_SOURCE_REVIEW',
   'reviewed_by','ADMIN_SESSION_MANUAL_ATTESTATION_V1',
   'observed_at','2026-10-31T12:00:00Z'),
  'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');"
}
call "$race" 'orotitan:p0d:race:same:one' 'US02079K3059' >"/tmp/p0d-race1-$$.log" 2>&1 &
pid1=$!
call "$race" 'orotitan:p0d:race:same:two' 'US02079K3059' >"/tmp/p0d-race2-$$.log" 2>&1 &
pid2=$!
wait "$pid1"
wait "$pid2"
test "$(psql -XAt -d "$race" -c "select count(*) from public.securities where ticker='CON' and exchange='XPAR'")" = "1"
test "$(psql -XAt -d "$race" -c "select count(*) from public.issuers where display_name='Concurrent Fixture'")" = "1"
test "$(psql -XAt -d "$race" -c "select count(*) from public.research_dossiers")" = "3"
test "$(psql -XAt -d "$race" -c "select count(*) from public.orotitan_identity_onboarding_receipts_v1")" = "2"

# Same listing but conflicting ISIN must never both succeed.
conflict="$(new_db conflict)"
set +e
call "$conflict" 'orotitan:p0d:race:conflict:one' 'FR0000120321' >"/tmp/p0d-conflict1-$$.log" 2>&1 &
pid1=$!
call "$conflict" 'orotitan:p0d:race:conflict:two' 'NL0010273215' >"/tmp/p0d-conflict2-$$.log" 2>&1 &
pid2=$!
wait "$pid1"; status1=$?
wait "$pid2"; status2=$?
set -e
if ! { { test "$status1" = 0 && test "$status2" != 0; } ||
        { test "$status1" != 0 && test "$status2" = 0; }; }; then
 echo "CONCURRENCY CONFLICT CHECK FAILED" >&2
 cat "/tmp/p0d-conflict1-$$.log" "/tmp/p0d-conflict2-$$.log" >&2
 exit 1
fi
test "$(psql -XAt -d "$conflict" -c "select count(*) from public.securities where ticker='CON' and exchange='XPAR'")" = "1"
test "$(psql -XAt -d "$conflict" -c 'select count(*) from public.orotitan_identity_onboarding_receipts_v1')" = "1"
rm -f "/tmp/p0d-race1-$$.log" "/tmp/p0d-race2-$$.log" "/tmp/p0d-conflict1-$$.log" "/tmp/p0d-conflict2-$$.log"
echo "P0D POSTGRES PASS: 15 functional matrix checks + idempotent migration + dual-submit concurrency + conflicting-ISIN concurrency"
