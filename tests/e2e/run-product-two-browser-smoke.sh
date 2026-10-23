#!/usr/bin/env bash
# Real Chromium acceptance on TWO independent browser contexts + disposable PG17.
# This script MUST NOT be run against any Supabase-hosted or production database.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
[[ "${CI:-}" == "true" ]] || { echo "Requires isolated CI" >&2; exit 2; }
case "$PGHOST" in localhost|127.0.0.1) ;; *) exit 2 ;; esac
[[ "$PGUSER" == "postgres" && "$PGPORT" == "5432" ]] || exit 2
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
db="orotitan_product_browser_${$}"
rest_pid=""; next_pid=""
cleanup() {
  if [[ -n "$next_pid" ]]; then kill "$next_pid" 2>/dev/null || true; fi
  if [[ -n "$rest_pid" ]]; then kill "$rest_pid" 2>/dev/null || true; fi
  dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
}
trap cleanup EXIT
createdb "$db"
version="$(psql -XAt -d "$db" -c 'SHOW server_version_num')"
[[ "$version" =~ ^17[0-9]{4}$ ]] || { echo "Requires PostgreSQL 17" >&2; exit 2; }
psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null <<'SQL'
DO $roles$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='anon') THEN
   CREATE ROLE anon NOLOGIN;
 END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='authenticated') THEN
   CREATE ROLE authenticated NOLOGIN;
 END IF;
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname='service_role') THEN
   CREATE ROLE service_role NOLOGIN BYPASSRLS;
 END IF;
END;
$roles$;
SQL
psql -X -v ON_ERROR_STOP=1 -d "$db" -f schema.sql >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 docs/orotitan-equity/proposals/PRODUCT_SHARED_RESEARCH_INBOX_PG17_REVIEW_ONLY.sql >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null <<'SQL'
INSERT INTO public.companies (
 id,slug,ticker,name,exchange,currency,quote_unit,price_decimals,
 market_data_multiplier,active
) VALUES
('00000000-0000-4000-8000-000000000091','test-alpha','TAL',
 'Synthetic Research Alpha','XPAR','EUR','MAJOR',2,1,true),
('00000000-0000-4000-8000-000000000092','test-beta','TBE',
 'Synthetic Research Beta','XAMS','EUR','MAJOR',2,1,true);
SQL

export OROTITAN_PRODUCT_BROWSER_FIXTURE=1
export OROTITAN_SHARED_INBOX_TEST_ENABLED=1
export OROTITAN_LOCAL_REQUESTS_ENABLED=0
export OROTITAN_SHARED_INBOX_TEST_PROJECT_REF=local-browser-fixture
export OROTITAN_SHARED_INBOX_TEST_URL=http://127.0.0.1:48761
export NEXT_PUBLIC_SUPABASE_URL=http://127.0.0.1:48761
export OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY='fake-ci-browser-service-key-only-0123456789-DO-NOT-USE'
export SUPABASE_SERVICE_ROLE_KEY="$OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY"
export ADMIN_PASSWORD='synthetic-product-acceptance-admin-password'
export ADMIN_SESSION_SECRET='synthetic-product-acceptance-session-secret-32chars'
export NODE_ENV=development
export NEXT_TELEMETRY_DISABLED=1

node tests/e2e/product-shared-postgrest-fixture.mjs "$db" > /tmp/product-browser-rest.log 2>&1 &
rest_pid="$!"
for i in $(seq 1 35); do
  if grep -q PRODUCT_BROWSER_REST_READY /tmp/product-browser-rest.log; then break; fi
  if ! kill -0 "$rest_pid" 2>/dev/null; then cat /tmp/product-browser-rest.log; exit 1; fi
  sleep 1
done
grep -q PRODUCT_BROWSER_REST_READY /tmp/product-browser-rest.log

npm run dev -- --port 3184 > /tmp/product-browser-next.log 2>&1 &
next_pid="$!"
for i in $(seq 1 90); do
  status="$(curl --silent --output /dev/null --write-out '%{http_code}' 'http://127.0.0.1:3184/admin' || true)"
  if [[ "$status" == 200 ]]; then break; fi
  if ! kill -0 "$next_pid" 2>/dev/null; then
    tail -120 /tmp/product-browser-next.log; exit 1
  fi
  sleep 1
done
[[ "${status:-}" == 200 ]] || {
 echo "Browser acceptance Next.js dev server unavailable" >&2
 tail -120 /tmp/product-browser-next.log; exit 1
}

if ! node tests/e2e/product-shared-two-browser.mjs; then
  echo 'Product browser test failed; Next.js server log:' >&2
  tail -150 /tmp/product-browser-next.log >&2
  echo 'Test-only REST fixture log:' >&2
  tail -100 /tmp/product-browser-rest.log >&2
  exit 1
fi
request_count="$(psql -XAt -d "$db" -c \
 'SELECT count(*) FROM public.orotitan_research_intake_requests')"
[[ "$request_count" == 1 ]] || exit 1
[[ "$(psql -XAt -d "$db" -c "SELECT to_regclass('public.orotitan_runs') IS NULL")" == t ]] || exit 1
echo 'PRODUCT FULL WEB E2E SUCCESS: two Chromium sessions, real Next admin, shared PG17 request receipt.'
