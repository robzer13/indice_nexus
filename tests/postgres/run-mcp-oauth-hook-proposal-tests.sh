#!/usr/bin/env bash
# D27: isolated PostgreSQL 17 ONLY. Never connect to a remote Supabase DB.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo 'D27: refusing non-loopback PGHOST' >&2; exit 2;;
esac
if [[ "$PGUSER" != "postgres" ]]; then
  echo 'D27: refusing non-fixture PGUSER' >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d27_mcp_oauth_$$"
tmp="$(mktemp)"
created=0
cleanup() {
  rm -f "$tmp"
  if [[ "$created" -eq 1 ]]; then dropdb --if-exists --force "$db" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'show server_version_num')"
if (( version < 170000 || version >= 180000 )); then
  echo "D27: PostgreSQL 17 required, got $version" >&2; exit 3
fi

psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL' >/dev/null
do $fixture$
begin
  if not exists(select 1 from pg_roles where rolname='supabase_auth_admin') then
    create role supabase_auth_admin nologin;
  end if;
  if not exists(select 1 from pg_roles where rolname='authenticated') then
    create role authenticated nologin;
  end if;
  if not exists(select 1 from pg_roles where rolname='anon') then
    create role anon nologin;
  end if;
end;
$fixture$;
SQL
sed -e 's/__APPROVED_SUPABASE_AUTH_USER_UUID__/11111111-1111-4111-8111-111111111111/g' \
    -e 's/__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__/approved-client-test-only/g' \
    "$root/docs/orotitan-equity/proposals/D27_MCP_OAUTH_RESOURCE_CLAIM_HOOK_REVIEW_ONLY.sql" > "$tmp"

psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$tmp" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/tests/postgres/mcp-oauth-hook-proposal-verify.sql" >/dev/null

echo 'D27 OAuth resource claim hook PG17 positive/negative tests PASS; production unchanged.'
