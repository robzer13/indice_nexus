#!/usr/bin/env bash
# D28 review-only RLS test. PostgreSQL 17 on LOOPBACK ONLY.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo 'D28 refusing non-loopback PGHOST' >&2; exit 2;;
esac
if [[ "$PGUSER" != "postgres" ]]; then
  echo 'D28 refusing non-fixture PGUSER' >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d28_besi_rls_$$"
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
  echo "D28 requires PostgreSQL 17 (got $version)" >&2; exit 3
fi

psql -X -v ON_ERROR_STOP=1 -d "$db" <<'SQL' >/dev/null
do $roles$
begin
  if not exists (select 1 from pg_roles where rolname='authenticated') then
    create role authenticated nologin;
  end if;
  if not exists (select 1 from pg_roles where rolname='anon') then
    create role anon nologin;
  end if;
end;
$roles$;

create schema auth;
create function auth.jwt() returns jsonb
  language sql stable
  as $$select coalesce(nullif(current_setting('request.jwt.claims',true),''),'{}')::jsonb$$;
create function auth.uid() returns uuid
  language sql stable
  as $$select nullif(auth.jwt()->>'sub','')::uuid$$;
grant usage on schema auth,public to authenticated,anon;
grant execute on function auth.jwt() to authenticated;
grant execute on function auth.uid() to authenticated;

create table public.orotitan_runs(
  run_id uuid primary key, issuer_id uuid, security_id uuid,
  run_status text, current_stage text, state_version bigint,
  data_cutoff date, process_version text,
  pilotage_contract_version text, canonical_mode text,
  creation_idempotency_key text
);
create table public.orotitan_run_stages(
  run_id uuid, stage_code text, lifecycle_status text,
  contract_status_code text, handoff_gate_name text,
  handoff_gate_state text, active_manifest_kind text,
  blocker_summary jsonb, state_version bigint
);
create table public.issuers(issuer_id uuid primary key,display_name text,internal_secret text);
create table public.securities(
  security_id uuid primary key,issuer_id uuid,ticker text,isin text,internal_secret text
);
alter table public.orotitan_runs enable row level security;
alter table public.orotitan_run_stages enable row level security;
alter table public.issuers enable row level security;
alter table public.securities enable row level security;

insert into public.orotitan_runs values
 ('11111111-1111-4111-8111-111111111111',
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  'cccccccc-cccc-4ccc-8ccc-cccccccccccc',
  'ACTIVE','RESEARCH',1,'2026-10-21','3.1','3.1','FULL','sensitive-fixture'),
 ('22222222-2222-4222-8222-222222222222',
  'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',
  'dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  'ACTIVE','RESEARCH',1,'2026-10-21','3.1','3.1','FULL','should-not-leak');
insert into public.orotitan_run_stages values
 ('11111111-1111-4111-8111-111111111111','RESEARCH','IN_PROGRESS',
  'PENDING','RESEARCH_GATE','NO','CHECKPOINT','{}'::jsonb,1),
 ('22222222-2222-4222-8222-222222222222','RESEARCH','IN_PROGRESS',
  'PENDING','RESEARCH_GATE','NO','CHECKPOINT','{}'::jsonb,1);
insert into public.issuers values
 ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa','Besi fixture','secret-A'),
 ('bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb','Other fixture','secret-B');
insert into public.securities values
 ('cccccccc-cccc-4ccc-8ccc-cccccccccccc',
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa','BESI','NL1234567890','secret-C'),
 ('dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb','OTHER','NL0000000000','secret-D');
SQL

# All substitutions are synthetic, never real user/client/run IDs.
sed -e 's/__APPROVED_SUPABASE_AUTH_USER_UUID__/33333333-3333-4333-8333-333333333333/g' \
    -e 's/__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__/pilot-client-fixture/g' \
    -e 's/__BESI_RUN_UUID__/11111111-1111-4111-8111-111111111111/g' \
    "$root/docs/orotitan-equity/proposals/D28_MCP_BESI_DELEGATED_READ_RLS_GRANT_REVIEW_ONLY.sql" > "$tmp"
if grep -q '__[A-Z_]*__' "$tmp"; then
  echo 'D28 unresolved placeholders' >&2; exit 4
fi
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$tmp" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/tests/postgres/mcp-besi-rls-grant-proposal-verify.sql" >/dev/null
echo 'D28 Besi exact user/client/resource/run read-only RLS + column GRANT: PG17 PASS, production unchanged.'
