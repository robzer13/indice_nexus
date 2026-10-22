#!/usr/bin/env bash
# D29b audited writer REVIEW-ONLY test, synthetic identities, LOOPBACK PG17 only.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
  echo 'D29b refusing non-loopback PGHOST' >&2; exit 2;;
esac
if [[ "$PGUSER" != "postgres" ]]; then
  echo 'D29b refusing non-fixture PGUSER' >&2; exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d29b_writer_$"
tmp="$(mktemp)"
tmpWriter="$(mktemp)"
created=0
cleanup() {
  rm -f "$tmp" "$tmpWriter"
  if [[ "$created" -eq 1 ]]; then dropdb --if-exists --force "$db" >/dev/null 2>&1 || true; fi
  psql -X -d postgres -c "drop role if exists orotitan_d29b_writer" >/dev/null 2>&1 || true
}
trap cleanup EXIT
createdb "$db"
created=1
version="$(psql -XAt -d "$db" -c 'show server_version_num')"
if (( version < 170000 || version >= 180000 )); then
  echo "D29b requires PostgreSQL 17 (got $version)" >&2; exit 3
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
create function auth.jwt() returns jsonb language sql stable
as $$select coalesce(nullif(current_setting('request.jwt.claims',true),''),'{}')::jsonb$$;
create function auth.uid() returns uuid language sql stable
as $$select nullif(auth.jwt()->>'sub','')::uuid$$;
grant usage on schema auth,public to authenticated,anon;
grant execute on function auth.jwt() to authenticated;
grant execute on function auth.uid() to authenticated;
create table public.orotitan_runs(
 run_id uuid primary key,run_status text not null,
 current_stage text not null,state_version bigint not null,data_cutoff date not null);
create table public.orotitan_run_stages(
 run_id uuid not null,stage_code text not null,state_version bigint not null,
 lifecycle_status text not null,contract_status_code text not null,
 handoff_gate_state text not null,active_manifest_kind text not null,
 primary key(run_id,stage_code));
alter table public.orotitan_runs enable row level security;
alter table public.orotitan_run_stages enable row level security;
insert into public.orotitan_runs values
 ('11111111-1111-4111-8111-111111111111','ACTIVE','RESEARCH',6,current_date),
 ('22222222-2222-4222-8222-222222222222','ACTIVE','RESEARCH',6,current_date);
insert into public.orotitan_run_stages values
 ('11111111-1111-4111-8111-111111111111','RESEARCH',4,
 'IN_PROGRESS','OBSERVED','NOT_EVALUATED','CHECKPOINT'),
 ('22222222-2222-4222-8222-222222222222','RESEARCH',4,
 'IN_PROGRESS','OBSERVED','NOT_EVALUATED','CHECKPOINT');
SQL

sed -e 's/__APPROVED_SUPABASE_AUTH_USER_UUID__/33333333-3333-4333-8333-333333333333/g' \
    -e 's/__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__/pilot-client-fixture/g' \
    -e 's/__BESI_RUN_UUID__/11111111-1111-4111-8111-111111111111/g' \
    "$root/docs/orotitan-equity/proposals/D29B_SIGNED_RECEIPT_RLS_SQL_REVIEW_ONLY.sql" > "$tmp"
if grep -q '__[A-Z_]*__' "$tmp"; then echo 'D29b placeholders remain' >&2; exit 4; fi
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$tmp" >/dev/null
sed -e 's/__APPROVED_SUPABASE_AUTH_USER_UUID__/33333333-3333-4333-8333-333333333333/g' \
    -e 's/__REGISTERED_CHATGPT_OAUTH_CLIENT_ID__/pilot-client-fixture/g' \
    -e 's/__BESI_RUN_UUID__/11111111-1111-4111-8111-111111111111/g' \
    "$root/docs/orotitan-equity/proposals/D29B_PRIVILEGED_RECEIPT_WRITER_SQL_REVIEW_ONLY.sql" > "$tmpWriter"
if grep -q '__[A-Z_]*__' "$tmpWriter"; then echo 'D29b writer placeholders remain' >&2; exit 4; fi
psql -X -v ON_ERROR_STOP=1 -d "$db" -f "$tmpWriter" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" \
  -f "$root/tests/postgres/mcp-d29b-audited-writer-verify.sql" >/dev/null
echo 'D29b private NOLOGIN writer: SERIALIZABLE CAS, signed receipt, append-only audit, revoke and OAuth concealment PASS (production unchanged).'
