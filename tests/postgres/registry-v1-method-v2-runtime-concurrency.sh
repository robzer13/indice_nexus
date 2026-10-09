#!/usr/bin/env bash
set -euo pipefail
database="${1:?disposable database required}"
case "${PGHOST:-127.0.0.1}" in 127.0.0.1|localhost|::1) ;; *) exit 2 ;; esac
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
# Existing canonical fixture identities only. No new dossier/security/snapshot is committed.
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >/dev/null
update public.orotitan_method_v2_runtime_control
set admission_mode='ACTIVE_FOR_NEW_RUNS',runtime_commit_sha=repeat('a',40);
SQL
cat > "$scratch/request.sql" <<'SQL'
begin;
select public.create_orotitan_method_v2_runtime_run(:'request_key',d.issuer_id,s.security_id,d.dossier_id,
  d.current_snapshot_id,'2026-10-07',repeat('c',64),'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0')
from public.research_dossiers d join public.legacy_company_identity_map m using(dossier_id,issuer_id)
join public.securities s on s.security_id=m.security_id and s.issuer_id=d.issuer_id
where m.legacy_company_id='00000000-0000-4000-8000-000000000002';
-- Hold the actual RPC's transaction locks while the competitor enters.
select pg_sleep(1);
commit;
SQL
# Distinct keys race on one dossier: exactly one admission, deterministic rejection.
psql -X -v ON_ERROR_STOP=1 -v request_key=v14:race:1 -d "$database" -f "$scratch/request.sql" >"$scratch/one.log" 2>&1 &
one=$!
psql -X -v ON_ERROR_STOP=1 -v request_key=v14:race:2 -d "$database" -f "$scratch/request.sql" >"$scratch/two.log" 2>&1 &
two=$!
first=0; second=0
wait "$one" || first=$?
wait "$two" || second=$?
if ! { [[ "$first" = 0 && "$second" != 0 ]] || [[ "$first" != 0 && "$second" = 0 ]]; }; then
  cat "$scratch/one.log" "$scratch/two.log"; exit 1
fi
grep -q METHOD_V2_FRESH_RUN_ALREADY_EXISTS "$scratch/one.log" "$scratch/two.log"
psql -X -v ON_ERROR_STOP=1 -d "$database" -c "update orotitan_runs set run_status='CANCELLED',cancelled_at=now() where creation_idempotency_key in ('v14:race:1','v14:race:2')" >/dev/null
# Same exact key races: both succeed, second must return exact first run.
psql -X -v ON_ERROR_STOP=1 -v request_key=v14:race:identical -d "$database" -f "$scratch/request.sql" >"$scratch/three.log" 2>&1 &
one=$!
psql -X -v ON_ERROR_STOP=1 -v request_key=v14:race:identical -d "$database" -f "$scratch/request.sql" >"$scratch/four.log" 2>&1 &
two=$!
wait "$one"; wait "$two"
grep -q '"idempotent_replay": true' "$scratch/three.log" "$scratch/four.log"
test "$(psql -XAt -d "$database" -c "select count(*) from orotitan_runs where creation_idempotency_key='v14:race:identical'")" = 1
# A dossier lock must actually delay admission in a separate PostgreSQL session.
psql -X -v ON_ERROR_STOP=1 -d "$database" -c "update orotitan_runs set run_status='CANCELLED',cancelled_at=now() where creation_idempotency_key='v14:race:identical'" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >"$scratch/lock.log" 2>&1 &
begin;
select dossier_id from public.legacy_company_identity_map
where legacy_company_id='00000000-0000-4000-8000-000000000002';
select 1 from public.research_dossiers where dossier_id=(select dossier_id from public.legacy_company_identity_map
where legacy_company_id='00000000-0000-4000-8000-000000000002') for update;
select pg_sleep(2);
commit;
SQL
lock_pid=$!
sleep 0.2
if psql -X -v ON_ERROR_STOP=1 -v request_key=v14:lock:timeout -d "$database" \
  -c "set lock_timeout='300ms'" -f "$scratch/request.sql" >"$scratch/timeout.log" 2>&1; then
  cat "$scratch/timeout.log"; exit 1
fi
wait "$lock_pid"
grep -q 'lock timeout' "$scratch/timeout.log"
psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >/dev/null
update public.orotitan_method_v2_runtime_control
set admission_mode='INSTALLED_INACTIVE',admission_scope='INITIAL_ONLY',publication_mode='DISABLED',runtime_commit_sha=null;
SQL
echo 'B17 B25 B26 actual PostgreSQL concurrent dossier admission / exact replay / dossier row locking: PASS'
