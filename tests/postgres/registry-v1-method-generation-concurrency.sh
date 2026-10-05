#!/usr/bin/env bash
set -euo pipefail
database="${1:?disposable database required}"
case "${PGHOST:-127.0.0.1}" in 127.0.0.1|localhost|::1) ;; *) exit 2 ;; esac
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

psql -X -v ON_ERROR_STOP=1 -d "$database" <<'SQL' >/dev/null
insert into public.orotitan_runs(run_id,creation_idempotency_key,issuer_id,security_id,dossier_id,
  entry_path,canonical_mode,run_type,run_status,current_stage,data_cutoff,
  process_version,pilotage_contract_version,contract_pins,contract_set_sha256,state_version)
select gen_random_uuid(),'method-generation:concurrent:'||n,issuer_id,security_id,dossier_id,
  entry_path,canonical_mode,'INITIAL','ACTIVE','DEEP_DIVE',data_cutoff,
  process_version,pilotage_contract_version,contract_pins,contract_set_sha256,7
from public.orotitan_runs cross join generate_series(1,2) n
where creation_idempotency_key='method-generation:parent';
SQL

cat > "$scratch/request.sql" <<'SQL'
begin;
-- The RPC itself must acquire the parent lock; the harness must not serialize it.
select public.create_orotitan_method_v2_successor_run(
  :'request_key',issuer_id,entry_path,canonical_mode,'INITIAL',data_cutoff,run_id,null,
  process_version,pilotage_contract_version,contract_pins,contract_set_sha256,repeat('c',64),
  state_version,'ACTIVE','DEEP_DIVE',contract_set_sha256,security_id,dossier_id,
  '1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2')
from public.orotitan_runs where creation_idempotency_key=:'parent_key';
select pg_sleep(1);
commit;
SQL

# Distinct requests race for a single parent. Exactly one must be admitted.
psql -X -v ON_ERROR_STOP=1 -v parent_key=method-generation:concurrent:1 \
  -v request_key=method-generation:race:1 -d "$database" -f "$scratch/request.sql" >"$scratch/one.log" 2>&1 &
one=$!
psql -X -v ON_ERROR_STOP=1 -v parent_key=method-generation:concurrent:1 \
  -v request_key=method-generation:race:2 -d "$database" -f "$scratch/request.sql" >"$scratch/two.log" 2>&1 &
two=$!
first=0; second=0
wait "$one" || first=$?
wait "$two" || second=$?
if ! { [[ "$first" = 0 && "$second" != 0 ]] || [[ "$first" != 0 && "$second" = 0 ]]; }; then
  cat "$scratch/one.log" "$scratch/two.log"; exit 1
fi
grep -q METHOD_V2_SUCCESSOR_ALREADY_EXISTS "$scratch/one.log" "$scratch/two.log"

# Identical requests race: both must succeed and the second returns the same child.
psql -X -v ON_ERROR_STOP=1 -v parent_key=method-generation:concurrent:2 \
  -v request_key=method-generation:identical -d "$database" -f "$scratch/request.sql" >"$scratch/three.log" 2>&1 &
one=$!
psql -X -v ON_ERROR_STOP=1 -v parent_key=method-generation:concurrent:2 \
  -v request_key=method-generation:identical -d "$database" -f "$scratch/request.sql" >"$scratch/four.log" 2>&1 &
two=$!
wait "$one"; wait "$two"
grep -q '"idempotent_replay": true' "$scratch/three.log" "$scratch/four.log"
test "$(psql -XAt -d "$database" -c "select count(*) from orotitan_runs c join orotitan_runs p on c.parent_run_id=p.run_id where p.creation_idempotency_key like 'method-generation:concurrent:%' and c.methodology_generation='METHOD_V2'")" = 2
echo 'Method-V2 concurrent fork rejection and exact replay: PASS'
