#!/usr/bin/env bash
# D29b: TRUE Registry RPC vs private receipt, in an isolated PG17 fixture only.
set -euo pipefail
db="${1:?disposable d29b fixture DB required}"
if [[ "$db" != orotitan_d29b_real_registry_* ]] ||
  [[ "${PGHOST:-}" != "127.0.0.1" && "${PGHOST:-}" != localhost && "${PGHOST:-}" != "::1" ]] ||
  [[ "${PGPORT:-}" != "5432" || "${PGUSER:-}" != "postgres" ]]; then
  echo "D29b real-RPC test refuses non-fixture DB/host/user/port" >&2; exit 2
fi
run="48b9426b-4d0d-4858-9d87-efcd13ac415d"
issuer="$(psql -XAt -d "$db" -c "select issuer_id::text from public.orotitan_runs where run_id='$run'")"
pins="$(psql -XAt -d "$db" -c "select contract_set_sha256 from public.orotitan_runs where run_id='$run'")"
[[ "$issuer" =~ ^[0-9a-f-]{36}$ && "$pins" =~ ^[0-9a-f]{64}$ ]] ||
  { echo "D29b: missing full-Registry fixture issuer/pins" >&2; exit 3; }
psql -X -v ON_ERROR_STOP=1 -d "$db" -v issuer="$issuer" -v pins="$pins" > /dev/null <<'SQL'
INSERT INTO orotitan_research_intake_private.submission_grants(
  grant_id,run_id,issuer_id,subject_id,client_id,mcp_resource,
  contract_pins_sha256,expected_run_version,expected_stage_version,
  data_cutoff,issued_at,expires_at)
VALUES('real-registry-grant','48b9426b-4d0d-4858-9d87-efcd13ac415d',
  :'issuer','approved-oauth-sub','approved-client',
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
  :'pins',6,4,'2026-10-12',now()-interval '10 seconds',now()+interval '3 minutes');
INSERT INTO orotitan_research_intake_private.verifier_grants(
  verifier_grant_id,run_id,issuer_id,reviewer_id,
  expected_run_version,expected_stage_version,issued_at,expires_at)
VALUES('real-registry-review-grant','48b9426b-4d0d-4858-9d87-efcd13ac415d',
  :'issuer','independent-d2',6,4,
  now()-interval '10 seconds',now()+interval '3 minutes');
SQL
mutation_log="$(mktemp)"
receipt_log="$(mktemp)"
mutation_pid=""
cleanup() {
  [[ -z "$mutation_pid" ]] || wait "$mutation_pid" 2>/dev/null || true
  rm -f "$mutation_log" "$receipt_log"
}
trap cleanup EXIT

# The *actual* public.pause_orotitan_stage RPC acquires row locks on BOTH
# Registry tables, updates both CAS counters, and appends a Registry event.
# Hold transaction open briefly to force overlap with the SERIALIZABLE receipt.
psql -X -v ON_ERROR_STOP=1 -d "$db" > "$mutation_log" 2>&1 <<'SQL' &
BEGIN;
SELECT public.pause_orotitan_stage(
  '48b9426b-4d0d-4858-9d87-efcd13ac415d','RESEARCH',6,4,
  'd29b:isolated-actual-rpc-pause',repeat('9',64),
  '{"reason":"isolated real RPC race test"}'::jsonb);
SELECT pg_sleep(3);
COMMIT;
SQL
mutation_pid="$!"
sleep 0.5

if ! psql -X -v ON_ERROR_STOP=1 -d "$db" -v issuer="$issuer" -v pins="$pins" > "$receipt_log" 2>&1 <<'SQL'
BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL ROLE orotitan_research_intake_writer;
SELECT * FROM orotitan_research_intake_private.commit_receipt_v1(
  '48b9426b-4d0d-4858-9d87-efcd13ac415d',
  :'issuer','real-registry-grant','approved-oauth-sub','approved-client',
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
  6,4,'2026-10-12',:'pins',repeat('e',64),
  'race.json','application/json',
  'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
    repeat('e',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac09.json',
  15,'2026-07-23');
COMMIT;
SQL
then
  if ! grep -Eq 'could not serialize|D29B_PRIVATE_GRANT_CAS_DENIED|D29B_PRIVATE_CHANGED_AFTER_LOCK' "$receipt_log"; then
    cat "$receipt_log" >&2
    echo 'D29b: unexpected private receipt/Registry RPC interaction' >&2; exit 3
  fi
fi
if ! wait "$mutation_pid"; then
  cat "$mutation_log" >&2
  echo 'D29b: real Registry pause RPC rejected' >&2; exit 3
fi
mutation_pid=""

psql -X -v ON_ERROR_STOP=1 -d "$db" -v issuer="$issuer" > /dev/null <<'SQL'
DO $after$
DECLARE n bigint; stopped boolean:=false; r public.orotitan_runs%ROWTYPE;
BEGIN
  SELECT * INTO STRICT r FROM public.orotitan_runs
    WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d';
  IF r.run_status<>'PAUSED' OR r.state_version<>7
    OR r.current_stage<>'RESEARCH' THEN
    RAISE EXCEPTION 'actual Registry pause did not update run CAS correctly';
  END IF;
  IF NOT EXISTS(SELECT 1 FROM public.orotitan_run_stages
    WHERE run_id=r.run_id AND stage_code='RESEARCH'
      AND lifecycle_status='PAUSED' AND state_version=5
      AND active_manifest_kind='CHECKPOINT' AND handoff_gate_state='NO') THEN
    RAISE EXCEPTION 'actual Registry pause did not update stage CAS';
  END IF;
  SELECT count(*) INTO n FROM orotitan_research_intake_private.receipts;
  IF n>1 THEN RAISE EXCEPTION 'unexpected concurrent duplicated receipt'; END IF;
  IF n<>(SELECT count(*) FROM orotitan_research_intake_private.receipt_audit) THEN
    RAISE EXCEPTION 'receipt and audit diverged';
  END IF;
  BEGIN
    PERFORM public.start_orotitan_stage(
      r.run_id,'DEEP_DIVE',r.state_version,
      r.contract_pins->'deep_dive_stage'->>'name',
      r.contract_pins->'deep_dive_stage'->>'version',
      r.contract_pins->'deep_dive_stage'->>'content_sha256',
      'd29b:forbidden-handoff',repeat('b',64),'PILOTAGE'
    );
  EXCEPTION WHEN check_violation THEN stopped:=true;
  END;
  IF NOT stopped THEN RAISE EXCEPTION 'quarantine receipt advanced real Registry'; END IF;
END;
$after$;
SET ROLE orotitan_research_intake_writer;
DO $stale$
DECLARE allowed boolean;
BEGIN
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    (SELECT issuer_id::text FROM public.orotitan_runs
      WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'),
    'real-registry-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',
    (SELECT contract_set_sha256 FROM public.orotitan_runs
      WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d')
  ) INTO allowed;
  IF allowed THEN RAISE EXCEPTION 'outdated real Registry RPC CAS allowed upload'; END IF;
END;
$stale$;
RESET ROLE;
SET ROLE orotitan_research_intake_reader;
DO $reader$
DECLARE n bigint;
BEGIN
  SELECT count(*) INTO n FROM orotitan_research_intake_private.verifier_metadata_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    (SELECT issuer_id::text FROM public.orotitan_runs
     WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'),
    'independent-d2','real-registry-review-grant',6,4,repeat('e',64));
  IF n<>0 THEN RAISE EXCEPTION 'D2 admitted stale metadata after actual Registry RPC'; END IF;
END;
$reader$;
RESET ROLE;
SQL
echo 'D29b real Registry pause RPC + private receipt race PASS (fixture-only, gate remains NO).'
