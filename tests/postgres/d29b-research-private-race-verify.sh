#!/usr/bin/env bash
# D29b concurrency simulation in disposable PostgreSQL 17 fixture DB ONLY.
# Called by run-d29b-research-private-sql-tests.sh, NEVER against Supabase.
set -euo pipefail
db="${1:?disposable fixture DB required}"
if [[ "$db" != orotitan_d29b_research_* ]]; then
  echo "D29b race: refusing non-fixture DB name" >&2; exit 2
fi
if [[ "${PGHOST:-}" != 127.0.0.1 && "${PGHOST:-}" != localhost && "${PGHOST:-}" != ::1 ]] ||
   [[ "${PGUSER:-}" != postgres || "${PGPORT:-}" != 5432 ]]; then
  echo "D29b race: refusing non-local or non-fixture database" >&2; exit 2
fi
update_log="$(mktemp)"
commit_log="$(mktemp)"
mutation_pid=""
cleanup() {
  if [[ -n "$mutation_pid" ]]; then wait "$mutation_pid" 2>/dev/null || true; fi
  rm -f "$update_log" "$commit_log"
}
trap cleanup EXIT

# Stage writer holds an uncommitted Registry update while an intake request
# attempts SERIALIZABLE commit. This deliberately does NOT assume a shared
# advisory lock; admissibility must remain false under either ordering.
psql -X -v ON_ERROR_STOP=1 -d "$db" >"$update_log" 2>&1 <<'SQL' &
BEGIN;
UPDATE public.orotitan_run_stages SET state_version=5
WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'
  AND stage_code='RESEARCH';
SELECT pg_sleep(3);
COMMIT;
SQL
mutation_pid="$!"
sleep 0.5

# A stale read may serialize just before the other writer's state update;
# that is acceptable ONLY for an unverified quarantine receipt, never D2.
if ! psql -X -v ON_ERROR_STOP=1 -d "$db" >"$commit_log" 2>&1 <<'SQL'
BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL ROLE orotitan_research_intake_writer;
SELECT * FROM orotitan_research_intake_private.commit_receipt_v1(
  '48b9426b-4d0d-4858-9d87-efcd13ac415d',
  '33333333-3333-4333-8333-333333333333',
  'approved-intake-grant','approved-oauth-sub','approved-client',
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
  6,4,'2026-10-12',repeat('a',64),repeat('e',64),
  'race.json','application/json',
  'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
  repeat('e',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac09.json',
  15,'2026-07-23'
);
COMMIT;
SQL
then
  if ! grep -Eq 'could not serialize|D29B_PRIVATE_GRANT_CAS_DENIED|D29B_PRIVATE_CHANGED_AFTER_LOCK' "$commit_log"; then
    cat "$commit_log" >&2
    echo "D29b race: unexpected commit failure" >&2; exit 1
  fi
fi
if ! wait "$mutation_pid"; then
  cat "$update_log" >&2
  echo "D29b race: Registry fixture writer failed" >&2; exit 1
fi
mutation_pid=""

psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null <<'SQL'
DO $verify$
DECLARE n bigint;
BEGIN
  IF NOT EXISTS(SELECT 1 FROM public.orotitan_run_stages
    WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'
    AND stage_code='RESEARCH' AND state_version=5) THEN
    RAISE EXCEPTION 'concurrent Registry version did not change';
  END IF;
  SELECT count(*) INTO n FROM orotitan_research_intake_private.receipts;
  IF n NOT BETWEEN 1 AND 2 THEN
    RAISE EXCEPTION 'unexpected concurrent receipt count %',n;
  END IF;
  IF n<>(SELECT count(*) FROM orotitan_research_intake_private.receipt_audit) THEN
    RAISE EXCEPTION 'receipt/audit diverged under concurrency';
  END IF;
  IF EXISTS(SELECT 1 FROM orotitan_research_intake_private.receipts
    WHERE receipt_status <> 'RECEIVED_NOT_VERIFIED') THEN
    RAISE EXCEPTION 'concurrency generated an admitted receipt';
  END IF;
END;
$verify$;
SET ROLE orotitan_research_intake_writer;
DO $stale$
DECLARE allowed boolean;
BEGIN
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64)
  ) INTO allowed;
  IF allowed THEN RAISE EXCEPTION 'stale post-race preflight authorized'; END IF;
END;
$stale$;
RESET ROLE;
SET ROLE orotitan_research_intake_reader;
DO $verify$
DECLARE n bigint;
BEGIN
  SELECT count(*) INTO n FROM orotitan_research_intake_private.verifier_metadata_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'independent-d2','approved-reviewer-grant',6,4,repeat('e',64)
  );
  IF n<>0 THEN RAISE EXCEPTION 'post-race D2 exposed stale receipt'; END IF;
END;
$verify$;
RESET ROLE;
SQL
echo "D29b race simulation: stale D2 denied; receipt/audit atomic; no admission."
