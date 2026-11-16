-- OroTitan C07: client-side, offline-only output of SHADOW Registry rows.
-- IMPORTANT: invoked ONLY by the protected workflow with hardcoded SHADOW host,
-- and a separately provisioned SELECT-only PostgreSQL login.
-- No RPC, INSERT, UPDATE, DELETE, DDL, or server-side file operations.
\set ON_ERROR_STOP on
\pset format unaligned
\pset tuples_only on
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '30s';

-- Fail closed before creating any JSONL. Baseline observed independently in SHADOW.
-- Legitimate CAS/count changes require a new reviewed baseline; do not silently rebase.
-- A data-dependent zero denominator raises a PostgreSQL error under ON_ERROR_STOP.
SELECT 1 / CASE WHEN
    (SELECT count(*) FROM public.orotitan_runs
     WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
       AND run_status = 'ACTIVE' AND state_version = 7) = 1
AND (SELECT count(*) FROM public.orotitan_run_stages
     WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 2
AND (SELECT count(*) FROM public.orotitan_run_stages
     WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
       AND stage_code = 'DEEP_DIVE' AND lifecycle_status = 'IN_PROGRESS'
       AND state_version = 2 AND active_manifest_kind = 'CHECKPOINT'
       AND active_manifest_version = 1) = 1
AND (SELECT count(*) FROM public.orotitan_artifacts
     WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 28
AND (SELECT count(*) FROM public.orotitan_artifact_edges
     WHERE child_run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
        OR parent_run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 8
AND (SELECT count(*) FROM public.orotitan_run_events
     WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 35
AND (SELECT count(*) FROM public.orotitan_artifacts
     WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
       AND artifact_type = 'FUNDAMENTALS_LOCK' AND version = 2) = 0
THEN 1 ELSE 0 END AS c07_registry_baseline_guard;

\o orotitan_runs.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT * FROM public.orotitan_runs
  WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'
) t;

\o orotitan_run_stages.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT * FROM public.orotitan_run_stages
  WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'
) t;

\o orotitan_artifacts.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT * FROM public.orotitan_artifacts
  WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'
) t;

\o orotitan_artifact_edges.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT * FROM public.orotitan_artifact_edges
  WHERE child_run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'
     OR parent_run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'
) t;

\o orotitan_run_events.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT * FROM public.orotitan_run_events
  WHERE run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'
) t;

\o
ROLLBACK;
