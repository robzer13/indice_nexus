-- OroTitan C07: client-side, offline-only output of SHADOW Registry rows.
-- IMPORTANT: invoked ONLY by the protected workflow with hardcoded SHADOW host,
-- and a separately provisioned SELECT-only PostgreSQL login.
-- No RPC, INSERT, UPDATE, DELETE, DDL, or server-side file operations.
\set ON_ERROR_STOP on
\pset format unaligned
\pset tuples_only on
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '30s';

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
