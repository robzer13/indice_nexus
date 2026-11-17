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

-- C08: four FK-parent and receipt rows in THIS SAME private READ ONLY snapshot.
-- Policies were granted in SHADOW separately; no extra connection or mutation.
SELECT 1 / CASE WHEN
    (SELECT count(*) FROM public.issuers i JOIN public.orotitan_runs r
       ON i.issuer_id = r.issuer_id
       WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
AND (SELECT count(*) FROM public.securities s JOIN public.orotitan_runs r
       ON s.security_id = r.security_id AND s.issuer_id = r.issuer_id
       WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
AND (SELECT count(*) FROM public.research_dossiers d JOIN public.orotitan_runs r
       ON d.dossier_id = r.dossier_id AND d.issuer_id = r.issuer_id
       WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
         AND d.current_snapshot_id IS NULL) = 1
AND (SELECT count(*) FROM public.orotitan_p0e_run_admission_receipts_v1 a
       JOIN public.orotitan_runs r ON a.run_id = r.run_id
         AND a.issuer_id = r.issuer_id AND a.security_id = r.security_id
         AND a.dossier_id = r.dossier_id
       WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
THEN 1 ELSE 0 END AS c08_fk_closure_guard;

\o issuers.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT i.* FROM public.issuers i JOIN public.orotitan_runs r
    ON i.issuer_id = r.issuer_id
  WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
) t;

\o securities.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT s.* FROM public.securities s JOIN public.orotitan_runs r
    ON s.security_id = r.security_id AND s.issuer_id = r.issuer_id
  WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
) t;

\o research_dossiers.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT d.* FROM public.research_dossiers d JOIN public.orotitan_runs r
    ON d.dossier_id = r.dossier_id AND d.issuer_id = r.issuer_id
  WHERE r.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
) t;

\o orotitan_p0e_run_admission_receipts_v1.jsonl
SELECT row_to_json(t)::text FROM (
  SELECT a.* FROM public.orotitan_p0e_run_admission_receipts_v1 a
    WHERE a.run_id = 'db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
) t;

\o
ROLLBACK;
