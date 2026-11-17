-- OROTITAN C08 SECURITY CHANGE PROPOSAL — REVIEW ONLY, NEVER AUTO-APPLY
-- PROJECT: Supabase SHADOW awgsurdyvsyolcgpnygh (not PRODUCTION).
-- OPERATOR GATE: verify project URL/ref, approved independent review, and
-- exact Git commit before executing manually as a SHADOW database administrator.
-- Narrow scope: four SELECT grants plus four row-scoped RLS policies.
-- No DML, DDL rights for the reader, no schema changes, no secrets.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $c08_preflight$
DECLARE
  reader_oid oid;
  target_count integer;
BEGIN
  SELECT oid INTO reader_oid FROM pg_roles
    WHERE rolname = 'orotitan_c07_reader'
      AND rolcanlogin AND NOT rolsuper AND NOT rolbypassrls
      AND NOT rolcreatedb AND NOT rolcreaterole;
  IF reader_oid IS NULL
     OR EXISTS (SELECT 1 FROM pg_auth_members WHERE member = reader_oid)
  THEN RAISE EXCEPTION 'BLOCKED_C08_ROLE_PRECONDITION'; END IF;

  IF NOT has_schema_privilege('orotitan_c07_reader','public','USAGE')
     OR NOT has_table_privilege('orotitan_c07_reader','public.orotitan_runs','SELECT')
  THEN RAISE EXCEPTION 'BLOCKED_C08_EXISTING_READER_SCOPE'; END IF;

  SELECT count(*) INTO target_count
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'public' AND c.relkind IN ('r','p')
    AND c.relname = ANY(ARRAY[
      'issuers','securities','research_dossiers',
      'orotitan_p0e_run_admission_receipts_v1'
    ])
    AND c.relrowsecurity AND pg_get_userbyid(c.relowner) = 'postgres'
    AND NOT has_table_privilege('orotitan_c07_reader',c.oid,'SELECT')
    AND NOT has_table_privilege('orotitan_c07_reader',c.oid,'INSERT')
    AND NOT has_table_privilege('orotitan_c07_reader',c.oid,'UPDATE')
    AND NOT has_table_privilege('orotitan_c07_reader',c.oid,'DELETE')
    AND NOT has_table_privilege('orotitan_c07_reader',c.oid,'TRUNCATE');
  IF target_count <> 4 THEN
    RAISE EXCEPTION 'BLOCKED_C08_TABLE_PRECONDITION';
  END IF;

  IF EXISTS (
    SELECT 1 FROM pg_policies WHERE schemaname='public'
    AND tablename = ANY(ARRAY[
      'issuers','securities','research_dossiers',
      'orotitan_p0e_run_admission_receipts_v1'
    ])
  ) THEN RAISE EXCEPTION 'BLOCKED_C08_UNEXPECTED_EXISTING_RLS_POLICY'; END IF;

  IF (SELECT count(*) FROM public.orotitan_runs r
      WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
        AND r.run_status='ACTIVE' AND r.state_version=7) <> 1
     OR (SELECT count(*) FROM public.issuers i
         JOIN public.orotitan_runs r ON r.issuer_id=i.issuer_id
         WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) <> 1
     OR (SELECT count(*) FROM public.securities s
         JOIN public.orotitan_runs r
           ON r.security_id=s.security_id AND r.issuer_id=s.issuer_id
         WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) <> 1
     OR (SELECT count(*) FROM public.research_dossiers d
         JOIN public.orotitan_runs r
           ON r.dossier_id=d.dossier_id AND r.issuer_id=d.issuer_id
         WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) <> 1
     OR (SELECT count(*) FROM public.orotitan_p0e_run_admission_receipts_v1 a
         JOIN public.orotitan_runs r
           ON a.run_id=r.run_id AND a.issuer_id=r.issuer_id
           AND a.security_id=r.security_id AND a.dossier_id=r.dossier_id
         WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) <> 1
  THEN RAISE EXCEPTION 'BLOCKED_C08_RUN_FK_CLOSURE_PRECONDITION'; END IF;
END
$c08_preflight$;

CREATE POLICY c08_schneider_reader_select
  ON public.issuers AS PERMISSIVE FOR SELECT TO orotitan_c07_reader
  USING (EXISTS (
    SELECT 1 FROM public.orotitan_runs r
    WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
      AND r.issuer_id=issuers.issuer_id
  ));

CREATE POLICY c08_schneider_reader_select
  ON public.securities AS PERMISSIVE FOR SELECT TO orotitan_c07_reader
  USING (EXISTS (
    SELECT 1 FROM public.orotitan_runs r
    WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
      AND r.security_id=securities.security_id
      AND r.issuer_id=securities.issuer_id
  ));

CREATE POLICY c08_schneider_reader_select
  ON public.research_dossiers AS PERMISSIVE FOR SELECT TO orotitan_c07_reader
  USING (EXISTS (
    SELECT 1 FROM public.orotitan_runs r
    WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
      AND r.dossier_id=research_dossiers.dossier_id
      AND r.issuer_id=research_dossiers.issuer_id
  ));

CREATE POLICY c08_schneider_reader_select
  ON public.orotitan_p0e_run_admission_receipts_v1 AS PERMISSIVE
  FOR SELECT TO orotitan_c07_reader
  USING (EXISTS (
    SELECT 1 FROM public.orotitan_runs r
    WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid
      AND r.run_id=orotitan_p0e_run_admission_receipts_v1.run_id
      AND r.issuer_id=orotitan_p0e_run_admission_receipts_v1.issuer_id
      AND r.security_id=orotitan_p0e_run_admission_receipts_v1.security_id
      AND r.dossier_id=orotitan_p0e_run_admission_receipts_v1.dossier_id
  ));

GRANT SELECT ON TABLE public.issuers TO orotitan_c07_reader;
GRANT SELECT ON TABLE public.securities TO orotitan_c07_reader;
GRANT SELECT ON TABLE public.research_dossiers TO orotitan_c07_reader;
GRANT SELECT ON TABLE public.orotitan_p0e_run_admission_receipts_v1 TO orotitan_c07_reader;
COMMIT;
