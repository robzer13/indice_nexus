-- C08 read-only smoke probe — run ONLY on SHADOW after separately approved grants.
-- No credentials, private rows, identifiers or payloads are printed.
-- Execute with a SHADOW admin connection; SET LOCAL ROLE impersonates the
-- C07 reader for an RLS/privilege test (not proof of actual pooler login).
\set ON_ERROR_STOP on
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL ROLE orotitan_c07_reader;

SELECT 1 / CASE WHEN
  (SELECT count(*) FROM public.orotitan_runs r
   WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
  AND (SELECT count(*) FROM public.issuers) = 1
  AND (SELECT count(*) FROM public.securities) = 1
  AND (SELECT count(*) FROM public.research_dossiers) = 1
  AND (SELECT count(*) FROM public.orotitan_p0e_run_admission_receipts_v1) = 1
  AND (SELECT count(*) FROM public.issuers i
       JOIN public.orotitan_runs r ON r.issuer_id=i.issuer_id
       WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
  AND (SELECT count(*) FROM public.securities s
       JOIN public.orotitan_runs r
         ON r.security_id=s.security_id AND r.issuer_id=s.issuer_id
       WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
  AND (SELECT count(*) FROM public.research_dossiers d
       JOIN public.orotitan_runs r
         ON r.dossier_id=d.dossier_id AND r.issuer_id=d.issuer_id
       WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
  AND (SELECT count(*) FROM public.orotitan_p0e_run_admission_receipts_v1 a
       JOIN public.orotitan_runs r ON r.run_id=a.run_id
         AND r.issuer_id=a.issuer_id AND r.security_id=a.security_id
         AND r.dossier_id=a.dossier_id
       WHERE r.run_id='db00ce77-04eb-492b-a76b-164ecbe9ccfe'::uuid) = 1
  AND NOT has_table_privilege(current_user,'public.issuers','INSERT')
  AND NOT has_table_privilege(current_user,'public.securities','UPDATE')
  AND NOT has_table_privilege(current_user,'public.research_dossiers','DELETE')
  AND NOT has_table_privilege(current_user,
     'public.orotitan_p0e_run_admission_receipts_v1','TRUNCATE')
THEN 1 ELSE 0 END AS c08_schneider_read_scope_guard;
ROLLBACK;
