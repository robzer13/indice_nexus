\set ON_ERROR_STOP on
-- D35 / review-only. Audit *disposable* PostgreSQL 17 ACLs after D34 E2E.
DO $audit$
DECLARE p record;priv_count int;
BEGIN
 IF current_setting('server_version_num')::int NOT BETWEEN 170000 AND 179999
 OR to_regclass('orotitan_mobile_private.encrypted_capability_vault') IS NULL
 OR to_regclass('public.orotitan_runs') IS NOT NULL THEN
   RAISE EXCEPTION 'D35_WRONG_DATABASE_OR_REGISTRY_PRESENT';
 END IF;
 IF (SELECT count(*) FROM pg_roles WHERE rolname IN(
  'orotitan_d32_approval_issuer','orotitan_d32_command_submitter',
  'orotitan_d32_private_executor','orotitan_d34_preview_issuer',
  'orotitan_d34_preview_reader','orotitan_d34_vault_issuer',
  'orotitan_d34_vault_reader'))<>7 THEN
  RAISE EXCEPTION 'D35_PRIVATE_ROLE_MATRIX_INCOMPLETE';
 END IF;
 FOR p IN SELECT rolname,rolcanlogin,rolinherit,rolbypassrls,rolsuper
  FROM pg_roles WHERE rolname IN(
  'orotitan_d32_approval_issuer','orotitan_d32_command_submitter',
  'orotitan_d32_private_executor','orotitan_d34_preview_issuer',
  'orotitan_d34_preview_reader','orotitan_d34_vault_issuer',
  'orotitan_d34_vault_reader') LOOP
  IF p.rolcanlogin OR p.rolinherit OR p.rolbypassrls OR p.rolsuper THEN
    RAISE EXCEPTION 'D35_PRIVATE_ROLE_ELEVATION:%',p.rolname;
  END IF;
 END LOOP;
 IF EXISTS(SELECT 1 FROM pg_auth_members am
     JOIN pg_roles r ON r.oid=am.roleid
     WHERE r.rolname IN(
      'orotitan_d32_private_executor','orotitan_d34_preview_issuer',
      'orotitan_d34_preview_reader','orotitan_d34_vault_issuer',
      'orotitan_d34_vault_reader')) THEN
     RAISE EXCEPTION 'D35_PRIVATE_ROLE_MEMBERSHIP_DETECTED';
 END IF;
 IF EXISTS(SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_roles r ON r.oid=p.proowner
  WHERE n.nspname='orotitan_mobile_private' AND p.proname IN(
    'create_preview_v3','read_preview_v3','approve_and_store_capability_v3',
    'read_encrypted_capability_v3')
  AND (NOT p.prosecdef OR r.rolname<>'orotitan_d32_private_executor'
    OR p.proconfig IS NULL)) OR
  (SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
   WHERE n.nspname='orotitan_mobile_private' AND p.proname IN(
    'create_preview_v3','read_preview_v3','approve_and_store_capability_v3',
    'read_encrypted_capability_v3'))<>4 THEN
    RAISE EXCEPTION 'D35_WRONG_DEFINER_OR_FUNCTION_OWNER';
 END IF;
 IF EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
  WHERE n.nspname='orotitan_mobile_private'
  AND c.relname IN('previews','command_requests','encrypted_capability_vault')
  AND NOT c.relrowsecurity) THEN
    RAISE EXCEPTION 'D35_RLS_MISSING';
 END IF;
 IF has_schema_privilege('anon','orotitan_mobile_private','USAGE')
 OR has_schema_privilege('authenticated','orotitan_mobile_private','USAGE')
 OR has_table_privilege('anon','orotitan_mobile_private.previews','SELECT')
 OR has_table_privilege('authenticated','orotitan_mobile_private.previews','SELECT')
 OR has_table_privilege('authenticated',
   'orotitan_mobile_private.encrypted_capability_vault','SELECT')
 OR has_table_privilege('orotitan_d34_preview_reader',
   'orotitan_mobile_private.previews','SELECT')
 OR has_table_privilege('orotitan_d34_vault_reader',
   'orotitan_mobile_private.encrypted_capability_vault','SELECT')
 OR has_table_privilege('orotitan_d34_preview_issuer',
   'orotitan_mobile_private.previews','INSERT')
 OR has_table_privilege('orotitan_d34_vault_issuer',
   'orotitan_mobile_private.encrypted_capability_vault','INSERT')
 OR has_table_privilege('orotitan_d32_command_submitter',
   'orotitan_mobile_private.previews','SELECT')
 OR has_table_privilege('orotitan_d32_approval_issuer',
   'orotitan_mobile_private.previews','SELECT') THEN
  RAISE EXCEPTION 'D35_PRIVATE_TABLE_GRANT_EXPOSURE';
 END IF;
 IF has_function_privilege('anon',
   'orotitan_mobile_private.create_preview_v3(uuid,text,text,text,text,jsonb,text,timestamptz)','EXECUTE')
 OR has_function_privilege('authenticated',
   'orotitan_mobile_private.read_preview_v3(uuid,text,text,text)','EXECUTE')
 OR has_function_privilege('authenticated',
   'orotitan_mobile_private.read_encrypted_capability_v3(uuid,text,text,text)','EXECUTE')
 OR has_function_privilege('orotitan_d34_preview_issuer',
   'orotitan_mobile_private.read_preview_v3(uuid,text,text,text)','EXECUTE')
 OR has_function_privilege('orotitan_d34_preview_reader',
   'orotitan_mobile_private.create_preview_v3(uuid,text,text,text,text,jsonb,text,timestamptz)','EXECUTE')
 OR has_function_privilege('orotitan_d34_vault_issuer',
   'orotitan_mobile_private.read_encrypted_capability_v3(uuid,text,text,text)','EXECUTE')
 OR has_function_privilege('orotitan_d34_vault_reader',
   'orotitan_mobile_private.approve_and_store_capability_v3(uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)','EXECUTE')
 OR has_function_privilege('orotitan_d34_preview_reader',
   'orotitan_mobile_private.confirm_approved_request_v2(uuid,text)','EXECUTE')
 OR has_function_privilege('orotitan_d34_preview_issuer',
   'orotitan_mobile_private.confirm_request_v1(uuid,text,text,text,text)','EXECUTE')
 OR has_function_privilege('orotitan_d32_command_submitter',
   'orotitan_mobile_private.read_encrypted_capability_v3(uuid,text,text,text)','EXECUTE') THEN
  RAISE EXCEPTION 'D35_CROSS_ROLE_EXECUTE_EXPOSURE';
 END IF;
 IF NOT has_function_privilege('orotitan_d34_preview_issuer',
   'orotitan_mobile_private.create_preview_v3(uuid,text,text,text,text,jsonb,text,timestamptz)','EXECUTE')
 OR NOT has_function_privilege('orotitan_d34_preview_reader',
   'orotitan_mobile_private.read_preview_v3(uuid,text,text,text)','EXECUTE')
 OR NOT has_function_privilege('orotitan_d34_vault_issuer',
   'orotitan_mobile_private.approve_and_store_capability_v3(uuid,text,text,text,text,text,text,text,bytea,bytea,bytea)','EXECUTE')
 OR NOT has_function_privilege('orotitan_d34_vault_reader',
   'orotitan_mobile_private.read_encrypted_capability_v3(uuid,text,text,text)','EXECUTE')
 THEN RAISE EXCEPTION 'D35_EXPECTED_EXECUTE_MISSING'; END IF;
END;
$audit$;

SET ROLE orotitan_d34_preview_reader;
DO $deny$
DECLARE denied boolean:=false;
BEGIN
 BEGIN
  EXECUTE 'SELECT count(*) FROM orotitan_mobile_private.previews';
 EXCEPTION WHEN insufficient_privilege THEN denied:=true;
 END;
 IF NOT denied THEN RAISE EXCEPTION 'D35_PREVIEW_READER_DIRECT_TABLE_ACCESS'; END IF;
END;
$deny$;
RESET ROLE;

SET ROLE orotitan_d34_vault_reader;
DO $deny$
DECLARE denied boolean:=false;
BEGIN
 BEGIN
  EXECUTE 'SELECT count(*) FROM orotitan_mobile_private.encrypted_capability_vault';
 EXCEPTION WHEN insufficient_privilege THEN denied:=true;
 END;
 IF NOT denied THEN RAISE EXCEPTION 'D35_VAULT_READER_DIRECT_TABLE_ACCESS'; END IF;
END;
$deny$;
RESET ROLE;
\echo 'D35 seven-role least-privilege and cross-actor ACL security audit PASS (DISPOSABLE PG17).'
