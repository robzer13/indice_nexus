\set ON_ERROR_STOP on
-- D32 test data: generated on DISPOSABLE I1 PostgreSQL 17 only.
UPDATE public.securities SET isin='GB1234567890'
WHERE issuer_id='00000000-0000-4000-8000-000000000001';

-- ISIN values in this fixture are syntactically shaped, not market evidence.
INSERT INTO orotitan_mobile_private.previews
(preview_id,kind,actor_subject,actor_client,actor_resource,payload,
 fingerprint_sha256,expires_at)
VALUES
('20000000-0000-4000-8000-000000000001','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"Lyra Test Co","ticker":"LYRA","exchange":"XAMS","isin":"NL1234567890"}',
 repeat('a',64),clock_timestamp()+interval '90 seconds'),
('20000000-0000-4000-8000-000000000002','request_research',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"issuerId":"00000000-0000-4000-8000-000000000001","name":"Alpha Display","ticker":"ALP","exchange":"LSE","isin":"GB1234567890"}',
 repeat('b',64),clock_timestamp()+interval '90 seconds'),
('20000000-0000-4000-8000-000000000003','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"Revoke Co","ticker":"RVK","exchange":"XAMS","isin":"NL1111111111"}',
 repeat('c',64),clock_timestamp()+interval '90 seconds'),
('20000000-0000-4000-8000-000000000004','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"Expired Co","ticker":"EXP","exchange":"XAMS","isin":"NL2222222222"}',
 repeat('d',64),clock_timestamp()-interval '1 second');

DO $priv$
DECLARE role record;
BEGIN
  FOR role IN SELECT rolname,rolcanlogin,rolinherit,rolbypassrls,rolsuper
    FROM pg_roles WHERE rolname IN (
    'orotitan_d32_private_executor','orotitan_d32_approval_issuer',
    'orotitan_d32_command_submitter') LOOP
    IF role.rolcanlogin OR role.rolinherit OR role.rolbypassrls OR role.rolsuper THEN
      RAISE EXCEPTION 'D32_PRIVILEGED_ROLE_PROPERTY';
    END IF;
  END LOOP;
  IF (SELECT count(*) FROM pg_roles WHERE rolname IN (
      'orotitan_d32_private_executor','orotitan_d32_approval_issuer',
      'orotitan_d32_command_submitter'))<>3
   OR has_table_privilege('orotitan_d32_command_submitter',
      'orotitan_mobile_private.previews','SELECT')
   OR has_table_privilege('orotitan_d32_command_submitter',
      'orotitan_mobile_private.command_requests','INSERT')
   OR has_table_privilege('orotitan_d32_approval_issuer',
      'orotitan_mobile_private.previews','UPDATE')
   OR has_table_privilege('authenticated',
      'orotitan_mobile_private.command_requests','INSERT')
   OR has_function_privilege('authenticated',
      'orotitan_mobile_private.confirm_approved_request_v2(uuid,text)','EXECUTE')
   OR has_function_privilege('orotitan_d32_command_submitter',
      'orotitan_mobile_private.confirm_request_v1(uuid,text,text,text,text)','EXECUTE')
   OR has_function_privilege('orotitan_d32_command_submitter',
      'orotitan_mobile_private.approve_preview_v2(uuid,text,text,text,text,text,text)',
      'EXECUTE')
   OR has_function_privilege('orotitan_d32_approval_issuer',
      'orotitan_mobile_private.confirm_approved_request_v2(uuid,text)','EXECUTE')
   OR NOT has_function_privilege('orotitan_d32_command_submitter',
      'orotitan_mobile_private.confirm_approved_request_v2(uuid,text)','EXECUTE')
   OR NOT has_function_privilege('orotitan_d32_approval_issuer',
      'orotitan_mobile_private.approve_preview_v2(uuid,text,text,text,text,text,text)',
      'EXECUTE')
   OR has_schema_privilege('orotitan_d32_private_executor',
      'orotitan_mobile_private','CREATE')
   OR EXISTS(SELECT 1 FROM pg_proc WHERE proname IN(
      'approve_preview_v2','revoke_preview_v2','confirm_approved_request_v2')
       AND NOT prosecdef)
   OR EXISTS(SELECT 1 FROM pg_proc p JOIN pg_roles r ON r.oid=p.proowner
      WHERE p.proname IN('approve_preview_v2','revoke_preview_v2',
        'confirm_approved_request_v2')
      AND r.rolname <> 'orotitan_d32_private_executor')
   OR EXISTS(SELECT 1 FROM pg_class WHERE relname IN('previews','command_requests')
       AND relnamespace='orotitan_mobile_private'::regnamespace
       AND NOT relrowsecurity)
  THEN RAISE EXCEPTION 'D32_ROLE_OR_RLS_PRIVILEGE_BREACH'; END IF;
END;
$priv$;

-- A missing server-issued proof cannot be used by the limited submitter.
SET ROLE orotitan_d32_command_submitter;
DO $denials$
DECLARE refused boolean := false; r record;
BEGIN
  BEGIN
    SELECT * INTO r FROM orotitan_mobile_private.confirm_approved_request_v2(
      '20000000-0000-4000-8000-000000000001',
      repeat('a',100)||'.'||repeat('b',43));
  EXCEPTION WHEN raise_exception THEN
    refused := SQLERRM='D32_CAPABILITY_ABSENT_REVOKED_EXPIRED_OR_MISMATCH';
  END;
  IF NOT refused THEN RAISE EXCEPTION 'D32_UNAPPROVED_SUBMITTER_ACCEPTED'; END IF;
  BEGIN
    PERFORM * FROM orotitan_mobile_private.confirm_request_v1(
      '20000000-0000-4000-8000-000000000001',
      'approved-subject','approved-chatgpt-client',
      'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',repeat('a',64));
    RAISE EXCEPTION 'D32_DIRECT_D30_FUNCTION_EXPOSED';
  EXCEPTION WHEN insufficient_privilege THEN NULL;
  END;
END;
$denials$;
RESET ROLE;

-- Approval is a separate, independently controlled server-side role.
SET ROLE orotitan_d32_approval_issuer;
DO $issue$
DECLARE tok text := repeat('a',100)||'.'||repeat('b',43);
  resource text := 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp';
  digest_hex text := encode(public.digest(convert_to(tok,'UTF8'),'sha256'),'hex');
  ok boolean := false; denied boolean := false;
BEGIN
  BEGIN
    PERFORM orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000001',
      'add_company','attacker','approved-chatgpt-client',resource,
      repeat('a',64),digest_hex);
  EXCEPTION WHEN raise_exception THEN
    denied:=SQLERRM='D32_APPROVAL_ACTOR_OR_PREVIEW_DENIED';
  END;
  IF NOT denied THEN RAISE EXCEPTION 'D32_ISSUER_ACCEPTED_FORGED_ACTOR'; END IF;
  SELECT orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000001',
      'add_company','approved-subject','approved-chatgpt-client',resource,
      repeat('a',64),digest_hex) INTO ok;
  IF NOT ok THEN RAISE EXCEPTION 'D32_INITIAL_APPROVAL_NOT_STORED'; END IF;
  SELECT orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000001',
      'add_company','approved-subject','approved-chatgpt-client',resource,
      repeat('a',64),digest_hex) INTO ok;
  IF ok THEN RAISE EXCEPTION 'D32_APPROVAL_REISSUED'; END IF;
  denied:=false;
  BEGIN
    PERFORM orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000001',
      'add_company','approved-subject','approved-chatgpt-client',resource,
      repeat('a',64),repeat('f',64));
  EXCEPTION WHEN raise_exception THEN
    denied:=SQLERRM='D32_ALREADY_APPROVED_WITH_DIFFERENT_PROOF';
  END;
  IF NOT denied THEN RAISE EXCEPTION 'D32_DIFFERENT_APPROVAL_PROOF_ACCEPTED'; END IF;

  PERFORM orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000002',
      'request_research','approved-subject','approved-chatgpt-client',
      resource,repeat('b',64),
      encode(public.digest(convert_to(repeat('c',100)||'.'||repeat('d',43),
        'UTF8'),'sha256'),'hex'));
  PERFORM orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000003',
      'add_company','approved-subject','approved-chatgpt-client',
      resource,repeat('c',64),
      encode(public.digest(convert_to(repeat('e',100)||'.'||repeat('f',43),
        'UTF8'),'sha256'),'hex'));
  PERFORM orotitan_mobile_private.revoke_preview_v2(
    '20000000-0000-4000-8000-000000000003');
  denied:=false;
  BEGIN
    PERFORM orotitan_mobile_private.approve_preview_v2(
      '20000000-0000-4000-8000-000000000004',
      'add_company','approved-subject','approved-chatgpt-client',
      resource,repeat('d',64),digest_hex);
  EXCEPTION WHEN raise_exception THEN
    denied:=SQLERRM='D32_APPROVAL_ACTOR_OR_PREVIEW_DENIED';
  END;
  IF NOT denied THEN RAISE EXCEPTION 'D32_EXPIRED_PREVIEW_APPROVED'; END IF;
END;
$issue$;
RESET ROLE;

-- The approver has no direct table rights, and submitter sees no proof digest.
SET ROLE orotitan_d32_command_submitter;
DO $confirm$
DECLARE refused boolean := false; r record;
BEGIN
  BEGIN
    SELECT * INTO r FROM orotitan_mobile_private.confirm_approved_request_v2(
      '20000000-0000-4000-8000-000000000001',
      repeat('z',100)||'.'||repeat('b',43));
  EXCEPTION WHEN raise_exception THEN
    refused:=SQLERRM='D32_CAPABILITY_ABSENT_REVOKED_EXPIRED_OR_MISMATCH';
  END;
  IF NOT refused THEN RAISE EXCEPTION 'D32_FORGED_TOKEN_ACCEPTED'; END IF;
  refused:=false;
  BEGIN
    SELECT * INTO r FROM orotitan_mobile_private.confirm_approved_request_v2(
      '20000000-0000-4000-8000-000000000002',
      repeat('a',100)||'.'||repeat('b',43));
  EXCEPTION WHEN raise_exception THEN
    refused:=SQLERRM='D32_CAPABILITY_ABSENT_REVOKED_EXPIRED_OR_MISMATCH';
  END;
  IF NOT refused THEN RAISE EXCEPTION 'D32_CROSS_PREVIEW_TOKEN_ACCEPTED'; END IF;
  refused:=false;
  BEGIN
    SELECT * INTO r FROM orotitan_mobile_private.confirm_approved_request_v2(
      '20000000-0000-4000-8000-000000000003',
      repeat('e',100)||'.'||repeat('f',43));
  EXCEPTION WHEN raise_exception THEN
    refused:=SQLERRM='D32_CAPABILITY_ABSENT_REVOKED_EXPIRED_OR_MISMATCH';
  END;
  IF NOT refused THEN RAISE EXCEPTION 'D32_REVOKED_TOKEN_ACCEPTED'; END IF;

  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_approved_request_v2(
    '20000000-0000-4000-8000-000000000001',
    repeat('a',100)||'.'||repeat('b',43));
  IF r.kind<>'add_company' OR r.duplicate OR r.state<>'REQUESTED_NOT_EXECUTED'
  THEN RAISE EXCEPTION 'D32_VALID_ADD_NOT_QUEUED'; END IF;
  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_approved_request_v2(
    '20000000-0000-4000-8000-000000000001',
    repeat('a',100)||'.'||repeat('b',43));
  IF NOT r.duplicate THEN RAISE EXCEPTION 'D32_REPLAY_NOT_IDEMPOTENT'; END IF;
  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_approved_request_v2(
    '20000000-0000-4000-8000-000000000002',
    repeat('c',100)||'.'||repeat('d',43));
  IF r.kind<>'request_research' OR r.duplicate OR
    r.state<>'REQUESTED_NOT_EXECUTED' THEN
    RAISE EXCEPTION 'D32_VALID_RESEARCH_NOT_QUEUED';
  END IF;
END;
$confirm$;
RESET ROLE;

DO $final$
DECLARE n int;
BEGIN
 SELECT count(*) INTO n FROM orotitan_mobile_private.command_requests;
 IF n<>2 OR EXISTS(SELECT 1 FROM orotitan_mobile_private.command_requests
   WHERE state<>'REQUESTED_NOT_EXECUTED')
 OR (SELECT count(*) FROM public.issuers)<>2
 OR (SELECT count(*) FROM public.securities)<>2
 OR (SELECT count(*) FROM public.research_dossiers)<>2
 OR (SELECT count(*) FROM public.companies)<>2
 OR to_regclass('public.orotitan_runs') IS NOT NULL THEN
  RAISE EXCEPTION 'D32_REQUESTS_CHANGED_CANONICAL_DATA_OR_REGISTRY';
 END IF;
END;
$final$;
