\set ON_ERROR_STOP on
-- D36 synthetic Research only. D33 fixture already assigned Beta title FR1234567890.
-- NO actual market evidence / run created.
INSERT INTO orotitan_mobile_private.previews(
 preview_id,kind,actor_subject,actor_client,actor_resource,
 payload,fingerprint_sha256,created_at,expires_at)
SELECT ('36000000-0000-4000-8000-'||lpad(i::text,12,'0'))::uuid,
 'request_research',
 CASE WHEN i=3 THEN 'person-b-account' ELSE 'person-a-account' END,
 CASE WHEN i=2 THEN 'approved-desktop-client' ELSE 'approved-mobile-client' END,
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"issuerId":"00000000-0000-4000-8000-000000000002","name":"Beta Display","ticker":"BET","exchange":"XPAR","isin":"FR1234567890"}'::jsonb,
 repeat(chr(96+i),64),clock_timestamp(),clock_timestamp()+interval '105 seconds'
FROM generate_series(1,6) AS i;

-- A scope cannot be invented for an arbitrary actor even with the narrow
-- scope issuer role. The actor arguments still require trusted-server binding.
SET ROLE orotitan_d36_scope_issuer;
DO $scope$
DECLARE v_resource text:='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp';
 v_bad boolean:=false;v_id uuid;v_actor text;v_client text;
BEGIN
 BEGIN
 PERFORM orotitan_mobile_private.issue_research_scope_v4(
 '36000000-0000-4000-8000-000000000001','person-b-account',
 'approved-mobile-client',v_resource,
 '00000000-0000-4000-8000-000000000002',repeat('a',64),current_date);
 EXCEPTION WHEN raise_exception THEN
  v_bad:=SQLERRM='D36_SCOPE_NOT_AUTHORIZED_OR_INVALID';
 END;
 IF NOT v_bad THEN RAISE EXCEPTION 'D36_ACTOR_SPOOF_SCOPE_ACCEPTED'; END IF;
 FOR i IN 1..6 LOOP
  v_id:=('36000000-0000-4000-8000-'||lpad(i::text,12,'0'))::uuid;
  v_actor:=CASE WHEN i=3 THEN 'person-b-account' ELSE 'person-a-account' END;
  v_client:=CASE WHEN i=2 THEN 'approved-desktop-client' ELSE 'approved-mobile-client' END;
  IF NOT orotitan_mobile_private.issue_research_scope_v4(
   v_id,v_actor,v_client,v_resource,
   '00000000-0000-4000-8000-000000000002',
   CASE WHEN i=5 THEN repeat('b',64) ELSE repeat('a',64) END,
   CASE WHEN i=4 THEN current_date-1 ELSE current_date END) THEN
    RAISE EXCEPTION 'D36_SCOPE_INSERT_FAILED';
  END IF;
 END LOOP;
END;
$scope$;
RESET ROLE;

-- Only a separate capability approver binds each preview to an opaque,
-- server-only token proof. Synthetic sealed values model D32 digest, not HMAC.
SET ROLE orotitan_d32_approval_issuer;
DO $approve$
DECLARE v_id uuid;v_resource text:='https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp';
 v_actor text;v_client text;v_token text;
BEGIN
 FOR i IN 1..6 LOOP
  v_id:=('36000000-0000-4000-8000-'||lpad(i::text,12,'0'))::uuid;
  v_actor:=CASE WHEN i=3 THEN 'person-b-account' ELSE 'person-a-account' END;
  v_client:=CASE WHEN i=2 THEN 'approved-desktop-client' ELSE 'approved-mobile-client' END;
  v_token:=repeat(chr(64+i),100)||'.'||repeat(chr(64+i),43);
  IF NOT orotitan_mobile_private.approve_preview_v2(
    v_id,'request_research',v_actor,v_client,v_resource,
    repeat(chr(96+i),64),
    encode(public.digest(convert_to(v_token,'UTF8'),'sha256'),'hex')) THEN
   RAISE EXCEPTION 'D36_APPROVAL_MISSING';
  END IF;
 END LOOP;
END;
$approve$;
RESET ROLE;

SET ROLE orotitan_d36_research_submitter;
DO $confirm$
DECLARE a uuid;b uuid;alt uuid;cutoff uuid;contract uuid; spoof boolean:=false;
 r record;
BEGIN
 -- Cannot use someone else's token even if all issuer fields are identical.
 BEGIN
  PERFORM * FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000001',
   repeat('C',100)||'.'||repeat('C',43));
 EXCEPTION WHEN raise_exception THEN
  spoof:=SQLERRM='D36_CAPABILITY_DENIED';
 END;
 IF NOT spoof THEN RAISE EXCEPTION 'D36_OTHER_ACTOR_TOKEN_ACCEPTED'; END IF;
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000001',
   repeat('A',100)||'.'||repeat('A',43));
 a:=r.request_id;
 IF r.duplicate OR r.state<>'REQUESTED_NOT_EXECUTED' THEN
  RAISE EXCEPTION 'D36_PRIMARY_REQUEST_FAILED'; END IF;
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000002',
   repeat('B',100)||'.'||repeat('B',43));
 IF r.request_id<>a OR NOT r.duplicate THEN
  RAISE EXCEPTION 'D36_SAME_USER_SECOND_DEVICE_NOT_DEDUPLICATED';
 END IF;
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000003',
   repeat('C',100)||'.'||repeat('C',43));
 b:=r.request_id;
 IF b=a OR r.duplicate THEN RAISE EXCEPTION 'D36_OTHER_USER_NOT_ISOLATED'; END IF;
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000004',
   repeat('D',100)||'.'||repeat('D',43));
 cutoff:=r.request_id;
 IF cutoff=a OR cutoff=b OR r.duplicate THEN
  RAISE EXCEPTION 'D36_DIFFERENT_CUTOFF_COLLISION'; END IF;
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000005',
   repeat('E',100)||'.'||repeat('E',43));
 contract:=r.request_id;
 IF contract=a OR contract=b OR contract=cutoff OR r.duplicate THEN
  RAISE EXCEPTION 'D36_DIFFERENT_CONTRACT_COLLISION'; END IF;
 -- Retry same approved preview never changes its linked request.
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
   '36000000-0000-4000-8000-000000000001',
   repeat('A',100)||'.'||repeat('A',43));
 IF r.request_id<>a OR NOT r.duplicate THEN
  RAISE EXCEPTION 'D36_SAME_PREVIEW_REPLAY_CHANGED_REQUEST'; END IF;
END;
$confirm$;
RESET ROLE;
-- Resolve the test request ID in the fixture owner session; the
-- narrow lifecycle role itself has NO direct table SELECT permission.
SELECT set_config('d36.fixture_request_to_close',request_id::text,false)
FROM orotitan_mobile_private.research_requests_v4
WHERE actor_subject='person-a-account'
AND contract_pack_sha256=repeat('a',64)
AND cutoff_date=current_date
AND state='REQUESTED_NOT_EXECUTED';
-- Independent worker closes A's original request, retaining its receipt.
SET ROLE orotitan_d36_lifecycle_controller;
DO $close$
DECLARE id uuid;denied boolean:=false;
BEGIN
 id:=current_setting('d36.fixture_request_to_close')::uuid;
 IF NOT orotitan_mobile_private.close_research_request_v4(id,'FULFILLED') THEN
  RAISE EXCEPTION 'D36_CLOSURE_FAILED';
 END IF;
 BEGIN
  PERFORM orotitan_mobile_private.close_research_request_v4(id,'FULFILLED');
 EXCEPTION WHEN raise_exception THEN
  denied:=SQLERRM='D36_UNKNOWN_OR_ALREADY_CLOSED_REQUEST';
 END;
 IF NOT denied THEN RAISE EXCEPTION 'D36_TERMINAL_REQUEST_REOPENED'; END IF;
END;
$close$;
RESET ROLE;

SET ROLE orotitan_d36_research_submitter;
DO $reissue$
DECLARE original uuid;new_id uuid;r record;
BEGIN
 original:=current_setting('d36.fixture_request_to_close')::uuid;
 SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_scoped_research_v4(
  '36000000-0000-4000-8000-000000000006',
  repeat('F',100)||'.'||repeat('F',43));
 new_id:=r.request_id;
 IF r.duplicate OR new_id=original THEN
  RAISE EXCEPTION 'D36_LIFECYCLE_REISSUE_DID_NOT_CREATE_NEW_ID'; END IF;
END;
$reissue$;
RESET ROLE;

DO $assertions$
BEGIN
 IF (SELECT count(*) FROM orotitan_mobile_private.research_requests_v4)<>5
 OR (SELECT count(*) FROM orotitan_mobile_private.research_requests_v4
    WHERE state='REQUESTED_NOT_EXECUTED')<>4
 OR (SELECT count(*) FROM orotitan_mobile_private.research_requests_v4
    WHERE state='FULFILLED')<>1
 OR (SELECT count(*) FROM orotitan_mobile_private.research_requests_v4
    WHERE actor_subject='person-b-account')<>1
 OR (SELECT count(DISTINCT confirmed_request_id)
    FROM orotitan_mobile_private.research_preview_scope_v4)<>5
 OR (SELECT count(*) FROM public.issuers)<>2
 OR (SELECT count(*) FROM public.research_dossiers)<>2
 OR to_regclass('public.orotitan_runs') IS NOT NULL THEN
   RAISE EXCEPTION 'D36_CROSS_USER_CUTOFF_OR_REGISTRY_INVARIANT_FAILED';
 END IF;
 IF has_table_privilege('authenticated',
   'orotitan_mobile_private.research_requests_v4','SELECT')
 OR has_table_privilege('orotitan_d36_research_submitter',
   'orotitan_mobile_private.research_requests_v4','SELECT')
 OR has_function_privilege('authenticated',
   'orotitan_mobile_private.confirm_scoped_research_v4(uuid,text)','EXECUTE')
 OR has_function_privilege('orotitan_d36_research_submitter',
   'orotitan_mobile_private.close_research_request_v4(uuid,text)','EXECUTE')
 OR has_function_privilege('orotitan_d36_lifecycle_controller',
   'orotitan_mobile_private.confirm_scoped_research_v4(uuid,text)','EXECUTE')
 OR NOT (SELECT relrowsecurity FROM pg_class
    WHERE oid='orotitan_mobile_private.research_requests_v4'::regclass)
 OR NOT (SELECT relrowsecurity FROM pg_class
    WHERE oid='orotitan_mobile_private.research_preview_scope_v4'::regclass) THEN
    RAISE EXCEPTION 'D36_ACCESS_CONTROL_MATRIX_EXPOSED';
 END IF;
END;
$assertions$;
\echo 'D36 tenant-scoped Research lifecycle and isolation PASS (disposable PG17 only).'
