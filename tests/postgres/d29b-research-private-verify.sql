-- D29b private SQL: isolated PG17 positive and adversarial assertions.
-- Executed only after disposable fixture and review-only SQL.
INSERT INTO orotitan_research_intake_private.submission_grants(
  grant_id,run_id,issuer_id,subject_id,client_id,mcp_resource,
  contract_pins_sha256,expected_run_version,expected_stage_version,
  data_cutoff,issued_at,expires_at
) VALUES(
  'approved-intake-grant',
  '48b9426b-4d0d-4858-9d87-efcd13ac415d',
  '33333333-3333-4333-8333-333333333333',
  'approved-oauth-sub','approved-client',
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
  repeat('a',64),6,4,'2026-10-12',now()-interval '10 sec',
  now()+interval '2 min'
),(
  'revoked-intake-grant',
  '48b9426b-4d0d-4858-9d87-efcd13ac415d',
  '33333333-3333-4333-8333-333333333333',
  'approved-oauth-sub','approved-client',
  'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
  repeat('a',64),6,4,'2026-10-12',now()-interval '10 sec',
  now()+interval '2 min'
);
UPDATE orotitan_research_intake_private.submission_grants
  SET revoked_at=now() WHERE grant_id='revoked-intake-grant';
INSERT INTO orotitan_research_intake_private.verifier_grants(
  verifier_grant_id,run_id,issuer_id,reviewer_id,
  expected_run_version,expected_stage_version,issued_at,expires_at
) VALUES(
  'approved-reviewer-grant',
  '48b9426b-4d0d-4858-9d87-efcd13ac415d',
  '33333333-3333-4333-8333-333333333333',
  'independent-d2',6,4,now()-interval '10 sec',now()+interval '2 min'
);

DO $privileges$
BEGIN
  IF has_schema_privilege('anon','orotitan_research_intake_private','USAGE')
    OR has_schema_privilege('authenticated','orotitan_research_intake_private','USAGE')
    OR has_table_privilege('authenticated',
      'orotitan_research_intake_private.receipts','SELECT')
    OR has_table_privilege('orotitan_research_intake_reader',
      'orotitan_research_intake_private.receipts','INSERT')
    OR has_table_privilege('orotitan_research_intake_writer',
      'public.orotitan_runs','UPDATE')
    OR has_table_privilege('orotitan_research_intake_writer',
      'public.orotitan_run_stages','UPDATE')
    OR has_table_privilege('orotitan_research_intake_writer',
      'orotitan_research_intake_private.receipts','INSERT')
    OR has_table_privilege('orotitan_research_intake_writer',
      'orotitan_research_intake_private.receipts','UPDATE')
    OR has_table_privilege('orotitan_research_intake_writer',
      'orotitan_research_intake_private.receipts','SELECT')
    OR has_table_privilege('orotitan_research_intake_writer',
      'orotitan_research_intake_private.receipt_audit','INSERT')
    OR has_table_privilege('orotitan_research_intake_reader',
      'orotitan_research_intake_private.receipts','UPDATE')
    OR has_function_privilege('authenticated',
      'orotitan_research_intake_private.commit_receipt_v1(uuid,text,text,text,text,text,bigint,bigint,date,text,text,text,text,text,bigint,text)',
      'EXECUTE')
    OR has_schema_privilege('orotitan_research_intake_receipt_executor',
      'orotitan_research_intake_private','CREATE') THEN
    RAISE EXCEPTION 'D29b unauthorized privilege detected';
  END IF;
END;
$privileges$;

SET ROLE orotitan_research_intake_writer;
DO $scope$
DECLARE v boolean;
BEGIN
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64)
  ) INTO v;
  IF NOT v THEN RAISE EXCEPTION 'valid Besi preflight unexpectedly failed'; END IF;
  SELECT orotitan_research_intake_private.preflight_v1(
    'd56a0bc9-4e80-48b3-b326-74d1fea15e63',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64)
  ) INTO v;
  IF v THEN RAISE EXCEPTION 'cross-run preflight admitted'; END IF;
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','forged-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64)
  ) INTO v;
  IF v THEN RAISE EXCEPTION 'wrong client preflight admitted'; END IF;
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'revoked-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64)
  ) INTO v;
  IF v THEN RAISE EXCEPTION 'revoked grant admitted'; END IF;
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,3,'2026-10-12',repeat('a',64)
  ) INTO v;
  IF v THEN RAISE EXCEPTION 'stale stage admitted'; END IF;
END;
$scope$;
RESET ROLE;

-- Direct caller DML bypass must be impossible even if it knows all values.
DO $privileged$
BEGIN
  IF EXISTS(SELECT 1 FROM pg_roles WHERE rolname='orotitan_research_intake_receipt_executor'
            AND (rolcanlogin OR rolbypassrls OR rolsuper)) THEN
    RAISE EXCEPTION 'isolated SECURITY DEFINER owner must have zero login/bypass';
  END IF;
  IF NOT has_function_privilege('orotitan_research_intake_writer',
      'orotitan_research_intake_private.commit_receipt_v1(uuid,text,text,text,text,text,bigint,bigint,date,text,text,text,text,text,bigint,text)',
      'EXECUTE') THEN
    RAISE EXCEPTION 'external writer cannot call the guarded function';
  END IF;
END;
$privileged$;
SET ROLE orotitan_research_intake_writer;
DO $direct_dml$
DECLARE denied boolean;
BEGIN
  denied:=false;
  BEGIN
    INSERT INTO orotitan_research_intake_private.receipts(
      run_id,issuer_id,stage,run_state_version,stage_state_version,
      contract_pins_sha256,grant_id,sha256,file_name,media_type,object_path,
      size_bytes,source_date,submitted_by_subject,submitted_by_client)
    VALUES('48b9426b-4d0d-4858-9d87-efcd13ac415d',
      '33333333-3333-4333-8333-333333333333','RESEARCH',6,4,
      repeat('a',64),'approved-intake-grant',repeat('f',64),
      'forged.json','application/json',
      'research-intake/v1/forged',15,'2026-07-23','rogue','rogue');
  EXCEPTION WHEN insufficient_privilege THEN denied:=true; END;
  IF NOT denied THEN RAISE EXCEPTION 'DIRECT_BYPASS_INSERT_ACCEPTED'; END IF;
  denied:=false;
  BEGIN
    UPDATE orotitan_research_intake_private.receipts SET file_name='rogue.json';
  EXCEPTION WHEN insufficient_privilege THEN denied:=true; END;
  IF NOT denied THEN RAISE EXCEPTION 'DIRECT_BYPASS_UPDATE_ACCEPTED'; END IF;
END;
$direct_dml$;
RESET ROLE;

BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL ROLE orotitan_research_intake_writer;
DO $first$
DECLARE r record;
BEGIN
  SELECT * INTO r FROM orotitan_research_intake_private.commit_receipt_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64),repeat('c',64),
    'evidence.json','application/json',
    'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
      repeat('c',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac01.json',
    15,'2026-07-23'
  );
  IF r.receipt_id IS NULL OR r.duplicate OR r.stage<>'RESEARCH' OR
     r.run_state_version<>6 OR r.stage_state_version<>4 THEN
    RAISE EXCEPTION 'first receipt should be valid and unverified';
  END IF;
END;
$first$;
COMMIT;

BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL ROLE orotitan_research_intake_writer;
DO $dup$
DECLARE r record;
BEGIN
  SELECT * INTO r FROM orotitan_research_intake_private.commit_receipt_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64),repeat('c',64),
    'evidence.json','application/json',
    'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
      repeat('c',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac02.json',
    15,'2026-07-23'
  );
  IF NOT r.duplicate OR r.object_path NOT LIKE '%.json' OR
      r.receipt_id IS NULL THEN
    RAISE EXCEPTION 'duplicate should yield old receipt';
  END IF;
END;
$dup$;
COMMIT;

BEGIN ISOLATION LEVEL SERIALIZABLE;
SET LOCAL ROLE orotitan_research_intake_writer;
DO $denials$
DECLARE caught boolean;
BEGIN
  caught:=false;
  BEGIN
    PERFORM * FROM orotitan_research_intake_private.commit_receipt_v1(
      '48b9426b-4d0d-4858-9d87-efcd13ac415d',
      '33333333-3333-4333-8333-333333333333',
      'approved-intake-grant','approved-oauth-sub','forged-client',
      'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
      6,4,'2026-10-12',repeat('a',64),repeat('d',64),
      'evidence.json','application/json',
      'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
        repeat('d',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac03.json',
      15,'2026-07-23');
  EXCEPTION WHEN OTHERS THEN caught:=true; END;
  IF NOT caught THEN RAISE EXCEPTION 'wrong OAuth client commit admitted'; END IF;

  caught:=false;
  BEGIN
    PERFORM * FROM orotitan_research_intake_private.commit_receipt_v1(
      '48b9426b-4d0d-4858-9d87-efcd13ac415d',
      '33333333-3333-4333-8333-333333333333',
      'approved-intake-grant','approved-oauth-sub','approved-client',
      'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
      6,4,'2026-10-12',repeat('a',64),repeat('d',64),
      'evidence.json','application/json',
      'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
        repeat('d',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac03.json',
      15,'2026-10-13');
  EXCEPTION WHEN OTHERS THEN caught:=true; END;
  IF NOT caught THEN RAISE EXCEPTION 'post-cutoff source admitted'; END IF;

  caught:=false;
  BEGIN
    PERFORM * FROM orotitan_research_intake_private.commit_receipt_v1(
      '48b9426b-4d0d-4858-9d87-efcd13ac415d',
      '33333333-3333-4333-8333-333333333333',
      'approved-intake-grant','approved-oauth-sub','approved-client',
      'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
      6,4,'2026-10-12',repeat('a',64),repeat('c',64),
      'different-name.json','application/json',
      'research-intake/v1/48b9426b-4d0d-4858-9d87-efcd13ac415d/RESEARCH/'||
        repeat('c',64)||'/b78d8826-9f6b-4d9c-a1b8-ec5d13faac04.json',
      15,'2026-07-23');
  EXCEPTION WHEN OTHERS THEN caught:=true; END;
  IF NOT caught THEN RAISE EXCEPTION 'mismatched duplicate admitted'; END IF;
END;
$denials$;
COMMIT;

RESET ROLE;
DO $audit$
DECLARE n bigint;
BEGIN
  SELECT count(*) INTO n FROM orotitan_research_intake_private.receipts;
  IF n<>1 THEN RAISE EXCEPTION 'expected exactly one idempotent receipt: %',n; END IF;
  SELECT count(*) INTO n FROM orotitan_research_intake_private.receipt_audit;
  IF n<>1 THEN RAISE EXCEPTION 'receipt+audit must be atomic'; END IF;
  SELECT count(*) INTO n FROM public.orotitan_runs
    WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d' AND state_version=6;
  IF n<>1 THEN RAISE EXCEPTION 'Registry run was mutated'; END IF;
  SELECT count(*) INTO n FROM public.orotitan_run_stages
    WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'
      AND stage_code='RESEARCH' AND state_version=4;
  IF n<>1 THEN RAISE EXCEPTION 'Registry stage was mutated'; END IF;
END;
$audit$;

SET ROLE orotitan_research_intake_reader;
DO $verifier$
DECLARE n bigint;
BEGIN
  SELECT count(*) INTO n
    FROM orotitan_research_intake_private.verifier_metadata_v1(
      '48b9426b-4d0d-4858-9d87-efcd13ac415d',
      '33333333-3333-4333-8333-333333333333','independent-d2',
      6,4,repeat('c',64));
  IF n<>1 THEN RAISE EXCEPTION 'independent verifier metadata missing'; END IF;
  SELECT count(*) INTO n
    FROM orotitan_research_intake_private.verifier_metadata_v1(
      '48b9426b-4d0d-4858-9d87-efcd13ac415d',
      '33333333-3333-4333-8333-333333333333','forged-reviewer',
      6,4,repeat('c',64));
  IF n<>0 THEN RAISE EXCEPTION 'forged reviewer could read'; END IF;
  SELECT count(*) INTO n
    FROM orotitan_research_intake_private.verifier_metadata_v1(
      'd56a0bc9-4e80-48b3-b326-74d1fea15e63',
      '33333333-3333-4333-8333-333333333333','independent-d2',
      6,4,repeat('c',64));
  IF n<>0 THEN RAISE EXCEPTION 'cross-run reviewer could read'; END IF;
END;
$verifier$;
RESET ROLE;

-- Staleness is checked against the CURRENT Registry, even though receipts
-- remain historical unverified drafts; no transition occurs in this test.
UPDATE public.orotitan_run_stages SET state_version=5
 WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'
   AND stage_code='RESEARCH';
SET ROLE orotitan_research_intake_writer;
DO $stale$
DECLARE v boolean;
BEGIN
  SELECT orotitan_research_intake_private.preflight_v1(
    '48b9426b-4d0d-4858-9d87-efcd13ac415d',
    '33333333-3333-4333-8333-333333333333',
    'approved-intake-grant','approved-oauth-sub','approved-client',
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
    6,4,'2026-10-12',repeat('a',64)
  ) INTO v;
  IF v THEN RAISE EXCEPTION 'new Registry version was ignored'; END IF;
END;
$stale$;
RESET ROLE;
UPDATE public.orotitan_run_stages SET state_version=4
 WHERE run_id='48b9426b-4d0d-4858-9d87-efcd13ac415d'
   AND stage_code='RESEARCH';
