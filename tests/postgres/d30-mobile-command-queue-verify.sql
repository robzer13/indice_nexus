\set ON_ERROR_STOP on
-- D30 synthetic local-only request verification. No production Besi identity.
UPDATE public.securities
 SET isin='GB1234567890'
 WHERE issuer_id='00000000-0000-4000-8000-000000000001';

INSERT INTO orotitan_mobile_private.previews(
 preview_id,kind,actor_subject,actor_client,actor_resource,
 payload,fingerprint_sha256,expires_at
) VALUES
 ('10000000-0000-4000-8000-000000000010','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"ASML Holding N.V.","ticker":"ASML","exchange":"XAMS","isin":"NL0010273215"}',
 repeat('a',64),clock_timestamp()+interval '90 seconds'),
 ('10000000-0000-4000-8000-000000000011','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"ASML Holding N.V.","ticker":"ASML","exchange":"XAMS","isin":"NL0010273215"}',
 repeat('a',64),clock_timestamp()+interval '90 seconds'),
 ('10000000-0000-4000-8000-000000000012','request_research',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"issuerId":"00000000-0000-4000-8000-000000000001","name":"Alpha Display","ticker":"ALP","exchange":"LSE","isin":"GB1234567890"}',
 repeat('b',64),clock_timestamp()+interval '90 seconds'),
 ('10000000-0000-4000-8000-000000000013','request_research',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"issuerId":"00000000-0000-4000-8000-000000000002","name":"Beta Display","ticker":"BET","exchange":"XPAR","isin":"UNKNOWN"}',
 repeat('c',64),clock_timestamp()+interval '90 seconds'),
 ('10000000-0000-4000-8000-000000000014','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"ASML Holding N.V.","ticker":"ASML","exchange":"XAMS","isin":"NL0010273215"}',
 repeat('d',64),
 clock_timestamp()+interval '90 seconds');
-- The fifth proposal is reserved for an identity-drift negative check.
DO $tests$
DECLARE r record; expected boolean; n int;
  v_resource constant text :=
    'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp';
BEGIN
  -- Direct private DML inaccessible to public / Supabase browser roles.
  IF has_schema_privilege('anon','orotitan_mobile_private','USAGE')
    OR has_schema_privilege('authenticated','orotitan_mobile_private','USAGE')
    OR has_table_privilege('anon','orotitan_mobile_private.command_requests','INSERT')
    OR has_table_privilege('authenticated','orotitan_mobile_private.command_requests','SELECT')
    OR has_function_privilege('authenticated',
      'orotitan_mobile_private.confirm_request_v1(uuid,text,text,text,text)','EXECUTE')
    OR EXISTS(SELECT 1 FROM pg_class WHERE relname IN('previews','command_requests')
        AND relnamespace='orotitan_mobile_private'::regnamespace
        AND NOT relrowsecurity) THEN
      RAISE EXCEPTION 'D30_PRIVATE_SURFACE_EXPOSED';
  END IF;
  -- A forged client cannot confirm someone else's preview.
  expected:=false;
  BEGIN
    PERFORM * FROM orotitan_mobile_private.confirm_request_v1(
      '10000000-0000-4000-8000-000000000010',
      'approved-subject','forged-client',v_resource,repeat('a',64));
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM='D30_PREVIEW_ACTOR_FINGERPRINT_OR_EXPIRY_DENIED' THEN
      expected:=true; ELSE RAISE; END IF;
  END;
  IF NOT expected THEN RAISE EXCEPTION 'D30_FORGED_ACTOR_ACCEPTED'; END IF;
  -- A forged hash cannot alter a confirmed proposal.
  expected:=false;
  BEGIN
    PERFORM * FROM orotitan_mobile_private.confirm_request_v1(
      '10000000-0000-4000-8000-000000000010',
      'approved-subject','approved-chatgpt-client',v_resource,repeat('f',64));
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM='D30_PREVIEW_ACTOR_FINGERPRINT_OR_EXPIRY_DENIED' THEN
      expected:=true; ELSE RAISE; END IF;
  END;
  IF NOT expected THEN RAISE EXCEPTION 'D30_FORGED_FINGERPRINT_ACCEPTED'; END IF;

  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_request_v1(
    '10000000-0000-4000-8000-000000000010',
    'approved-subject','approved-chatgpt-client',v_resource,repeat('a',64));
  IF r.kind<>'add_company' OR r.state<>'REQUESTED_NOT_EXECUTED'
    OR r.duplicate THEN RAISE EXCEPTION 'D30_ADD_NOT_QUEUED'; END IF;

  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_request_v1(
    '10000000-0000-4000-8000-000000000010',
    'approved-subject','approved-chatgpt-client',v_resource,repeat('a',64));
  IF NOT r.duplicate THEN RAISE EXCEPTION 'D30_SAME_PREVIEW_NOT_IDEMPOTENT'; END IF;

  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_request_v1(
    '10000000-0000-4000-8000-000000000011',
    'approved-subject','approved-chatgpt-client',v_resource,repeat('a',64));
  IF NOT r.duplicate THEN RAISE EXCEPTION 'D30_SAME_TARGET_NOT_IDEMPOTENT'; END IF;

  SELECT * INTO STRICT r FROM orotitan_mobile_private.confirm_request_v1(
    '10000000-0000-4000-8000-000000000012',
    'approved-subject','approved-chatgpt-client',v_resource,repeat('b',64));
  IF r.kind<>'request_research' OR r.state<>'REQUESTED_NOT_EXECUTED'
    OR r.duplicate THEN RAISE EXCEPTION 'D30_RESEARCH_NOT_QUEUED'; END IF;
  -- Reconstructing issuer identity from a bogus legacy UNKNOWN ISIN fails.
  expected:=false;
  BEGIN
    PERFORM * FROM orotitan_mobile_private.confirm_request_v1(
      '10000000-0000-4000-8000-000000000013',
      'approved-subject','approved-chatgpt-client',v_resource,repeat('c',64));
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM='D30_CANONICAL_ISSUER_OR_SECURITY_CHANGED' THEN
      expected:=true; ELSE RAISE; END IF;
  END;
  IF NOT expected THEN RAISE EXCEPTION 'D30_UNVERIFIED_ISIN_ACCEPTED'; END IF;

  -- A company which becomes tracked between preview and confirm must fail.
  INSERT INTO public.companies(id,slug,ticker,name,exchange,currency,
    quote_unit,price_decimals,market_data_multiplier,country)
  VALUES('90000000-0000-4000-8000-000000000001','asml-fixture',
    'ASML','ASML Holding N.V.','XAMS','EUR','MAJOR',2,1,'NL');
  expected:=false;
  BEGIN
    PERFORM * FROM orotitan_mobile_private.confirm_request_v1(
      '10000000-0000-4000-8000-000000000014',
      'approved-subject','approved-chatgpt-client',v_resource,repeat('d',64));
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM='D30_COMPANY_ALREADY_EXISTS' THEN
      expected:=true; ELSE RAISE; END IF;
  END;
  IF NOT expected THEN RAISE EXCEPTION 'D30_CHANGED_IDENTITY_ACCEPTED'; END IF;

  SELECT count(*) INTO n FROM orotitan_mobile_private.command_requests;
  IF n<>2 OR (SELECT count(*) FROM public.issuers)<>2
    OR (SELECT count(*) FROM public.research_dossiers)<>2
    OR EXISTS(SELECT 1 FROM orotitan_mobile_private.command_requests
      WHERE state<>'REQUESTED_NOT_EXECUTED') THEN
    RAISE EXCEPTION 'D30_REQUESTS_MUTATED_CANONICAL_STATE';
  END IF;
END $tests$;
-- Late expiry tested outside main DO using an expired preview with valid CHECK.
INSERT INTO orotitan_mobile_private.previews(preview_id,kind,actor_subject,
 actor_client,actor_resource,payload,fingerprint_sha256,created_at,expires_at)
VALUES('10000000-0000-4000-8000-000000000015','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"Another Company","ticker":"ZZZZ","exchange":"XAMS","isin":"NL1234567890"}',
 repeat('e',64),clock_timestamp()-interval '110 seconds',
 clock_timestamp()-interval '20 seconds');
DO $expired$
DECLARE denied boolean:=false;
BEGIN
  BEGIN
    PERFORM * FROM orotitan_mobile_private.confirm_request_v1(
      '10000000-0000-4000-8000-000000000015',
      'approved-subject','approved-chatgpt-client',
      'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
      repeat('e',64));
  EXCEPTION WHEN raise_exception THEN
    IF SQLERRM='D30_PREVIEW_ACTOR_FINGERPRINT_OR_EXPIRY_DENIED' THEN
      denied:=true; ELSE RAISE; END IF;
  END;
  IF NOT denied THEN RAISE EXCEPTION 'D30_EXPIRED_PREVIEW_CONFIRMED'; END IF;
  IF (SELECT count(*) FROM orotitan_mobile_private.command_requests)<>2 THEN
    RAISE EXCEPTION 'D30_EXPIRED_PREVIEW_SIDE_EFFECT';
  END IF;
END $expired$;
