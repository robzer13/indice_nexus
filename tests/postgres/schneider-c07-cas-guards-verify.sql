\set ON_ERROR_STOP on
-- C07 public CI: SYNTHETIC CAS guard checks only. Not Schneider historical replay.
-- All mutations occur in the throwaway local PG17 database and are rolled back.
BEGIN;
DO $c07$
DECLARE
  v_run uuid := gen_random_uuid();
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_pins jsonb := '{}'::jsonb;
  v_key text;
  v_name text;
  v_version text;
  v_hash text;
  v_contract_hash text;
  v_before_artifacts bigint;
  v_before_edges bigint;
  v_before_events bigint;
  v_err text;
  v_count integer := 0;
BEGIN
  FOREACH v_key IN ARRAY ARRAY[
    'process','pilotage','research_stage','deep_dive_stage','integration_stage',
    'analysis_standard','master_prompt','investment_policy','execution_patch',
    'integration_spec','screener_schema','i2','i3b'
  ] LOOP
    v_name := CASE v_key
      WHEN 'research_stage' THEN 'RESEARCH'
      WHEN 'deep_dive_stage' THEN 'DEEP_DIVE'
      WHEN 'integration_stage' THEN 'INTEGRATION'
      ELSE upper(v_key) END;
    v_version := CASE v_key
      WHEN 'pilotage' THEN '1.0.1'
      WHEN 'investment_policy' THEN '1.0.0'
      WHEN 'execution_patch' THEN '1.0.1'
      WHEN 'screener_schema' THEN '1.0.0'
      ELSE '1.0' END;
    v_hash := encode(extensions.digest(convert_to('c07:'||v_key,'UTF8'),'sha256'),'hex');
    v_pins := v_pins || jsonb_build_object(v_key,jsonb_build_object(
      'name',v_name,'version',v_version,'content_sha256',v_hash,
      'locator',jsonb_build_object(
        'backend','GITHUB_IMMUTABLE','repository','robzer13/indice_nexus',
        'path','contracts/orotitan-equity/v1/execution/'||v_key||'.md',
        'commit_sha',repeat('a',40),'blob_sha',repeat('b',40))));
  END LOOP;
  v_contract_hash := public.orotitan_contract_set_sha256(v_pins);

  SELECT m.issuer_id,m.security_id,m.dossier_id INTO v_issuer,v_security,v_dossier
    FROM public.legacy_company_identity_map m
    ORDER BY m.legacy_company_id LIMIT 1;
  IF v_issuer IS NULL THEN RAISE EXCEPTION 'C07_FIXTURE_IDENTITY_MISSING'; END IF;

  INSERT INTO public.orotitan_runs (
    run_id,creation_idempotency_key,run_scope,issuer_id,security_id,dossier_id,
    entry_path,canonical_mode,run_type,run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,contract_pins,contract_set_sha256,state_version
  ) VALUES (
    v_run,'c07-synthetic-cas:'||v_run::text,'COMPANY_ANALYSIS',
    v_issuer,v_security,v_dossier,'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'ACTIVE','DEEP_DIVE',date '2026-10-30',
    v_pins->'process'->>'version',v_pins->'pilotage'->>'version',
    v_pins,v_contract_hash,7
  );
  INSERT INTO public.orotitan_run_stages (
    run_id,stage_code,stage_revision,stage_contract_name,stage_contract_version,
    stage_contract_sha256,lifecycle_status,contract_status_code,
    handoff_gate_name,handoff_gate_state,state_version,started_at
  ) VALUES (
    v_run,'DEEP_DIVE',1,
    v_pins->'deep_dive_stage'->>'name',
    v_pins->'deep_dive_stage'->>'version',
    v_pins->'deep_dive_stage'->>'content_sha256',
    'IN_PROGRESS','DEEP_DIVE_IN_PROGRESS','READY_FOR_INTEGRATION','NOT_EVALUATED',
    2,'2026-10-30T10:00:00Z'
  );

  SELECT count(*) INTO v_before_artifacts FROM public.orotitan_artifacts WHERE run_id=v_run;
  SELECT count(*) INTO v_before_edges FROM public.orotitan_artifact_edges WHERE child_run_id=v_run;
  SELECT count(*) INTO v_before_events FROM public.orotitan_run_events WHERE run_id=v_run;

  -- Invalid run CAS 6/2 must be rejected before manifest handling.
  BEGIN
    PERFORM public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',6,2,'{}'::jsonb,'{}'::jsonb,'[]'::jsonb,'[]'::jsonb,
      'IN_PROGRESS','c07-run-cas:'||v_run::text,repeat('c',64),'PILOTAGE');
    RAISE EXCEPTION 'C07_RUN_CAS_UNEXPECTED_ACCEPT';
  EXCEPTION WHEN SQLSTATE '40001' THEN
    GET STACKED DIAGNOSTICS v_err = MESSAGE_TEXT;
    IF v_err NOT LIKE '%RUN_STATE_VERSION_MISMATCH%' THEN RAISE; END IF;
    v_count := v_count+1;
  END;

  -- Invalid Deep Dive CAS 7/1 must be rejected separately.
  BEGIN
    PERFORM public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',7,1,'{}'::jsonb,'{}'::jsonb,'[]'::jsonb,'[]'::jsonb,
      'IN_PROGRESS','c07-stage-cas:'||v_run::text,repeat('d',64),'PILOTAGE');
    RAISE EXCEPTION 'C07_STAGE_CAS_UNEXPECTED_ACCEPT';
  EXCEPTION WHEN SQLSTATE '40001' THEN
    GET STACKED DIAGNOSTICS v_err = MESSAGE_TEXT;
    IF v_err NOT LIKE '%STAGE_STATE_VERSION_MISMATCH%' THEN RAISE; END IF;
    v_count := v_count+1;
  END;

  -- A matching 7/2 CAS never authorizes an absent manifest or receipts.
  BEGIN
    PERFORM public.checkpoint_orotitan_stage(
      v_run,'DEEP_DIVE',7,2,'{}'::jsonb,'{}'::jsonb,'[]'::jsonb,'[]'::jsonb,
      'IN_PROGRESS','c07-bad-bundle:'||v_run::text,repeat('e',64),'PILOTAGE');
    RAISE EXCEPTION 'C07_UNATTESTED_BUNDLE_UNEXPECTED_ACCEPT';
  EXCEPTION WHEN SQLSTATE '23514' THEN
    GET STACKED DIAGNOSTICS v_err = MESSAGE_TEXT;
    IF v_err NOT LIKE '%MANIFEST_CONTRACT_MISMATCH%'
       AND v_err NOT LIKE '%BUNDLE_PERSISTENCE%'
       AND v_err NOT LIKE '%ARTIFACT_NOT_IN_MANIFEST%'
    THEN RAISE; END IF;
    v_count := v_count+1;
  END;

  IF v_count <> 3 THEN RAISE EXCEPTION 'C07_NEGATIVE_COUNT_MISMATCH'; END IF;
  IF (SELECT state_version FROM public.orotitan_runs WHERE run_id=v_run) <> 7
     OR (SELECT state_version FROM public.orotitan_run_stages
         WHERE run_id=v_run AND stage_code='DEEP_DIVE') <> 2
     OR (SELECT count(*) FROM public.orotitan_artifacts WHERE run_id=v_run) <> v_before_artifacts
     OR (SELECT count(*) FROM public.orotitan_artifact_edges WHERE child_run_id=v_run) <> v_before_edges
     OR (SELECT count(*) FROM public.orotitan_run_events WHERE run_id=v_run) <> v_before_events
  THEN RAISE EXCEPTION 'C07_REJECTED_RPC_LEFT_PARTIAL_MUTATION'; END IF;
  RAISE NOTICE 'C07 synthetic checkpoint CAS guards: PASS 3/3; state 7/2 unchanged';
END
$c07$;
ROLLBACK;
-- SCHNEIDER_REAL_REPLAY_STATUS=NOT_EXECUTED; not a GO for V2 persistence.
