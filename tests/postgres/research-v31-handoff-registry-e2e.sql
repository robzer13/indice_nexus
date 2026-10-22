\set ON_ERROR_STOP on
-- This is a full Registry transaction test, on a disposable local PostgreSQL 17 DB.
-- Run the live V1 bundle attestation regression in the SAME session; reuse
-- its management-plane, exact-byte receipt helpers for the V3.1 transition.
\ir registry-v1-bundle-persistence-integrity-verify.sql

DO $$
DECLARE
  v_issuer uuid;
  v_security uuid;
  v_dossier uuid;
  v_run uuid;
  v_pins jsonb := $pins${"process":{"name":"OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"7e074c26f3bb0f1949e9e7dc0e8c9c3404b0e751561a2e9e8c4f72e0408861ee","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"d1419e097e19b49c119608a316d94d48a318d29f"}},"pilotage":{"name":"OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"050a1a4c6b3662cc19dd78357a4ba556c61aaa227f95c74475d66f388172709a","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"49a346bfeccef51f798b648c823df4db1e81b7eb"}},"research_stage":{"name":"OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0","version":"2.0","content_sha256":"cdd62f087ff8f8dc97f0c16586d634001f2cef1c44f9e4540e087778cdaa7a01","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md","commit_sha":"86b227a75275cf4aaec6ef61ea2a27e87e2bfec7","blob_sha":"4b49d4ab670717d844ff0ed3c925a2004c043dfa"}},"deep_dive_stage":{"name":"OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"2a5cd0a7a7004212e9b3942f129b63d1743710023148c25de4d99b75e4c55ab9","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"2cc80f42a37930044465b6a8112b4ba1b9b4d7b2"}},"integration_stage":{"name":"OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"426aa192b02372f9081f383d28323482ba8878077d25a0ff579c04dadaf3d05a","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"13e0b9a6953d6502989a74111c57e86d73f8b40d"}},"analysis_standard":{"name":"OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"538b70a975d00b8bfc9c3abf705d5efd425852516680d499db13d8c2ac17174f","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/01_ANALYSIS_STANDARD_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"bfd034c6a650bb2bd0db1ad3cb3d4476aa9ca66e"}},"master_prompt":{"name":"02_OROTITAN_MASTER_PROMPT_V1_2_VALUATION_DATE_ALIGNMENT_PATCH","version":"1.2","content_sha256":"b6962e07ffc36b33ac08984aab0edf3abac58deb4f597d37c495a461e5dabc5a","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/02_OROTITAN_MASTER_PROMPT_V1_2_VALUATION_DATE_ALIGNMENT_PATCH.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"48c5ba2ff86bb958126dd61d5ac3a0eb63a5d7e8"}},"investment_policy":{"name":"OROTITAN_INVESTMENT_POLICY_V1.0.0","version":"1.0.0","content_sha256":"66cc29ccaccf4ccf65f58c2959bf95bcf676e67567aa1dd57863b2cb9a2936c8","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_INVESTMENT_POLICY_V1.0.0.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"fd0121bbb14e35370ddd700d9e9845eaf1ec9f75","locator_format":"RAW_CANONICAL_V1"}},"execution_patch":{"name":"OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1","version":"1.0.1","content_sha256":"6e45f39c03e79c34911850c84904b36f3d95266288a5528127a59d0b8d675973","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"449bb43b664eae9d8c9fd98440e5511c6fbf4be3","locator_format":"RAW_CANONICAL_V1"}},"integration_spec":{"name":"04_INTEGRATION_SPEC_V3.1","version":"3.1","content_sha256":"8a912b821769d30db1e6953b5b8e47a79c5bdb90490d0f3d9ff2f570640d41a2","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_INTEGRATION_SPEC_V3_1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"8099d1c7f4544e4902e48cbd2491671fafc20e9f"}},"screener_schema":{"name":"04_SCREENER_SCHEMA_V3_1","version":"3.1.0","content_sha256":"8d27e947b5d0989ed43f1600bf03f2afae00c7b211bbfe008f13fe6cd6c4c64d","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_SCREENER_SCHEMA_V3_1.json","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"be68c1dec97c7e0a0967241152858c65298dfa43"}},"i2":{"name":"I2_CANONICAL_COMPUTATION_V1.1","version":"1.1","content_sha256":"ebf4d0e23784aaf67fb48f1bd372ebe1bd8efdbef9dafbeb8ecee6b9f4082b00","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I2_CANONICAL_COMPUTATION_V1.1.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"e145448c1f1d901527a6fe9b62309a0f4159a25a"}},"i3b":{"name":"I3B_VALIDATED_SNAPSHOT_WRITER_V1.2","version":"1.2","content_sha256":"b98c3e22e52800eb387c452c28b3051510472ecaa11697fff912865275847557","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER_V1.2.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"1183054128cf276abc626c770da58d39accba49e"}},"dcf_timing":{"name":"OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"9bec7dcb3af85019806f255a0d504cd1770f50fe9343fb92d8f367cb13b5c4ee","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0.md","commit_sha":"1f63baf3c5ae03e5e0a0f90a2794a0f358398c5c","blob_sha":"4316d8e574151882ae25f83c3f7b97f8bad48cd3"}},"valuation_date_alignment":{"name":"OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"3c4e315a5b0759d13b99d77b8af5eb0298e81e46fe9ee3706ed08f3947b68ee5","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"14fe1dcfe413475690e945a78131753dda9df419"}}}$pins$::jsonb;
  v_contract_hash text;
  v_run_sv bigint;
  v_stage_sv bigint;
  v_res jsonb;
  v_manifest jsonb;
  v_checkpoint_id uuid := '40000000-0000-4000-8000-000000000001';
  v_checkpoint_output_id uuid := '40000000-0000-4000-8000-000000000002';
  v_final_id uuid := '40000000-0000-4000-8000-000000000003';
  v_checkpoint_regs jsonb;
  v_checkpoint_refs jsonb;
  v_checkpoint_manifest_reg jsonb;
  v_final_regs jsonb;
  v_final_refs jsonb;
  v_final_manifest_reg jsonb;
  v_false_manifest jsonb;
  v_false_manifest_reg jsonb;
  v_failure boolean;
  v_required text[] := ARRAY[
    'RESEARCH_SOURCE_MANIFEST','EVIDENCE_LEDGER','CONFLICT_LEDGER',
    'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER','RESEARCH_GAP_REGISTER',
    'INPUT_SUFFICIENCY_ASSESSMENT','CLASSIFICATION_INPUT_RECORD'
  ];
BEGIN
  SELECT issuer_id, security_id, dossier_id
  INTO v_issuer, v_security, v_dossier
  FROM public.legacy_company_identity_map ORDER BY legacy_company_id LIMIT 1;
  IF v_issuer IS NULL OR v_security IS NULL OR v_dossier IS NULL THEN
    RAISE EXCEPTION 'D11 E2E: identity fixture is absent';
  END IF;

  v_contract_hash := public.orotitan_contract_set_sha256(v_pins);
  IF v_contract_hash <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2' THEN
    RAISE EXCEPTION 'D11 E2E: actual V3.1 contract pin-pack SHA mismatch';
  END IF;

  v_res := public.create_orotitan_run(
    'd11:v31:create', v_issuer, 'IMPOSED_COMPANY', 'ANALYZE', 'INITIAL',
    date '2026-10-12', NULL, NULL, '3.1', '3.1',
    v_pins, v_contract_hash, repeat('1', 64)
  );
  v_run := (v_res->>'run_id')::uuid;
  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id = v_run;
  PERFORM public.bind_orotitan_run_identity(
    v_run, v_run_sv, v_security, v_dossier, NULL, NULL,
    'd11:v31:bind', repeat('2', 64)
  );
  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id = v_run;
  PERFORM public.start_orotitan_stage(
    v_run, 'RESEARCH', v_run_sv,
    v_pins->'research_stage'->>'name',
    v_pins->'research_stage'->>'version',
    v_pins->'research_stage'->>'content_sha256',
    'd11:v31:research:start',repeat('3',64),'PILOTAGE'
  );

  -- CHECKPOINT / NO must not admit Deep Dive. No downstream stage exists.
  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id = v_run;
  v_failure := FALSE;
  BEGIN
    PERFORM public.start_orotitan_stage(
      v_run, 'DEEP_DIVE', v_run_sv,
      v_pins->'deep_dive_stage'->>'name',
      v_pins->'deep_dive_stage'->>'version',
      v_pins->'deep_dive_stage'->>'content_sha256',
      'd11:v31:early', repeat('4',64), 'PILOTAGE'
    );
  EXCEPTION WHEN check_violation THEN v_failure := TRUE;
  END;
  IF NOT v_failure THEN RAISE EXCEPTION 'D11 E2E: Deep Dive started without Research FINAL'; END IF;

  v_checkpoint_regs := jsonb_build_array(
    pg_temp.bpi_attested_registration(
      v_run,'RESEARCH',v_checkpoint_output_id,1,
      'EVIDENCE_LEDGER_CHECKPOINT','evidence_ledger_checkpoint',
      '{"kind":"CHECKPOINT"}','d11/v31/checkpoint-output.json',
      'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT','{"kind":"CHECKPOINT"}'::jsonb
    )
  );
  SELECT jsonb_agg(pg_temp.bpi_ref(value))
  INTO v_checkpoint_refs FROM jsonb_array_elements(v_checkpoint_regs);
  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0', 'manifest_id',v_checkpoint_id,
    'manifest_kind','CHECKPOINT','run_id',v_run,
    'stage','RESEARCH','stage_revision',1,
    'issuer_id',v_issuer,'security_id',v_security,'dossier_id',v_dossier,
    'canonical_mode','ANALYZE','run_type','INITIAL','data_cutoff','2026-10-12',
    'baseline_snapshot_id',NULL,'process_version','3.1',
    'pilotage_contract_version','3.1','contract_pins',v_pins,
    'stage_contract',v_pins->'research_stage',
    'contract_set_sha256',v_contract_hash,
    'input_artifacts','[]'::jsonb,'output_artifacts',v_checkpoint_refs,
    'stage_status','IN_PROGRESS','contract_status_code','BLOCKED_INSUFFICIENT_INPUT',
    'handoff_gate',jsonb_build_object('name','READY_FOR_DEEP_DIVE','state','NO'),
    'critical_blockers',jsonb_build_array(jsonb_build_object('code','G003','summary','Needs independent adoption evidence')),
    'open_material_limitations','[]'::jsonb,'parent_manifests','[]'::jsonb,
    'started_at','2026-10-12T12:00:00Z','completed_at',NULL
  );
  v_checkpoint_manifest_reg := pg_temp.bpi_attested_registration(
    v_run,'RESEARCH',v_checkpoint_id,1,'RESEARCH_STAGE_MANIFEST',
    'research_stage_manifest',v_manifest::text,
    'd11/v31/checkpoint-manifest.json',
    'CHECKPOINT_STAGE_OUTPUT','CHECKPOINT',v_manifest
  );
  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id = v_run;
  SELECT state_version INTO v_stage_sv FROM public.orotitan_run_stages WHERE run_id = v_run AND stage_code='RESEARCH';
  PERFORM public.checkpoint_orotitan_stage(
    v_run,'RESEARCH',v_run_sv,v_stage_sv,v_manifest,
    v_checkpoint_manifest_reg,v_checkpoint_regs,'[]'::jsonb,
    'IN_PROGRESS','d11:v31:checkpoint',repeat('5',64),'RESEARCH_WORKER'
  );
  IF NOT EXISTS (SELECT 1 FROM public.orotitan_run_stages
    WHERE run_id=v_run AND stage_code='RESEARCH'
    AND active_manifest_kind='CHECKPOINT' AND handoff_gate_state='NO'
    AND lifecycle_status='IN_PROGRESS')
  THEN RAISE EXCEPTION 'D11 E2E: Research checkpoint not durable'; END IF;
  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id=v_run;
  v_failure := FALSE;
  BEGIN
    PERFORM public.start_orotitan_stage(
      v_run,'DEEP_DIVE',v_run_sv,
      v_pins->'deep_dive_stage'->>'name',
      v_pins->'deep_dive_stage'->>'version',
      v_pins->'deep_dive_stage'->>'content_sha256',
      'd11:v31:checkpoint-block',repeat('6',64),'PILOTAGE');
  EXCEPTION WHEN check_violation THEN v_failure := TRUE;
  END;
  IF NOT v_failure THEN RAISE EXCEPTION 'D11 E2E: CHECKPOINT admitted Deep Dive'; END IF;

  -- Generate the seven *actual* V2 Research contract outputs.
  WITH items AS (
    SELECT artifact_type, ordinal,
      gen_random_uuid() AS artifact_id
    FROM unnest(v_required) WITH ORDINALITY AS t(artifact_type,ordinal)
  ), regs AS (
    SELECT ordinal,
      pg_temp.bpi_attested_registration(
        v_run,'RESEARCH',artifact_id,1,artifact_type,
        lower(artifact_type),
        jsonb_build_object('artifact_type',artifact_type,'ordinal',ordinal)::text,
        'd11/v31/final-'||ordinal||'.json',
        'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE',
        jsonb_build_object('artifact_type',artifact_type,'ordinal',ordinal)
      ) AS reg FROM items
  )
  SELECT jsonb_agg(reg ORDER BY ordinal),
    jsonb_agg(pg_temp.bpi_ref(reg) ORDER BY ordinal)
  INTO v_final_regs,v_final_refs FROM regs;

  v_manifest := jsonb_build_object(
    'manifest_schema_version','1.0.0','manifest_id',v_final_id,
    'manifest_kind','FINAL','run_id',v_run,'stage','RESEARCH','stage_revision',1,
    'issuer_id',v_issuer,'security_id',v_security,'dossier_id',v_dossier,
    'canonical_mode','ANALYZE','run_type','INITIAL','data_cutoff','2026-10-12',
    'baseline_snapshot_id',NULL,'process_version','3.1','pilotage_contract_version','3.1',
    'contract_pins',v_pins,'stage_contract',v_pins->'research_stage',
    'contract_set_sha256',v_contract_hash,
    'input_artifacts','[]'::jsonb,'output_artifacts',v_final_refs,
    'stage_status','COMPLETE','contract_status_code','COMPLETE',
    'handoff_gate',jsonb_build_object('name','READY_FOR_DEEP_DIVE','state','YES'),
    'critical_blockers','[]'::jsonb,'open_material_limitations','[]'::jsonb,
    'parent_manifests',jsonb_build_array(jsonb_build_object(
      'manifest_id',v_checkpoint_id,'version',1,
      'content_sha256',v_checkpoint_manifest_reg->>'content_sha256'
    )),
    'started_at','2026-10-12T12:00:00Z',
    'completed_at','2026-10-12T15:00:00Z'
  );
  v_final_manifest_reg := pg_temp.bpi_attested_registration(
    v_run,'RESEARCH',v_final_id,1,'RESEARCH_STAGE_MANIFEST',
    'research_stage_manifest',v_manifest::text,
    'd11/v31/final-manifest.json',
    'AUTHORITATIVE_STAGE_OUTPUT','AUTHORITATIVE',v_manifest
  );

  -- Refuse bogus output vocabulary in the *actual* finalization transaction.
  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id=v_run;
  SELECT state_version INTO v_stage_sv FROM public.orotitan_run_stages WHERE run_id=v_run AND stage_code='RESEARCH';
  v_false_manifest := jsonb_set(v_manifest,'{output_artifacts}',
    (SELECT jsonb_agg(CASE
      WHEN value->>'artifact_type'='INPUT_SUFFICIENCY_ASSESSMENT'
        THEN jsonb_set(value,'{artifact_type}','"DD_INPUT_SUFFICIENCY_RECORD"'::jsonb)
      WHEN value->>'artifact_type'='CLASSIFICATION_INPUT_RECORD'
        THEN jsonb_set(value,'{artifact_type}','"ANALYSIS_INPUT_LOCK"'::jsonb)
      ELSE value END)
     FROM jsonb_array_elements(v_final_refs))
  );
  -- The V2 contract check fails before the bundle receipt boundary.
  v_false_manifest_reg := v_final_manifest_reg;
  v_failure := FALSE;
  BEGIN
    PERFORM public.finalize_orotitan_stage(
      v_run,'RESEARCH',v_run_sv,v_stage_sv,v_false_manifest,
      v_false_manifest_reg,v_final_regs,'[]'::jsonb,
      'd11:v31:false-v1-labels',repeat('7',64),'RESEARCH_WORKER');
  EXCEPTION WHEN check_violation THEN v_failure := TRUE;
  END;
  IF NOT v_failure THEN RAISE EXCEPTION 'D11 E2E: V1 aliases passed V2 FINAL transaction'; END IF;

  -- Missing classification output likewise cannot finalize.
  v_false_manifest := jsonb_set(v_manifest,'{output_artifacts}',v_final_refs - 6);
  v_false_manifest_reg := v_final_manifest_reg;
  v_failure := FALSE;
  BEGIN
    PERFORM public.finalize_orotitan_stage(
      v_run,'RESEARCH',v_run_sv,v_stage_sv,v_false_manifest,
      v_false_manifest_reg,v_final_regs - 6,'[]'::jsonb,
      'd11:v31:missing-classification',repeat('8',64),'RESEARCH_WORKER');
  EXCEPTION WHEN check_violation THEN v_failure := TRUE;
  END;
  IF NOT v_failure THEN RAISE EXCEPTION 'D11 E2E: missing V2 classification passed FINAL transaction'; END IF;

  -- CAS stale versions must fail without moving state or writing the bundle.
  v_failure := FALSE;
  BEGIN
    PERFORM public.finalize_orotitan_stage(
      v_run,'RESEARCH',v_run_sv-1,v_stage_sv,v_manifest,
      v_final_manifest_reg,v_final_regs,'[]'::jsonb,
      'd11:v31:stale',repeat('9',64),'RESEARCH_WORKER');
  EXCEPTION WHEN serialization_failure THEN v_failure := TRUE;
  END;
  IF NOT v_failure THEN RAISE EXCEPTION 'D11 E2E: stale CAS admitted'; END IF;
  -- Persistence *candidates* and audit attestations are intentionally durable
  -- before finalization. What MUST remain absent after a failed CAS is any
  -- AUTHORITATIVE FINAL promotion or active FINAL stage pointer.
  IF EXISTS (SELECT 1 FROM public.orotitan_artifacts WHERE run_id=v_run
    AND artifact_id=v_final_id AND authority_state='AUTHORITATIVE')
    OR EXISTS (SELECT 1 FROM public.orotitan_run_stages
      WHERE run_id=v_run AND stage_code='RESEARCH'
      AND active_manifest_kind='FINAL') THEN
    RAISE EXCEPTION 'D11 E2E: failed FINAL promoted canonical authority';
  END IF;

  -- Successful persisted finalize, one real transaction, exact seven V2 types.
  v_res := public.finalize_orotitan_stage(
    v_run,'RESEARCH',v_run_sv,v_stage_sv,v_manifest,
    v_final_manifest_reg,v_final_regs,'[]'::jsonb,
    'd11:v31:final',repeat('a',64),'RESEARCH_WORKER'
  );
  IF v_res->>'handoff_gate_state' <> 'YES' OR v_res->>'run_status' <> 'ACTIVE'
    THEN RAISE EXCEPTION 'D11 E2E: wrong finalization response %',v_res;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM public.orotitan_run_stages
    WHERE run_id=v_run AND stage_code='RESEARCH'
      AND lifecycle_status='COMPLETE' AND active_manifest_kind='FINAL'
      AND handoff_gate_state='YES' AND active_manifest_artifact_id=v_final_id
  ) THEN RAISE EXCEPTION 'D11 E2E: Research FINAL not persisted'; END IF;
  IF (SELECT count(*) FROM public.orotitan_artifacts a
      WHERE a.run_id=v_run AND a.stage_code='RESEARCH'
        AND a.manifest_artifact_id=v_final_id
        AND a.authority_class='AUTHORITATIVE_STAGE_OUTPUT') <> 7
  THEN RAISE EXCEPTION 'D11 E2E: genuine V2 final output count not seven'; END IF;
  IF EXISTS (SELECT 1 FROM public.orotitan_artifacts
    WHERE run_id=v_run AND artifact_id=v_checkpoint_id
      AND authority_state <> 'SUPERSEDED')
  THEN RAISE EXCEPTION 'D11 E2E: former CHECKPOINT not superseded'; END IF;

  -- Retry with exact idempotency identity cannot duplicate the finalized bundle.
  v_res := public.finalize_orotitan_stage(
    v_run,'RESEARCH',v_run_sv,v_stage_sv,v_manifest,
    v_final_manifest_reg,v_final_regs,'[]'::jsonb,
    'd11:v31:final',repeat('a',64),'RESEARCH_WORKER');
  IF (v_res->>'idempotent_replay')::boolean IS DISTINCT FROM TRUE
  THEN RAISE EXCEPTION 'D11 E2E: exact finalization replay is not idempotent'; END IF;

  SELECT state_version INTO v_run_sv FROM public.orotitan_runs WHERE run_id=v_run;
  v_res := public.start_orotitan_stage(
    v_run,'DEEP_DIVE',v_run_sv,
    v_pins->'deep_dive_stage'->>'name',
    v_pins->'deep_dive_stage'->>'version',
    v_pins->'deep_dive_stage'->>'content_sha256',
    'd11:v31:dd:start',repeat('b',64),'PILOTAGE');
  IF NOT EXISTS (SELECT 1 FROM public.orotitan_runs
    WHERE run_id=v_run AND current_stage='DEEP_DIVE' AND run_status='ACTIVE')
  OR NOT EXISTS (SELECT 1 FROM public.orotitan_run_stages
    WHERE run_id=v_run AND stage_code='DEEP_DIVE' AND lifecycle_status='IN_PROGRESS')
  THEN RAISE EXCEPTION 'D11 E2E: real Deep Dive start not persisted'; END IF;

  -- No Integration or publication ever appears in this scenario.
  IF EXISTS (SELECT 1 FROM public.orotitan_run_stages
    WHERE run_id=v_run AND stage_code='INTEGRATION')
  THEN RAISE EXCEPTION 'D11 E2E: Integration was spuriously started'; END IF;
  RAISE NOTICE 'D11 V3.1 EXACT PINNING, CHECKPOINT REJECTION, FINAL PERSISTENCE, CAS, IDEMPOTENCY AND DEEP_DIVE START PASSED; run=%',v_run;
END $$;

SELECT 'D11 TRUE Research V3.1 -> Deep Dive/Fundamentals Registry E2E PASS' AS result;
