-- D29b full-Registry integration: SYNTHETIC Besi-shaped local row only.
-- Requires all real V1.1–V1.12 Registry migrations and D11 V3.1 E2E.
-- This never refers to the production Besi row/metadata or original source bytes.
\set ON_ERROR_STOP on
DO $fixture$
DECLARE
  v_source public.orotitan_runs%ROWTYPE;
  v_source_stage public.orotitan_run_stages%ROWTYPE;
  v_artifact public.orotitan_artifacts%ROWTYPE;
  v_besi CONSTANT uuid := '48b9426b-4d0d-4858-9d87-efcd13ac415d';
  v_artifact_id CONSTANT uuid := '8b8c2086-e295-45ae-8566-e718d8c2ac08';
BEGIN
  SELECT * INTO STRICT v_source FROM public.orotitan_runs
    WHERE creation_idempotency_key = 'd11:v31:create';
  SELECT * INTO STRICT v_source_stage FROM public.orotitan_run_stages
    WHERE run_id=v_source.run_id AND stage_code='RESEARCH';

  IF EXISTS(SELECT 1 FROM public.orotitan_runs WHERE run_id=v_besi) THEN
    RAISE EXCEPTION 'fixture UUID collision; refusing test';
  END IF;

  INSERT INTO public.orotitan_runs SELECT
    (jsonb_populate_record(NULL::public.orotitan_runs,
      to_jsonb(v_source)||jsonb_build_object(
        'run_id',v_besi,'creation_idempotency_key','d29b:synthetic-besi-rpc',
        'run_display_key',NULL,'run_status','ACTIVE','current_stage','RESEARCH',
        'state_version',6,'published_at',NULL,'cancelled_at',NULL
      )
    )).*;

  INSERT INTO public.orotitan_run_stages (
    run_id,stage_code,stage_revision,stage_contract_name,
    stage_contract_version,stage_contract_sha256,lifecycle_status,
    contract_status_code,handoff_gate_name,handoff_gate_state,
    blocker_summary,state_version,started_at
  ) VALUES (
    v_besi,'RESEARCH',v_source_stage.stage_revision,
    v_source_stage.stage_contract_name,v_source_stage.stage_contract_version,
    v_source_stage.stage_contract_sha256,'IN_PROGRESS',
    'BLOCKED_INSUFFICIENT_INPUT','READY_FOR_DEEP_DIVE','NO',
    '[{"code":"C003","state":"OPEN_UNVERIFIED"}]'::jsonb,
    4,clock_timestamp()
  );

  -- Reuse only the STRUCTURE of a D11 test CHECKPOINT artifact,
  -- not a real source or an attested Besi proof. DB FK checks are real.
  SELECT * INTO STRICT v_artifact FROM public.orotitan_artifacts
    WHERE run_id=v_source.run_id AND stage_code='RESEARCH'
      AND authority_class='CHECKPOINT_STAGE_OUTPUT'
      AND artifact_type='RESEARCH_STAGE_MANIFEST'
    ORDER BY version DESC LIMIT 1;

  INSERT INTO public.orotitan_artifacts SELECT
    (jsonb_populate_record(NULL::public.orotitan_artifacts,
      to_jsonb(v_artifact)||jsonb_build_object(
        'artifact_id',v_artifact_id,'run_id',v_besi,
        'manifest_artifact_id',NULL,'manifest_version',NULL,
        'authority_state','CHECKPOINT','authority_class','CHECKPOINT_STAGE_OUTPUT'
      )
    )).*;

  UPDATE public.orotitan_run_stages
  SET active_manifest_artifact_id=v_artifact_id,
      active_manifest_version=v_artifact.version,
      active_manifest_kind='CHECKPOINT'
  WHERE run_id=v_besi AND stage_code='RESEARCH';

  IF NOT EXISTS(SELECT 1 FROM public.orotitan_run_stages s
    JOIN public.orotitan_runs r ON r.run_id=s.run_id
    WHERE s.run_id=v_besi AND r.state_version=6 AND s.state_version=4
      AND s.active_manifest_kind='CHECKPOINT'
      AND s.lifecycle_status='IN_PROGRESS'
      AND s.handoff_gate_state='NO') THEN
    RAISE EXCEPTION 'synthetic Besi-shaped Registry fixture invalid';
  END IF;
END;
$fixture$;
