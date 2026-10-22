-- PROPOSED / NOT INSTALLED.
-- A separate production migration authorization is required.
-- Scoped successor for the deployed V1-orotitan_assert_required_outputs function.
-- V1 is unchanged; V2 Research stage-contract pins select the V2 vocabulary.
-- All stage-manifest identity, bytes, version CAS and authority checks remain separate.
CREATE OR REPLACE FUNCTION public.orotitan_assert_required_outputs(
  p_stage_code text,
  p_manifest jsonb
)
RETURNS void
LANGUAGE plpgsql
IMMUTABLE
SET search_path = pg_catalog, public
AS $$
DECLARE
  v_required text[];
  v_missing text;
  v_stage_contract text;
  v_stage_contract_version text;
BEGIN
  IF p_manifest->>'manifest_kind' <> 'FINAL' THEN
    RETURN;
  END IF;
  IF jsonb_typeof(p_manifest->'output_artifacts') IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'MANIFEST_CONTRACT_MISMATCH: output_artifacts must be an array'
      USING errcode = '23514';
  END IF;

  IF p_stage_code = 'RESEARCH' THEN
    v_stage_contract := p_manifest->'stage_contract'->>'name';
    v_stage_contract_version := p_manifest->'stage_contract'->>'version';
    IF v_stage_contract = 'OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0'
       AND v_stage_contract_version = '2.0' THEN
      v_required := ARRAY[
        'RESEARCH_SOURCE_MANIFEST',
        'EVIDENCE_LEDGER',
        'CONFLICT_LEDGER',
        'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER',
        'RESEARCH_GAP_REGISTER',
        'INPUT_SUFFICIENCY_ASSESSMENT',
        'CLASSIFICATION_INPUT_RECORD'
      ];
    ELSIF v_stage_contract = 'OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0'
       AND v_stage_contract_version = '1.0' THEN
      v_required := ARRAY[
        'RESEARCH_SOURCE_MANIFEST',
        'EVIDENCE_LEDGER',
        'CONFLICT_LEDGER',
        'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER',
        'RESEARCH_GAP_REGISTER',
        'DD_INPUT_SUFFICIENCY_RECORD',
        'ANALYSIS_INPUT_LOCK'
      ];
    ELSE
      RAISE EXCEPTION
        'MANIFEST_CONTRACT_MISMATCH: unknown Research stage-contract authority'
        USING errcode = '23514';
    END IF;
  ELSIF p_stage_code = 'DEEP_DIVE' THEN
    v_required := ARRAY[
      'DEEP_DIVE_REPORT', 'EVIDENCE_LEDGER', 'CONFLICT_LEDGER',
      'CALCULATION_LEDGER', 'MATERIAL_ASSUMPTION_REGISTER',
      'ANALYTICAL_BLOCK_OUTPUTS', 'CROSS_BLOCK_RECONCILIATION_RECORD',
      'RED_TEAM_PREMORTEM_RECORD', 'VALUATION_ARTIFACT',
      'CERTIFICATION_ARTIFACT', 'OROTITAN_TERMINAL_GATE_ARTIFACT',
      'READINESS_NEXT_ACTION_ARTIFACT'
    ];
  ELSIF p_stage_code = 'INTEGRATION' THEN
    v_required := ARRAY[
      'CANONICAL_SNAPSHOT_CANDIDATE', 'INTEGRATION_MAPPING_RECORD',
      'SCHEMA_VALIDATION_REPORT', 'I2_RECONCILIATION_REPORT',
      'HISTORY_TRANSITION_VALIDATION_REPORT', 'I3B_ADMISSION_REPORT',
      'PRE_PUBLICATION_CONTROL_CARD'
    ];
  ELSE
    RAISE EXCEPTION 'unsupported stage code %', p_stage_code
      USING errcode = '22023';
  END IF;

  SELECT string_agg(req, ', ' ORDER BY req) INTO v_missing
  FROM unnest(v_required) req
  WHERE NOT EXISTS (
    SELECT 1 FROM jsonb_array_elements(p_manifest->'output_artifacts') ref
    WHERE ref->>'artifact_type' = req
      AND ref->>'authority_class' = 'AUTHORITATIVE_STAGE_OUTPUT'
  );

  IF v_missing IS NOT NULL THEN
    RAISE EXCEPTION
      'MANIFEST_CONTRACT_MISMATCH: missing required authoritative outputs: %', v_missing
      USING errcode = '23514';
  END IF;
END;
$$;
