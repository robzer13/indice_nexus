-- PostgreSQL 17, local throwaway DB only. The proposed function has been
-- installed ONLY in this isolated fixture by its test runner.
DO $$
DECLARE
  v_v1 text[] := ARRAY[
    'RESEARCH_SOURCE_MANIFEST','EVIDENCE_LEDGER','CONFLICT_LEDGER',
    'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER','RESEARCH_GAP_REGISTER',
    'DD_INPUT_SUFFICIENCY_RECORD','ANALYSIS_INPUT_LOCK'
  ];
  v_v2 text[] := ARRAY[
    'RESEARCH_SOURCE_MANIFEST','EVIDENCE_LEDGER','CONFLICT_LEDGER',
    'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER','RESEARCH_GAP_REGISTER',
    'INPUT_SUFFICIENCY_ASSESSMENT','CLASSIFICATION_INPUT_RECORD'
  ];
  v_m1 jsonb;
  v_m2 jsonb;
  v_temp jsonb;
BEGIN
  SELECT jsonb_build_object(
    'manifest_kind','FINAL',
    'stage_contract',jsonb_build_object(
      'name','OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0','version','1.0'),
    'output_artifacts',jsonb_agg(jsonb_build_object(
      'artifact_type',t,'authority_class','AUTHORITATIVE_STAGE_OUTPUT'))
  ) INTO v_m1 FROM unnest(v_v1) AS t;
  SELECT jsonb_build_object(
    'manifest_kind','FINAL',
    'stage_contract',jsonb_build_object(
      'name','OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0','version','2.0'),
    'output_artifacts',jsonb_agg(jsonb_build_object(
      'artifact_type',t,'authority_class','AUTHORITATIVE_STAGE_OUTPUT'))
  ) INTO v_m2 FROM unnest(v_v2) AS t;

  -- Both exact stage authorities keep their own seven real outputs.
  PERFORM public.orotitan_assert_required_outputs('RESEARCH',v_m1);
  PERFORM public.orotitan_assert_required_outputs('RESEARCH',v_m2);

  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',
      jsonb_set(v_m2,'{output_artifacts}',v_m1->'output_artifacts'));
    RAISE EXCEPTION 'D11 negative: V1 aliases incorrectly admitted in V2 FINAL';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',
      jsonb_set(v_m1,'{output_artifacts}',v_m2->'output_artifacts'));
    RAISE EXCEPTION 'D11 negative: V2 aliases incorrectly admitted in V1 FINAL';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    v_temp := jsonb_set(v_m2,'{output_artifacts}',(v_m2->'output_artifacts') - 6);
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',v_temp);
    RAISE EXCEPTION 'D11 negative: missing classification record admitted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    v_temp := jsonb_set(v_m2,'{output_artifacts}','[]'::jsonb);
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',v_temp);
    RAISE EXCEPTION 'D11 negative: missing outputs admitted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    v_temp := jsonb_set(v_m2,'{stage_contract,name}','"FAKE_CONTRACT"'::jsonb);
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',v_temp);
    RAISE EXCEPTION 'D11 negative: unpinned contract admitted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    v_temp := jsonb_set(v_m2,'{output_artifacts,6,authority_class}',
      '"CHECKPOINT_STAGE_OUTPUT"'::jsonb);
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',v_temp);
    RAISE EXCEPTION 'D11 negative: CHECKPOINT output promoted to FINAL';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',
      jsonb_set(v_m2,'{output_artifacts}','{}'::jsonb));
    RAISE EXCEPTION 'D11 negative: malformed FINAL output list accepted';
  EXCEPTION WHEN check_violation THEN NULL;
  END;
  -- CHECKPOINT does not admit a downstream stage: this function must not
  -- reinterpret it as FINAL. Other Registry gates remain authoritative.
  PERFORM public.orotitan_assert_required_outputs('RESEARCH',
    jsonb_set(v_m2,'{manifest_kind}','"CHECKPOINT"'::jsonb));
END $$;
SELECT 'D11 proposed V1/V2 Research FINAL SQL contract negative matrix PASS' AS result;
