-- Read-only PostgreSQL verification of Research V2 output compatibility.
-- Run AFTER the validator migration, in an isolated test DB or in SHADOW.
-- No artifacts, Run stages or Registry rows are modified.
DO $tests$
DECLARE
  v2 text[] := ARRAY['RESEARCH_SOURCE_MANIFEST','EVIDENCE_LEDGER','CONFLICT_LEDGER','MATERIAL_RESEARCH_HYPOTHESIS_REGISTER','RESEARCH_GAP_REGISTER','INPUT_SUFFICIENCY_ASSESSMENT','CLASSIFICATION_INPUT_RECORD'];
  v1 text[] := ARRAY['RESEARCH_SOURCE_MANIFEST','EVIDENCE_LEDGER','CONFLICT_LEDGER','MATERIAL_RESEARCH_HYPOTHESIS_REGISTER','RESEARCH_GAP_REGISTER','DD_INPUT_SUFFICIENCY_RECORD','ANALYSIS_INPUT_LOCK'];
  dd text[] := ARRAY['DEEP_DIVE_REPORT','EVIDENCE_LEDGER','CONFLICT_LEDGER','CALCULATION_LEDGER','MATERIAL_ASSUMPTION_REGISTER','ANALYTICAL_BLOCK_OUTPUTS','CROSS_BLOCK_RECONCILIATION_RECORD','RED_TEAM_PREMORTEM_RECORD','VALUATION_ARTIFACT','CERTIFICATION_ARTIFACT','OROTITAN_TERMINAL_GATE_ARTIFACT','READINESS_NEXT_ACTION_ARTIFACT'];
  integ text[] := ARRAY['CANONICAL_SNAPSHOT_CANDIDATE','INTEGRATION_MAPPING_RECORD','SCHEMA_VALIDATION_REPORT','I2_RECONCILIATION_REPORT','HISTORY_TRANSITION_VALIDATION_REPORT','I3B_ADMISSION_REPORT','PRE_PUBLICATION_CONTROL_CARD'];
  m jsonb;
  arr jsonb;
  passed int := 0;
BEGIN
  SELECT jsonb_agg(jsonb_build_object('artifact_type',x,'authority_class','AUTHORITATIVE_STAGE_OUTPUT')) INTO arr FROM unnest(v2) x;
  m:=jsonb_build_object('manifest_kind','FINAL','stage_contract',jsonb_build_object('name','OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0','version','2.0'),'output_artifacts',arr);
  PERFORM public.orotitan_assert_required_outputs('RESEARCH',m); passed:=passed+1;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',m||jsonb_build_object('output_artifacts',arr-0));
    RAISE EXCEPTION 'Incomplete Research V2 outputs unexpectedly accepted';
  EXCEPTION WHEN SQLSTATE '23514' THEN passed:=passed+1; END;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',m||jsonb_build_object('output_artifacts',(SELECT jsonb_agg(jsonb_set(item,'{authority_class}','"ROUTING_ONLY"')) FROM jsonb_array_elements(arr) item)));
    RAISE EXCEPTION 'Research V2 outputs with invalid authority unexpectedly accepted';
  EXCEPTION WHEN SQLSTATE '23514' THEN passed:=passed+1; END;
  SELECT jsonb_agg(jsonb_build_object('artifact_type',x,'authority_class','AUTHORITATIVE_STAGE_OUTPUT')) INTO arr FROM unnest(v1) x;
  PERFORM public.orotitan_assert_required_outputs('RESEARCH',jsonb_build_object('manifest_kind','FINAL','stage_contract',jsonb_build_object('name','LEGACY','version','1.0'),'output_artifacts',arr)); passed:=passed+1;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',m||jsonb_build_object('output_artifacts',arr));
    RAISE EXCEPTION 'Legacy outputs unexpectedly accepted as Research V2';
  EXCEPTION WHEN SQLSTATE '23514' THEN passed:=passed+1; END;
  SELECT jsonb_agg(jsonb_build_object('artifact_type',x,'authority_class','AUTHORITATIVE_STAGE_OUTPUT')) INTO arr FROM unnest(v2) x;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('RESEARCH',jsonb_build_object('manifest_kind','FINAL','stage_contract',jsonb_build_object('name','LEGACY','version','1.0'),'output_artifacts',arr));
    RAISE EXCEPTION 'Research V2 outputs unexpectedly accepted as legacy';
  EXCEPTION WHEN SQLSTATE '23514' THEN passed:=passed+1; END;
  PERFORM public.orotitan_assert_required_outputs('RESEARCH',jsonb_build_object('manifest_kind','CHECKPOINT','output_artifacts','[]'::jsonb)); passed:=passed+1;
  SELECT jsonb_agg(jsonb_build_object('artifact_type',x,'authority_class','AUTHORITATIVE_STAGE_OUTPUT')) INTO arr FROM unnest(dd) x;
  PERFORM public.orotitan_assert_required_outputs('DEEP_DIVE',jsonb_build_object('manifest_kind','FINAL','output_artifacts',arr)); passed:=passed+1;
  SELECT jsonb_agg(jsonb_build_object('artifact_type',x,'authority_class','AUTHORITATIVE_STAGE_OUTPUT')) INTO arr FROM unnest(integ) x;
  PERFORM public.orotitan_assert_required_outputs('INTEGRATION',jsonb_build_object('manifest_kind','FINAL','output_artifacts',arr)); passed:=passed+1;
  BEGIN
    PERFORM public.orotitan_assert_required_outputs('UNKNOWN',jsonb_build_object('manifest_kind','FINAL','output_artifacts',arr));
    RAISE EXCEPTION 'Unsupported stage unexpectedly accepted';
  EXCEPTION WHEN SQLSTATE '22023' THEN passed:=passed+1; END;
  IF passed <> 10 THEN RAISE EXCEPTION 'Regression test failure: %/10',passed; END IF;
  RAISE NOTICE 'Research V2 registry output compatibility: PASS 10/10';
END;
$tests$;
