-- OroTitan Registry — Research V2 required output compatibility.
-- Contract source: robzer13/indice_nexus / Research V2 FREEZE 2.0.
-- Scope: only the Research FINAL required-output names for the exact V2 contract.
-- Legacy Research, Deep Dive and Integration requirements remain unchanged.
-- Safe: modifies only a validator, never a Research Run or artifact.

CREATE OR REPLACE FUNCTION public.orotitan_assert_required_outputs(
  p_stage_code text,
  p_manifest jsonb
)
 RETURNS void
 LANGUAGE plpgsql
 IMMUTABLE
 SET search_path TO 'pg_catalog', 'public'
AS $function$
declare
  v_required text[];
  v_missing text;
begin
  if p_manifest->>'manifest_kind' <> 'FINAL' then
    return;
  end if;

  if p_stage_code = 'RESEARCH' then
    if p_manifest->'stage_contract'->>'name' = 'OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0'
       and p_manifest->'stage_contract'->>'version' = '2.0' then
      -- Outputs contractuels Research V2; nom logique, pas chemin physique.
      v_required := array[
        'RESEARCH_SOURCE_MANIFEST',
        'EVIDENCE_LEDGER',
        'CONFLICT_LEDGER',
        'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER',
        'RESEARCH_GAP_REGISTER',
        'INPUT_SUFFICIENCY_ASSESSMENT',
        'CLASSIFICATION_INPUT_RECORD'
      ];
    else
      -- Comportement historique preserve pour les contrats non V2.
      v_required := array[
        'RESEARCH_SOURCE_MANIFEST',
        'EVIDENCE_LEDGER',
        'CONFLICT_LEDGER',
        'MATERIAL_RESEARCH_HYPOTHESIS_REGISTER',
        'RESEARCH_GAP_REGISTER',
        'DD_INPUT_SUFFICIENCY_RECORD',
        'ANALYSIS_INPUT_LOCK'
      ];
    end if;
  elsif p_stage_code = 'DEEP_DIVE' then
    v_required := array[
      'DEEP_DIVE_REPORT',
      'EVIDENCE_LEDGER',
      'CONFLICT_LEDGER',
      'CALCULATION_LEDGER',
      'MATERIAL_ASSUMPTION_REGISTER',
      'ANALYTICAL_BLOCK_OUTPUTS',
      'CROSS_BLOCK_RECONCILIATION_RECORD',
      'RED_TEAM_PREMORTEM_RECORD',
      'VALUATION_ARTIFACT',
      'CERTIFICATION_ARTIFACT',
      'OROTITAN_TERMINAL_GATE_ARTIFACT',
      'READINESS_NEXT_ACTION_ARTIFACT'
    ];
  elsif p_stage_code = 'INTEGRATION' then
    v_required := array[
      'CANONICAL_SNAPSHOT_CANDIDATE',
      'INTEGRATION_MAPPING_RECORD',
      'SCHEMA_VALIDATION_REPORT',
      'I2_RECONCILIATION_REPORT',
      'HISTORY_TRANSITION_VALIDATION_REPORT',
      'I3B_ADMISSION_REPORT',
      'PRE_PUBLICATION_CONTROL_CARD'
    ];
  else
    raise exception 'unsupported stage code %', p_stage_code using errcode = '22023';
  end if;

  select string_agg(req, ', ' order by req)
  into v_missing
  from unnest(v_required) req
  where not exists (
    select 1
    from jsonb_array_elements(coalesce(p_manifest->'output_artifacts', '[]'::jsonb)) out_ref
    where out_ref->>'artifact_type' = req
      and out_ref->>'authority_class' = 'AUTHORITATIVE_STAGE_OUTPUT'
  );

  if v_missing is not null then
    raise exception 'MANIFEST_CONTRACT_MISMATCH: missing required authoritative outputs: %', v_missing
      using errcode = '23514';
  end if;
end;
$function$;
