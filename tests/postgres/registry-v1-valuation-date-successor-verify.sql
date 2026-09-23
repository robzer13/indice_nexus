\set ON_ERROR_STOP on

do $verify$
declare
  v_fn regprocedure := 'public.create_orotitan_methodology_successor_run(text,uuid,text,text,text,date,uuid,uuid,text,text,jsonb,text,text,bigint,text,text,text,uuid,uuid)'::regprocedure;
begin
  if not (select prosecdef from pg_proc where oid=v_fn) then
    raise exception 'V1.12 successor RPC must be SECURITY DEFINER';
  end if;
  if not ((select proconfig from pg_proc where oid=v_fn) @> array['search_path=pg_catalog, public']::text[]) then
    raise exception 'V1.12 successor RPC search_path guard missing';
  end if;
  if has_function_privilege('public',v_fn,'EXECUTE')
     or has_function_privilege('anon',v_fn,'EXECUTE')
     or has_function_privilege('authenticated',v_fn,'EXECUTE')
     or not has_function_privilege('service_role',v_fn,'EXECUTE') then
    raise exception 'V1.12 successor RPC privilege boundary invalid';
  end if;
end;
$verify$;

do $verify$
declare
  v_parent_id uuid := '32000000-0000-4000-8000-000000000001';
  v_identity record;
  v_old_pins jsonb := '{"process":{"name":"OROTITAN_EXECUTION_PROCESS_V3_FREEZE_V3.0","version":"3.0","content_sha256":"8bf1817d4a3d3655b386e452b54eb50be1025ce56d4933bd1cbb5f0ef8c04dd7","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_EXECUTION_PROCESS_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"11e3ae0acd6fa0645572e778aef6e51fcf70ddc6"}},"pilotage":{"name":"OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_FREEZE_V3.0","version":"3.0","content_sha256":"724d74ec82c5908f4d5d3c8a8062e64bfa19ced9abadee634ffd353e2ad21fa9","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"9aaa2a2f87af1e00d7880ee97926db4662afcbcb"}},"research_stage":{"name":"OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0","version":"2.0","content_sha256":"cdd62f087ff8f8dc97f0c16586d634001f2cef1c44f9e4540e087778cdaa7a01","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md","commit_sha":"86b227a75275cf4aaec6ef61ea2a27e87e2bfec7","blob_sha":"4b49d4ab670717d844ff0ed3c925a2004c043dfa"}},"deep_dive_stage":{"name":"OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0","version":"3.0","content_sha256":"f52932630702d4249d47da370604899c07d7aad9046fbd616945cdfccf2dbd16","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"0cbf3c8ecc08b6b947152345f7a1532163d993c1"}},"integration_stage":{"name":"OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_FREEZE_V3.0","version":"3.0","content_sha256":"09534d88e3ac4d37cfa0f55b7e762692cd198b7baff424ed169396bdd6851759","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_FREEZE_V3.0.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"4ca5214238b1ff08a4da12dec8a89e942ac905a0"}},"analysis_standard":{"name":"OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"538b70a975d00b8bfc9c3abf705d5efd425852516680d499db13d8c2ac17174f","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/01_ANALYSIS_STANDARD_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"bfd034c6a650bb2bd0db1ad3cb3d4476aa9ca66e"}},"master_prompt":{"name":"02_OROTITAN_MASTER_PROMPT_V1_1_ECONOMIC_SHARE_COUNT_PATCH","version":"1.1","content_sha256":"2447ad836d9507eb82db884b1bdb1e0062d1512526d12afa4593339ce51dd0df","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/02_OROTITAN_MASTER_PROMPT_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"b04ad374b79ed958a29dcb5b3d9943d394ece190"}},"investment_policy":{"name":"OROTITAN_INVESTMENT_POLICY_V1.0.0","version":"1.0.0","content_sha256":"66cc29ccaccf4ccf65f58c2959bf95bcf676e67567aa1dd57863b2cb9a2936c8","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_INVESTMENT_POLICY_V1.0.0.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"fd0121bbb14e35370ddd700d9e9845eaf1ec9f75","locator_format":"RAW_CANONICAL_V1"}},"execution_patch":{"name":"OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1","version":"1.0.1","content_sha256":"6e45f39c03e79c34911850c84904b36f3d95266288a5528127a59d0b8d675973","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"449bb43b664eae9d8c9fd98440e5511c6fbf4be3","locator_format":"RAW_CANONICAL_V1"}},"integration_spec":{"name":"04_INTEGRATION_SPEC_V3","version":"3.0","content_sha256":"2ab3923f18973cad469510c13906a0641275f2d112c43cbdf7a9047975395e39","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_INTEGRATION_SPEC_V3.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"45bb894b3081d4cdbe464fb668d8176752085931"}},"screener_schema":{"name":"04_SCREENER_SCHEMA_V3","version":"3.0.0","content_sha256":"372a4e162b8f5d510661ae6ffadb6347c28fedce62779cf2ac08da3425ae4890","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_SCREENER_SCHEMA_V3.json","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"a53e8506bfd4dd348c8ffd46282e46da341aa0ca"}},"i2":{"name":"I2_CANONICAL_COMPUTATION_V1.1","version":"1.1","content_sha256":"ebf4d0e23784aaf67fb48f1bd372ebe1bd8efdbef9dafbeb8ecee6b9f4082b00","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I2_CANONICAL_COMPUTATION_V1.1.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"e145448c1f1d901527a6fe9b62309a0f4159a25a"}},"i3b":{"name":"I3B_VALIDATED_SNAPSHOT_WRITER_V1.1","version":"1.1","content_sha256":"fbb803f5ee26d056bcaf9b1e67ee2c2d4b59bbc5ff637ceceee667e053eacd60","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER_V1.1.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"6fe5b719d83e21b50568d007a4fba9df4e5fd0e0"}},"dcf_timing":{"name":"OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"9bec7dcb3af85019806f255a0d504cd1770f50fe9343fb92d8f367cb13b5c4ee","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0.md","commit_sha":"1f63baf3c5ae03e5e0a0f90a2794a0f358398c5c","blob_sha":"4316d8e574151882ae25f83c3f7b97f8bad48cd3"}}}'::jsonb;
  v_new_pins jsonb := '{"process":{"name":"OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"7e074c26f3bb0f1949e9e7dc0e8c9c3404b0e751561a2e9e8c4f72e0408861ee","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"d1419e097e19b49c119608a316d94d48a318d29f"}},"pilotage":{"name":"OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"050a1a4c6b3662cc19dd78357a4ba556c61aaa227f95c74475d66f388172709a","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"49a346bfeccef51f798b648c823df4db1e81b7eb"}},"research_stage":{"name":"OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0","version":"2.0","content_sha256":"cdd62f087ff8f8dc97f0c16586d634001f2cef1c44f9e4540e087778cdaa7a01","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md","commit_sha":"86b227a75275cf4aaec6ef61ea2a27e87e2bfec7","blob_sha":"4b49d4ab670717d844ff0ed3c925a2004c043dfa"}},"deep_dive_stage":{"name":"OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"2a5cd0a7a7004212e9b3942f129b63d1743710023148c25de4d99b75e4c55ab9","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"2cc80f42a37930044465b6a8112b4ba1b9b4d7b2"}},"integration_stage":{"name":"OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_1_FREEZE_V3.1","version":"3.1","content_sha256":"426aa192b02372f9081f383d28323482ba8878077d25a0ff579c04dadaf3d05a","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_1_FREEZE_V3.1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"13e0b9a6953d6502989a74111c57e86d73f8b40d"}},"analysis_standard":{"name":"OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"538b70a975d00b8bfc9c3abf705d5efd425852516680d499db13d8c2ac17174f","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/01_ANALYSIS_STANDARD_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"bfd034c6a650bb2bd0db1ad3cb3d4476aa9ca66e"}},"master_prompt":{"name":"02_OROTITAN_MASTER_PROMPT_V1_2_VALUATION_DATE_ALIGNMENT_PATCH","version":"1.2","content_sha256":"b6962e07ffc36b33ac08984aab0edf3abac58deb4f597d37c495a461e5dabc5a","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/02_OROTITAN_MASTER_PROMPT_V1_2_VALUATION_DATE_ALIGNMENT_PATCH.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"48c5ba2ff86bb958126dd61d5ac3a0eb63a5d7e8"}},"investment_policy":{"name":"OROTITAN_INVESTMENT_POLICY_V1.0.0","version":"1.0.0","content_sha256":"66cc29ccaccf4ccf65f58c2959bf95bcf676e67567aa1dd57863b2cb9a2936c8","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_INVESTMENT_POLICY_V1.0.0.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"fd0121bbb14e35370ddd700d9e9845eaf1ec9f75","locator_format":"RAW_CANONICAL_V1"}},"execution_patch":{"name":"OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1","version":"1.0.1","content_sha256":"6e45f39c03e79c34911850c84904b36f3d95266288a5528127a59d0b8d675973","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"449bb43b664eae9d8c9fd98440e5511c6fbf4be3","locator_format":"RAW_CANONICAL_V1"}},"integration_spec":{"name":"04_INTEGRATION_SPEC_V3.1","version":"3.1","content_sha256":"8a912b821769d30db1e6953b5b8e47a79c5bdb90490d0f3d9ff2f570640d41a2","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_INTEGRATION_SPEC_V3_1.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"8099d1c7f4544e4902e48cbd2491671fafc20e9f"}},"screener_schema":{"name":"04_SCREENER_SCHEMA_V3_1","version":"3.1.0","content_sha256":"8d27e947b5d0989ed43f1600bf03f2afae00c7b211bbfe008f13fe6cd6c4c64d","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/04_SCREENER_SCHEMA_V3_1.json","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"be68c1dec97c7e0a0967241152858c65298dfa43"}},"i2":{"name":"I2_CANONICAL_COMPUTATION_V1.1","version":"1.1","content_sha256":"ebf4d0e23784aaf67fb48f1bd372ebe1bd8efdbef9dafbeb8ecee6b9f4082b00","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I2_CANONICAL_COMPUTATION_V1.1.md","commit_sha":"26e3b55cc933990ad5fd5e0a847b537cc5ddb739","blob_sha":"e145448c1f1d901527a6fe9b62309a0f4159a25a"}},"i3b":{"name":"I3B_VALIDATED_SNAPSHOT_WRITER_V1.2","version":"1.2","content_sha256":"b98c3e22e52800eb387c452c28b3051510472ecaa11697fff912865275847557","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER_V1.2.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"1183054128cf276abc626c770da58d39accba49e"}},"dcf_timing":{"name":"OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"9bec7dcb3af85019806f255a0d504cd1770f50fe9343fb92d8f367cb13b5c4ee","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0.md","commit_sha":"1f63baf3c5ae03e5e0a0f90a2794a0f358398c5c","blob_sha":"4316d8e574151882ae25f83c3f7b97f8bad48cd3"}},"valuation_date_alignment":{"name":"OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0","version":"1.0","content_sha256":"3c4e315a5b0759d13b99d77b8af5eb0298e81e46fe9ee3706ed08f3947b68ee5","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v3/OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0.md","commit_sha":"63643bdea38683f14a450ad4c4d24de56f9d5183","blob_sha":"14fe1dcfe413475690e945a78131753dda9df419"}}}'::jsonb;
  v_old_hash text;
  v_new_hash text;
  v_before jsonb;
  v_after jsonb;
  v_result jsonb;
  v_child public.orotitan_runs%rowtype;
  v_failed boolean;
begin
  select d.issuer_id, s.security_id, d.dossier_id
    into v_identity
  from public.research_dossiers d
  join public.securities s on s.issuer_id=d.issuer_id
  where d.current_snapshot_id is null
  order by d.created_at, s.created_at
  limit 1;

  if not found then
    raise exception 'V1.12 fixture requires one dossier without current snapshot';
  end if;

  v_old_hash := public.orotitan_contract_set_sha256(v_old_pins);
  v_new_hash := public.orotitan_contract_set_sha256(v_new_pins);

  if v_old_hash <> '257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c' then
    raise exception 'V1.12 historical Contract Set fixture mismatch: %',v_old_hash;
  end if;
  if v_new_hash <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2' then
    raise exception 'V1.12 successor Contract Set fingerprint mismatch: %',v_new_hash;
  end if;

  insert into public.orotitan_runs(
    run_id,creation_idempotency_key,
    issuer_id,security_id,dossier_id,
    parent_run_id,baseline_snapshot_id,
    entry_path,canonical_mode,run_type,
    run_status,current_stage,data_cutoff,
    process_version,pilotage_contract_version,
    contract_pins,contract_set_sha256,state_version
  ) values (
    v_parent_id,'test:v1.12:blocked-parent',
    v_identity.issuer_id,v_identity.security_id,v_identity.dossier_id,
    null,null,
    'IMPOSED_COMPANY','ANALYZE','INITIAL',
    'BLOCKED','DEEP_DIVE',date '2026-09-22',
    '3.0','3.0',
    v_old_pins,v_old_hash,8
  );

  insert into public.orotitan_run_stages(
    run_id,stage_code,stage_revision,
    stage_contract_name,stage_contract_version,stage_contract_sha256,
    lifecycle_status,contract_status_code,
    handoff_gate_name,handoff_gate_state,
    blocker_summary,state_version,started_at
  ) values (
    v_parent_id,'DEEP_DIVE',1,
    'OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0','3.0',
    'f52932630702d4249d47da370604899c07d7aad9046fbd616945cdfccf2dbd16',
    'BLOCKED','VALUATION_BLOCKED_PINNED_TIMING_DENOMINATOR_DATE_CONFLICT',
    'READY_FOR_INTEGRATION','NOT_EVALUATED',
    '[{"code":"V3_VALUATION_TIMING_DENOMINATOR_DATE_CONFLICT","classification":"PINNED_AUTHORITY_CONFLICT","scope":"VALUATION_DCF_AND_DEPENDENT_OUTPUTS"}]'::jsonb,
    4,clock_timestamp()
  );

  select to_jsonb(r) into v_before
  from public.orotitan_runs r
  where r.run_id=v_parent_id;

  v_result := public.create_orotitan_methodology_successor_run(
    'test:v1.12:successor:blocked:ok',
    v_identity.issuer_id,
    'IMPOSED_COMPANY',
    'ANALYZE',
    'INITIAL',
    date '2026-09-22',
    v_parent_id,
    null,
    '3.1',
    '3.1',
    v_new_pins,
    v_new_hash,
    repeat('a',64),
    8,
    'BLOCKED',
    'DEEP_DIVE',
    v_old_hash,
    v_identity.security_id,
    v_identity.dossier_id
  );

  if coalesce((v_result->>'blocked_methodology_defect_repair')::boolean,false) is not true then
    raise exception 'V1.12 exact BLOCKED defect route not identified';
  end if;

  select * into v_child
  from public.orotitan_runs
  where run_id=(v_result->>'run_id')::uuid;

  if v_child.parent_run_id <> v_parent_id
     or v_child.run_type <> 'INITIAL'
     or v_child.baseline_snapshot_id is not null
     or v_child.data_cutoff <> date '2026-09-22'
     or v_child.issuer_id <> v_identity.issuer_id
     or v_child.security_id <> v_identity.security_id
     or v_child.dossier_id <> v_identity.dossier_id
     or v_child.contract_set_sha256 <> v_new_hash
     or v_child.run_status <> 'CREATED' then
    raise exception 'V1.12 blocked successor child persistence mismatch';
  end if;

  select to_jsonb(r) into v_after
  from public.orotitan_runs r
  where r.run_id=v_parent_id;

  if v_after is distinct from v_before then
    raise exception 'V1.12 successor mutated historical parent';
  end if;

  if (public.create_orotitan_methodology_successor_run(
    'test:v1.12:successor:blocked:ok',
    v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL',date '2026-09-22',
    v_parent_id,null,'3.1','3.1',v_new_pins,v_new_hash,repeat('a',64),
    8,'BLOCKED','DEEP_DIVE',v_old_hash,v_identity.security_id,v_identity.dossier_id
  )->>'run_id')::uuid <> v_child.run_id then
    raise exception 'V1.12 idempotent replay returned different successor';
  end if;

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:v1.12:successor:stale-cas',
      v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL',date '2026-09-22',
      v_parent_id,null,'3.1','3.1',v_new_pins,v_new_hash,repeat('b',64),
      7,'BLOCKED','DEEP_DIVE',v_old_hash,v_identity.security_id,v_identity.dossier_id
    );
  exception when serialization_failure then
    if position('SUCCESSOR_PARENT_STATE_VERSION_MISMATCH' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'V1.12 stale parent CAS was admitted';
  end if;

  update public.orotitan_run_stages
  set blocker_summary='[{"code":"WRONG_BLOCKER","classification":"PINNED_AUTHORITY_CONFLICT","scope":"VALUATION_DCF_AND_DEPENDENT_OUTPUTS"}]'::jsonb
  where run_id=v_parent_id and stage_code='DEEP_DIVE';

  v_failed := false;
  begin
    perform public.create_orotitan_methodology_successor_run(
      'test:v1.12:successor:wrong-blocker',
      v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL',date '2026-09-22',
      v_parent_id,null,'3.1','3.1',v_new_pins,v_new_hash,repeat('c',64),
      8,'BLOCKED','DEEP_DIVE',v_old_hash,v_identity.security_id,v_identity.dossier_id
    );
  exception when check_violation then
    if position('SUCCESSOR_BLOCKED_PARENT_NOT_REPAIRABLE' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'V1.12 wrong BLOCKED defect was admitted';
  end if;

  v_failed := false;
  begin
    perform public.create_orotitan_run(
      'test:v1.12:generic-parent-bypass',
      v_identity.issuer_id,'IMPOSED_COMPANY','ANALYZE','INITIAL',date '2026-09-22',
      v_parent_id,null,'3.1','3.1',v_new_pins,v_new_hash,repeat('d',64)
    );
  exception when check_violation then
    if position('PARENT_LINK_REQUIRES_CONTROLLED_SUCCESSOR_RPC' in sqlerrm)>0 then
      v_failed := true;
    else
      raise;
    end if;
  end;
  if not v_failed then
    raise exception 'V1.12 generic parent-link bypass was admitted';
  end if;
end;
$verify$;

select 'Registry V1.12 valuation-date successor: PASS' as result;
