-- OroTitan Registry V1.12 - valuation-date-alignment methodology successor.
-- Forward-only governance repair. Historical runs, prior Contract Sets and artifacts remain immutable.

begin;

do $$
begin
  if to_regclass('public.orotitan_runs') is null
     or to_regclass('public.orotitan_run_stages') is null
     or to_regclass('public.research_dossiers') is null
     or to_regprocedure('public.orotitan_contract_set_sha256(jsonb)') is null
     or to_regprocedure('public.orotitan_insert_event(uuid,text,text,text,text,text,jsonb)') is null then
    raise exception 'OroTitan Registry prerequisites missing before V1.12';
  end if;
end;
$$;

create or replace function public.create_orotitan_methodology_successor_run(
  p_creation_idempotency_key text,
  p_issuer_id uuid,
  p_entry_path text,
  p_canonical_mode text,
  p_run_type text,
  p_data_cutoff date,
  p_parent_run_id uuid,
  p_baseline_snapshot_id uuid,
  p_process_version text,
  p_pilotage_contract_version text,
  p_contract_pins jsonb,
  p_contract_set_sha256 text,
  p_request_fingerprint_sha256 text,
  p_expected_parent_state_version bigint,
  p_expected_parent_run_status text,
  p_expected_parent_current_stage text,
  p_expected_parent_contract_set_sha256 text,
  p_expected_parent_security_id uuid,
  p_expected_parent_dossier_id uuid
)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $function$
declare
  v_existing public.orotitan_runs%rowtype;
  v_parent public.orotitan_runs%rowtype;
  v_parent_stage public.orotitan_run_stages%rowtype;
  v_current_snapshot_id uuid;
  v_run_id uuid := gen_random_uuid();
  v_event_id uuid;
  v_is_blocked_repair boolean := false;
begin
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;

  select *
    into v_existing
  from public.orotitan_runs
  where creation_idempotency_key = p_creation_idempotency_key;

  if found then
    if v_existing.parent_run_id is distinct from p_parent_run_id
       or v_existing.issuer_id is distinct from p_issuer_id
       or v_existing.data_cutoff is distinct from p_data_cutoff
       or v_existing.run_type is distinct from 'INITIAL'
       or v_existing.baseline_snapshot_id is not null
       or v_existing.contract_set_sha256 is distinct from p_contract_set_sha256
       or not exists (
         select 1
         from public.orotitan_run_events e
         where e.run_id = v_existing.run_id
           and e.event_type = 'RUN_CREATED'
           and e.idempotency_key = p_creation_idempotency_key
           and e.request_fingerprint_sha256 = p_request_fingerprint_sha256
       ) then
      raise exception 'IDEMPOTENCY_CONFLICT: methodology successor creation key reused with different request'
        using errcode = '23514';
    end if;

    return jsonb_build_object(
      'run_id', v_existing.run_id,
      'state_version', v_existing.state_version,
      'run_status', v_existing.run_status,
      'idempotent_replay', true
    );
  end if;

  if p_contract_set_sha256 <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2'
     or public.orotitan_contract_set_sha256(p_contract_pins) <> '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2' then
    raise exception 'SUCCESSOR_ACTIVE_CONTRACT_SET_MISMATCH'
      using errcode = '23514';
  end if;

  if (select count(*) from jsonb_object_keys(p_contract_pins)) <> 15
     or p_contract_pins #>> '{process,name}' <> 'OROTITAN_EXECUTION_PROCESS_V3_1_FREEZE_V3.1'
     or p_contract_pins #>> '{process,version}' <> '3.1'
     or p_contract_pins #>> '{pilotage,name}' <> 'OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V3_1_FREEZE_V3.1'
     or p_contract_pins #>> '{pilotage,version}' <> '3.1'
     or p_contract_pins #>> '{deep_dive_stage,version}' <> '3.1'
     or p_contract_pins #>> '{integration_stage,version}' <> '3.1'
     or p_contract_pins #>> '{dcf_timing,name}' <> 'OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0'
     or p_contract_pins #>> '{dcf_timing,version}' <> '1.0'
     or p_contract_pins #>> '{dcf_timing,content_sha256}' <> '9bec7dcb3af85019806f255a0d504cd1770f50fe9343fb92d8f367cb13b5c4ee'
     or p_contract_pins #>> '{valuation_date_alignment,name}' <> 'OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0'
     or p_contract_pins #>> '{valuation_date_alignment,version}' <> '1.0'
     or p_contract_pins #>> '{valuation_date_alignment,content_sha256}' <> '3c4e315a5b0759d13b99d77b8af5eb0298e81e46fe9ee3706ed08f3947b68ee5'
     or p_process_version <> '3.1'
     or p_pilotage_contract_version <> '3.1' then
    raise exception 'SUCCESSOR_ACTIVE_CONTRACT_SET_MISMATCH: active authority composition is not exact'
      using errcode = '23514';
  end if;

  if p_parent_run_id is null then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND: parent_run_id is required'
      using errcode = '23514';
  end if;
  if p_run_type is distinct from 'INITIAL' then
    raise exception 'SUCCESSOR_RUN_TYPE_INVALID: legal RUN_TYPE is INITIAL'
      using errcode = '23514';
  end if;
  if p_baseline_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: methodology replay must not inherit a baseline snapshot'
      using errcode = '23514';
  end if;
  if p_expected_parent_run_status not in ('ACTIVE','BLOCKED') then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH: expected status must be ACTIVE or exact governed BLOCKED repair'
      using errcode = '23514';
  end if;
  if p_expected_parent_current_stage is distinct from 'DEEP_DIVE' then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH: expected parent stage must be DEEP_DIVE'
      using errcode = '23514';
  end if;

  select *
    into v_parent
  from public.orotitan_runs
  where run_id = p_parent_run_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND' using errcode = 'P0002';
  end if;
  if v_parent.state_version <> p_expected_parent_state_version then
    raise exception 'SUCCESSOR_PARENT_STATE_VERSION_MISMATCH'
      using errcode = '40001';
  end if;
  if v_parent.run_status <> p_expected_parent_run_status then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.current_stage is distinct from p_expected_parent_current_stage then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.published_at is not null
     or v_parent.cancelled_at is not null
     or v_parent.run_status in ('PUBLISHED','CANCELLED','READY_TO_PUBLISH') then
    raise exception 'SUCCESSOR_PARENT_TERMINAL' using errcode = '23514';
  end if;
  if v_parent.data_cutoff <> p_data_cutoff then
    raise exception 'SUCCESSOR_PARENT_CUTOFF_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.contract_set_sha256 <> p_expected_parent_contract_set_sha256 then
    raise exception 'SUCCESSOR_PARENT_CONTRACT_SET_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.contract_set_sha256 = p_contract_set_sha256 then
    raise exception 'SUCCESSOR_PARENT_ALREADY_ON_ACTIVE_CONTRACT_SET' using errcode = '23514';
  end if;
  if v_parent.issuer_id is distinct from p_issuer_id
     or v_parent.security_id is distinct from p_expected_parent_security_id
     or v_parent.dossier_id is distinct from p_expected_parent_dossier_id then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH' using errcode = '23514';
  end if;
  if v_parent.entry_path is distinct from p_entry_path
     or v_parent.canonical_mode is distinct from p_canonical_mode then
    raise exception 'SUCCESSOR_PARENT_ROUTING_MISMATCH' using errcode = '23514';
  end if;

  if v_parent.run_status = 'BLOCKED' then
    if v_parent.contract_set_sha256 <> '257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c' then
      raise exception 'SUCCESSOR_BLOCKED_PARENT_NOT_REPAIRABLE: wrong historical Contract Set'
        using errcode = '23514';
    end if;

    select *
      into v_parent_stage
    from public.orotitan_run_stages
    where run_id = p_parent_run_id
      and stage_code = 'DEEP_DIVE'
    for update;

    if not found
       or v_parent_stage.lifecycle_status <> 'BLOCKED'
       or v_parent_stage.contract_status_code <> 'VALUATION_BLOCKED_PINNED_TIMING_DENOMINATOR_DATE_CONFLICT'
       or v_parent_stage.stage_contract_name <> 'OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V3_FREEZE_V3.0'
       or v_parent_stage.stage_contract_version <> '3.0'
       or v_parent_stage.stage_contract_sha256 <> 'f52932630702d4249d47da370604899c07d7aad9046fbd616945cdfccf2dbd16'
       or v_parent_stage.handoff_gate_state <> 'NOT_EVALUATED'
       or jsonb_typeof(v_parent_stage.blocker_summary) <> 'array'
       or jsonb_array_length(v_parent_stage.blocker_summary) <> 1
       or v_parent_stage.blocker_summary->0->>'code' <> 'V3_VALUATION_TIMING_DENOMINATOR_DATE_CONFLICT'
       or v_parent_stage.blocker_summary->0->>'classification' <> 'PINNED_AUTHORITY_CONFLICT'
       or v_parent_stage.blocker_summary->0->>'scope' <> 'VALUATION_DCF_AND_DEPENDENT_OUTPUTS' then
      raise exception 'SUCCESSOR_BLOCKED_PARENT_NOT_REPAIRABLE: exact timing-conflict state not proven'
        using errcode = '23514';
    end if;

    v_is_blocked_repair := true;
  end if;

  select d.current_snapshot_id
    into v_current_snapshot_id
  from public.research_dossiers d
  where d.dossier_id = p_expected_parent_dossier_id
    and d.issuer_id = p_issuer_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH: dossier not found'
      using errcode = '23514';
  end if;
  if v_current_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: current canonical snapshot now exists'
      using errcode = '23514';
  end if;

  insert into public.orotitan_runs (
    run_id,
    creation_idempotency_key,
    issuer_id,
    security_id,
    dossier_id,
    parent_run_id,
    baseline_snapshot_id,
    entry_path,
    canonical_mode,
    run_type,
    run_status,
    data_cutoff,
    process_version,
    pilotage_contract_version,
    contract_pins,
    contract_set_sha256
  ) values (
    v_run_id,
    p_creation_idempotency_key,
    p_issuer_id,
    p_expected_parent_security_id,
    p_expected_parent_dossier_id,
    p_parent_run_id,
    null,
    p_entry_path,
    p_canonical_mode,
    'INITIAL',
    'CREATED',
    p_data_cutoff,
    p_process_version,
    p_pilotage_contract_version,
    p_contract_pins,
    p_contract_set_sha256
  );

  v_event_id := public.orotitan_insert_event(
    v_run_id,
    null,
    'RUN_CREATED',
    p_creation_idempotency_key,
    p_request_fingerprint_sha256,
    'PILOTAGE',
    jsonb_build_object(
      'run_id', v_run_id,
      'canonical_mode', p_canonical_mode,
      'run_type', 'INITIAL',
      'parent_run_id', p_parent_run_id,
      'creation_reason', 'METHODOLOGY_REPLAY_SUCCESSOR',
      'parent_state_version_cas', p_expected_parent_state_version,
      'parent_status_cas', p_expected_parent_run_status,
      'blocked_methodology_defect_repair', v_is_blocked_repair
    )
  );

  return jsonb_build_object(
    'run_id', v_run_id,
    'state_version', 1,
    'run_status', 'CREATED',
    'event_id', v_event_id,
    'idempotent_replay', false,
    'blocked_methodology_defect_repair', v_is_blocked_repair
  );
end;
$function$;

revoke all on function public.create_orotitan_methodology_successor_run(
  text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text,
  bigint, text, text, text, uuid, uuid
) from public, anon, authenticated;
grant execute on function public.create_orotitan_methodology_successor_run(
  text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text,
  bigint, text, text, text, uuid, uuid
) to service_role;

comment on function public.create_orotitan_methodology_successor_run(
  text, uuid, text, text, text, date, uuid, uuid, text, text, jsonb, text, text,
  bigint, text, text, text, uuid, uuid
) is 'Registry V1.12 controlled same-cutoff methodology replay to Contract Set 3644e501...; transactional parent+Deep-Dive-stage+dossier locking; exact blocked timing-defect recovery; historical parent immutable.';

commit;
