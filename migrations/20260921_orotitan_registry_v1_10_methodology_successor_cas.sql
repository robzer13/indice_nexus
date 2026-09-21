-- OroTitan Registry V1.10 — transactional CAS for controlled methodology-replay successor creation.
-- Forward-only additive RPC. Does not mutate existing runs or create a successor by itself.

begin;

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
as $$
declare
  v_existing public.orotitan_runs%rowtype;
  v_parent public.orotitan_runs%rowtype;
  v_current_snapshot_id uuid;
begin
  if p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$' then
    raise exception 'invalid request fingerprint' using errcode = '22023';
  end if;

  -- Preserve successful idempotent creation even if the historical parent moves later.
  select * into v_existing
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

  if p_parent_run_id is null then
    raise exception 'SUCCESSOR_PARENT_NOT_FOUND: parent_run_id is required' using errcode = '23514';
  end if;
  if p_run_type is distinct from 'INITIAL' then
    raise exception 'SUCCESSOR_RUN_TYPE_INVALID: legal RUN_TYPE is INITIAL' using errcode = '23514';
  end if;
  if p_baseline_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: methodology replay must not inherit a baseline snapshot'
      using errcode = '23514';
  end if;
  if p_expected_parent_run_status is distinct from 'ACTIVE' then
    raise exception 'SUCCESSOR_PARENT_STATUS_MISMATCH: expected parent status must be ACTIVE'
      using errcode = '23514';
  end if;
  if p_expected_parent_current_stage is distinct from 'DEEP_DIVE' then
    raise exception 'SUCCESSOR_PARENT_STAGE_MISMATCH: expected parent stage must be DEEP_DIVE'
      using errcode = '23514';
  end if;

  select * into v_parent
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
  if v_parent.published_at is not null or v_parent.cancelled_at is not null
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

  -- Lock the dossier row in the same transaction so publication cannot race
  -- a no-baseline methodology replay admission.
  select d.current_snapshot_id
    into v_current_snapshot_id
  from public.research_dossiers d
  where d.dossier_id = p_expected_parent_dossier_id
    and d.issuer_id = p_issuer_id
  for update;

  if not found then
    raise exception 'SUCCESSOR_PARENT_IDENTITY_MISMATCH: dossier not found' using errcode = '23514';
  end if;
  if v_current_snapshot_id is not null then
    raise exception 'SUCCESSOR_BASELINE_MISMATCH: current canonical snapshot now exists'
      using errcode = '23514';
  end if;

  return public.create_orotitan_run(
    p_creation_idempotency_key,
    p_issuer_id,
    p_entry_path,
    p_canonical_mode,
    'INITIAL',
    p_data_cutoff,
    p_parent_run_id,
    null,
    p_process_version,
    p_pilotage_contract_version,
    p_contract_pins,
    p_contract_set_sha256,
    p_request_fingerprint_sha256
  );
end;
$$;

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
) is 'Controlled same-cutoff methodology replay creation with transactional parent + dossier CAS. SUCCESSOR is lineage; legal run_type remains INITIAL.';

commit;
