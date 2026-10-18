-- D7 SANDBOX-ONLY PROTOTYPE. Not a migration and not a V3.1 production writer.
-- Executed ONLY after run-d7-writer-candidate-tests.sh creates a disposable
-- localhost database. The test authority below IS NOT a freeze-approved attester.
\set ON_ERROR_STOP on

begin;
do $d7$
begin
  if current_database() !~ '^orotitan_d7_[0-9]+$'
     or session_user <> 'postgres'
     or inet_server_addr() not in ('127.0.0.1'::inet,'::1'::inet) then
    raise exception 'D7_SANDBOX_ONLY: refusing non-fixture database/session';
  end if;
  if to_regprocedure('public.persist_orotitan_research_snapshot_v2(uuid,uuid,jsonb)') is null then
    raise exception 'D7 requires existing V1/V2 sandbox baseline';
  end if;
end;
$d7$;

-- ONLY in the isolated database, extend the physical contract-version matrix.
-- Existing V1 and V2 rows are unchanged.
alter table public.research_snapshots
  drop constraint research_snapshots_contract_version_check,
  drop constraint research_snapshots_schema_version_check,
  drop constraint research_snapshots_contract_schema_pair_check,
  drop constraint research_snapshots_payload_version_firewall_check;
alter table public.research_snapshots
  add constraint research_snapshots_contract_version_check
    check (contract_version in ('04_SCREENER_SCHEMA_V1','04_SCREENER_SCHEMA_V2','04_SCREENER_SCHEMA_V3_1')),
  add constraint research_snapshots_schema_version_check
    check (schema_version in ('1.0.0','2.0.0','3.1.0')),
  add constraint research_snapshots_contract_schema_pair_check
    check (
      (contract_version='04_SCREENER_SCHEMA_V1' and schema_version='1.0.0')
      or (contract_version='04_SCREENER_SCHEMA_V2' and schema_version='2.0.0')
      or (contract_version='04_SCREENER_SCHEMA_V3_1' and schema_version='3.1.0')
    ),
  add constraint research_snapshots_payload_version_firewall_check
    check (
      (contract_version='04_SCREENER_SCHEMA_V1' and not (canonical_payload ? 'v2_product') and not (canonical_payload ? 'v3_methodology'))
      or (contract_version='04_SCREENER_SCHEMA_V2' and jsonb_typeof(canonical_payload->'v2_product')='object' and not (canonical_payload ? 'v3_methodology'))
      or (contract_version='04_SCREENER_SCHEMA_V3_1' and jsonb_typeof(canonical_payload->'v2_product')='object' and jsonb_typeof(canonical_payload->'v3_methodology')='object')
    );

create schema d7_sandbox authorization postgres;
revoke all on schema d7_sandbox from public;
grant usage on schema d7_sandbox to service_role;

-- Strictly a test fixture: the physical writer reads proofs pre-inserted by the
-- local superuser. The service role has no way to mint or update them.
create table d7_sandbox.approved_inputs (
  candidate_snapshot_id uuid primary key,
  run_id uuid not null,
  dossier_id uuid not null,
  issuer_id uuid not null,
  security_id uuid not null,
  payload_sha256 text not null check (payload_sha256 ~ '^[a-f0-9]{64}$'),
  approval_source text not null check (approval_source = 'D7_TEST_ONLY_APPROVED')
);
revoke all on d7_sandbox.approved_inputs from public, anon, authenticated, service_role;

-- Test harness only. Not named as a production API; does not authorize a run.
create function d7_sandbox.persist_v31_candidate(
  p_run_id uuid,
  p_dossier_id uuid,
  p_expected_current_snapshot_id uuid,
  p_canonical_payload jsonb
) returns jsonb
language plpgsql security definer
set search_path = pg_catalog, public, d7_sandbox
as $function$
declare
  d record;
  previous public.research_snapshots%rowtype;
  id uuid;
  issuer uuid;
  security uuid;
  lock jsonb;
  v3 jsonb;
  denominator jsonb;
  timing jsonb;
  data_cutoff date;
  reference_date date;
  denominator_repr text;
  digest_sha256 text;
begin
  if p_run_id is null or p_dossier_id is null or p_canonical_payload is null
     or jsonb_typeof(p_canonical_payload) is distinct from 'object' then
    raise exception 'D7_PAYLOAD_OR_RUN_MISSING' using errcode = '22023';
  end if;

  -- THIS MUST be a row lock inside the same transaction as the actual pointer.
  select d0.dossier_id,d0.issuer_id,d0.current_snapshot_id,d0.active
    into d from public.research_dossiers d0
    where d0.dossier_id=p_dossier_id for update;
  if not found or d.active is distinct from true then
    raise exception 'D7_DOSSIER_MISSING_OR_INACTIVE' using errcode = '23503';
  end if;

  begin
    id := (p_canonical_payload->>'snapshot_id')::uuid;
    issuer := (p_canonical_payload->>'issuer_id')::uuid;
    security := (p_canonical_payload->>'security_id')::uuid;
    data_cutoff := (p_canonical_payload->'data_lock'->>'data_cutoff')::date;
    reference_date := (p_canonical_payload->'data_lock'->>'reference_price_date')::date;
  exception when invalid_text_representation or datetime_field_overflow then
    raise exception 'D7_LOCK_VALUE_INVALID' using errcode = '22023';
  end;
  if id is null or issuer is distinct from d.issuer_id or security is null then
    raise exception 'D7_IDENTITY_INCORRECT' using errcode = '23514';
  end if;

  lock := p_canonical_payload->'data_lock';
  v3 := p_canonical_payload->'v3_methodology';
  denominator := v3->'economic_share_count';
  timing := v3->'valuation_timing';
  if jsonb_typeof(lock) is distinct from 'object'
     or jsonb_typeof(p_canonical_payload->'v2_product') is distinct from 'object'
     or p_canonical_payload->'v2_product'->'classification'->>'taxonomy_version' is distinct from 'OROTITAN_TAXONOMY_V2.0'
     or jsonb_typeof(denominator) is distinct from 'object'
     or jsonb_typeof(timing) is distinct from 'object'
     or data_cutoff is null or reference_date is null
     or p_canonical_payload->>'execution_mode' not in ('ANALYZE','REFRESH','DISCOVER + ANALYZE') then
    raise exception 'D7_V31_SHAPE_INVALID' using errcode = '23514';
  end if;
  if denominator->>'methodology_version' is distinct from 'OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0'
     or timing->>'methodology_version' is distinct from 'OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0'
     or denominator->>'valuation_admission' is distinct from 'YES'
     or reference_date > data_cutoff
     or timing->>'valuation_date' is distinct from data_cutoff::text
     or timing->>'time_origin' is distinct from data_cutoff::text
     or timing->>'economic_share_count_date' is distinct from data_cutoff::text
     or timing->>'share_count_as_of_date' is distinct from data_cutoff::text
     or timing->>'per_share_output_date' is distinct from data_cutoff::text
     or denominator->>'share_count_as_of_date' is distinct from data_cutoff::text
     or timing->>'reference_price_date' is distinct from reference_date::text then
    raise exception 'D7_DATE_OR_GOVERNANCE_INVALID' using errcode = '23514';
  end if;
  if jsonb_typeof(lock->'reference_price') = 'number'
     and timing->>'market_cap_date' is distinct from reference_date::text then
    raise exception 'D7_MARKETCAP_DATE_INVALID' using errcode = '23514';
  end if;

  denominator_repr := denominator->>'representation';
  if denominator_repr = 'EXACT' then
    if denominator->>'propagation_status' is distinct from 'SCALAR_EXACT'
       or jsonb_typeof(denominator->'exact_count') is distinct from 'number'
       or (denominator->>'exact_count')::numeric <= 0
       or (denominator->>'exact_count')::numeric <> trunc((denominator->>'exact_count')::numeric)
       or denominator ?| array['lower_bound','upper_bound','bound_completeness','unbounded_movement_classes','unknown_reason'] then
      raise exception 'D7_EXACT_DENOMINATOR_INVALID' using errcode='23514';
    end if;
  elsif denominator_repr = 'BOUNDED' then
    if denominator->>'propagation_status' is distinct from 'RANGE_PROPAGATED'
       or denominator->>'bound_completeness' is distinct from 'COMPLETE'
       or denominator->>'unbounded_movement_classes' is distinct from '0'
       or jsonb_typeof(denominator->'lower_bound') is distinct from 'number'
       or jsonb_typeof(denominator->'upper_bound') is distinct from 'number'
       or (denominator->>'lower_bound')::numeric <= 0
       or (denominator->>'lower_bound')::numeric > (denominator->>'upper_bound')::numeric
       or (denominator->>'lower_bound')::numeric <> trunc((denominator->>'lower_bound')::numeric)
       or (denominator->>'upper_bound')::numeric <> trunc((denominator->>'upper_bound')::numeric)
       or denominator ?| array['exact_count','unknown_reason'] then
      raise exception 'D7_BOUNDED_DENOMINATOR_INVALID' using errcode='23514';
    end if;
  else
    raise exception 'D7_UNKNOWN_DENOMINATOR_BLOCKED' using errcode='23514';
  end if;

  -- No self-asserted PASS is consumed. The sandbox authorization is held in a
  -- superuser-managed fixture table; there is no API role INSERT privilege.
  digest_sha256 := encode(public.digest(convert_to(p_canonical_payload::text,'UTF8'),'sha256'),'hex');
  if not exists (
    select 1 from d7_sandbox.approved_inputs a
    where a.candidate_snapshot_id=id and a.run_id=p_run_id
      and a.dossier_id=p_dossier_id and a.issuer_id=issuer and a.security_id=security
      and a.payload_sha256=digest_sha256
      and a.approval_source='D7_TEST_ONLY_APPROVED'
  ) then
    raise exception 'D7_TRUSTED_ADMISSION_MISSING_OR_STALE' using errcode='23514';
  end if;

  select * into previous from public.research_snapshots where snapshot_id=id;
  if found then
    if previous.dossier_id=p_dossier_id and previous.issuer_id=issuer
       and previous.security_id=security and previous.contract_version='04_SCREENER_SCHEMA_V3_1'
       and previous.schema_version='3.1.0' and previous.canonical_payload=p_canonical_payload
       and d.current_snapshot_id=id then
      return jsonb_build_object('status','IDEMPOTENT_SANDBOX','snapshot_id',id,'current_snapshot_id',d.current_snapshot_id);
    end if;
    raise exception 'D7_SNAPSHOT_ID_CONFLICT' using errcode='23505';
  end if;

  if d.current_snapshot_id is distinct from p_expected_current_snapshot_id then
    raise exception 'D7_STALE_POINTER' using errcode='40001';
  end if;

  insert into public.research_snapshots(
    snapshot_id,dossier_id,issuer_id,security_id,report_id,execution_mode,
    data_cutoff,calculation_date,report_version,method_version,calculation_version,
    evidence_ledger_version,contract_version,schema_version,canonical_payload
  ) values (
    id,p_dossier_id,issuer,security,p_canonical_payload->>'report_id',
    p_canonical_payload->>'execution_mode',data_cutoff,
    (p_canonical_payload->'data_lock'->>'calculation_date')::date,
    p_canonical_payload->'versions'->>'report_version',
    p_canonical_payload->'versions'->>'method_version',
    p_canonical_payload->'versions'->>'calculation_version',
    p_canonical_payload->'versions'->>'evidence_ledger_version',
    '04_SCREENER_SCHEMA_V3_1','3.1.0',p_canonical_payload
  );
  update public.research_dossiers set current_snapshot_id=id where dossier_id=p_dossier_id;
  if not found then
    raise exception 'D7_POINTER_UPDATE_FAILED' using errcode='23503';
  end if;

  return jsonb_build_object('status','INSERTED_SANDBOX','snapshot_id',id,'current_snapshot_id',id);
end;
$function$;

revoke all on function d7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)
  from public, anon, authenticated;
grant execute on function d7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)
  to service_role;

comment on function d7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb) is
  'TEST-ONLY D7 transactional prototype; NEVER install in production or mark V3.1 admission approved';
commit;
