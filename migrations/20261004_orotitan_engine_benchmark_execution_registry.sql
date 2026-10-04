-- OroTitan Engine Benchmark execution registry.
-- Append-only, non-canonical, service-role RPC write boundary.
-- This table MUST NOT participate in research snapshot publication or dossier pointers.

begin;

create table if not exists public.orotitan_engine_benchmark_executions (
  execution_id uuid primary key default gen_random_uuid(),
  campaign_id text not null,
  phase_id text not null,
  case_id text not null,
  repetition_index integer not null check (repetition_index >= 1),
  execution_profile_id text not null check (execution_profile_id ~ '^[a-f0-9]{64}$'),
  engine_fingerprint text not null check (engine_fingerprint ~ '^[a-f0-9]{64}$'),
  model_provider text not null,
  model_name text not null,
  model_version text not null,
  reasoning_config text not null,
  gateway_provider_only text not null,
  max_output_tokens integer not null check (max_output_tokens >= 1),
  runner_version text not null,
  prompt_artifact_version text not null,
  sanitizer_version text not null,
  input_package_sha256 text not null check (input_package_sha256 ~ '^[a-f0-9]{64}$'),
  started_at timestamptz not null,
  finished_at timestamptz not null,
  duration_ms bigint not null check (duration_ms >= 0),
  dimension_scores jsonb not null,
  dimension_rationales jsonb not null,
  oqs_raw numeric,
  weak_link_cap numeric,
  oqs numeric,
  i2_computation_status text not null check (i2_computation_status in ('PASS', 'NOT_COMPUTABLE')),
  finish_reason text not null,
  usage jsonb,
  output_sha256 text not null check (output_sha256 ~ '^[a-f0-9]{64}$'),
  created_at timestamptz not null default now(),
  constraint orotitan_engine_benchmark_case_id_check check (case_id ~ '^V2REF-[0-9]{3}$'),
  constraint orotitan_engine_benchmark_time_check check (finished_at >= started_at),
  constraint orotitan_engine_benchmark_oqs_bounds check (
    (oqs_raw is null or (oqs_raw >= 0 and oqs_raw <= 100))
    and (weak_link_cap is null or (weak_link_cap >= 0 and weak_link_cap <= 100))
    and (oqs is null or (oqs >= 0 and oqs <= 100))
  ),
  constraint orotitan_engine_benchmark_i2_nullability_check check (
    (i2_computation_status = 'PASS' and oqs_raw is not null and weak_link_cap is not null and oqs is not null)
    or
    (i2_computation_status = 'NOT_COMPUTABLE' and oqs_raw is null and weak_link_cap is null and oqs is null)
  ),
  constraint orotitan_engine_benchmark_unique_execution
    unique (campaign_id, phase_id, case_id, repetition_index, execution_profile_id)
);

create index if not exists orotitan_engine_benchmark_campaign_case_idx
  on public.orotitan_engine_benchmark_executions (campaign_id, phase_id, case_id, repetition_index);

create or replace function public.orotitan_benchmark_dimensions_valid(p_scores jsonb)
returns boolean
language plpgsql
immutable
set search_path = pg_catalog, public
as $$
declare
  v_key text;
  v_value jsonb;
  v_count integer := 0;
  v_allowed constant text[] := array[
    'MOAT',
    'RUNWAY',
    'RETURN_QUALITY',
    'CASH_ECONOMICS',
    'CAPITAL_ALLOCATION',
    'MANAGEMENT_GOVERNANCE',
    'RESILIENCE_RISK'
  ];
begin
  if p_scores is null or jsonb_typeof(p_scores) <> 'object' then
    return false;
  end if;

  for v_key, v_value in select key, value from jsonb_each(p_scores)
  loop
    v_count := v_count + 1;
    if not (v_key = any(v_allowed)) then
      return false;
    end if;

    if jsonb_typeof(v_value) = 'null' then
      continue;
    end if;

    if jsonb_typeof(v_value) <> 'number' then
      return false;
    end if;

    if (v_value #>> '{}')::numeric < 0
       or (v_value #>> '{}')::numeric > 100
       or mod((v_value #>> '{}')::numeric, 5) <> 0 then
      return false;
    end if;
  end loop;

  return v_count = 7
    and not exists (
      select 1
      from unnest(v_allowed) as required_key
      where not (p_scores ? required_key)
    );
end;
$$;

alter table public.orotitan_engine_benchmark_executions
  add constraint orotitan_engine_benchmark_dimensions_check
  check (public.orotitan_benchmark_dimensions_valid(dimension_scores));

create or replace function public.reject_orotitan_benchmark_execution_mutation()
returns trigger
language plpgsql
set search_path = pg_catalog, public
as $$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$$;

drop trigger if exists orotitan_engine_benchmark_executions_immutable
  on public.orotitan_engine_benchmark_executions;
create trigger orotitan_engine_benchmark_executions_immutable
before update or delete on public.orotitan_engine_benchmark_executions
for each row execute function public.reject_orotitan_benchmark_execution_mutation();

alter table public.orotitan_engine_benchmark_executions enable row level security;

revoke all on table public.orotitan_engine_benchmark_executions
  from public, anon, authenticated, service_role;
grant select on table public.orotitan_engine_benchmark_executions to service_role;

create or replace function public.record_orotitan_benchmark_execution(p_execution jsonb)
returns jsonb
language plpgsql
security definer
set search_path = pg_catalog, public
as $$
declare
  v_existing public.orotitan_engine_benchmark_executions%rowtype;
  v_execution_id uuid;
  v_dimension_scores jsonb;
  v_dimension_rationales jsonb;
  v_usage jsonb;
  v_oqs_raw numeric;
  v_weak_link_cap numeric;
  v_oqs numeric;
  v_i2_status text;
begin
  if p_execution is null or jsonb_typeof(p_execution) <> 'object' then
    raise exception 'benchmark execution payload must be a JSON object' using errcode = '22023';
  end if;

  if nullif(p_execution->>'campaign_id', '') is null
     or nullif(p_execution->>'phase_id', '') is null
     or nullif(p_execution->>'case_id', '') is null
     or nullif(p_execution->>'execution_profile_id', '') is null
     or nullif(p_execution->>'engine_fingerprint', '') is null
     or nullif(p_execution->>'model_provider', '') is null
     or nullif(p_execution->>'model_name', '') is null
     or nullif(p_execution->>'model_version', '') is null
     or nullif(p_execution->>'reasoning_config', '') is null
     or nullif(p_execution->>'gateway_provider_only', '') is null
     or nullif(p_execution->>'max_output_tokens', '') is null
     or nullif(p_execution->>'runner_version', '') is null
     or nullif(p_execution->>'prompt_artifact_version', '') is null
     or nullif(p_execution->>'sanitizer_version', '') is null
     or nullif(p_execution->>'input_package_sha256', '') is null
     or nullif(p_execution->>'started_at', '') is null
     or nullif(p_execution->>'finished_at', '') is null
     or nullif(p_execution->>'finish_reason', '') is null
     or nullif(p_execution->>'i2_computation_status', '') is null
     or nullif(p_execution->>'output_sha256', '') is null then
    raise exception 'benchmark execution payload missing required field' using errcode = '22023';
  end if;

  if coalesce((p_execution->>'repetition_index')::integer, 0) < 1 then
    raise exception 'benchmark repetition_index must be >= 1' using errcode = '22023';
  end if;

  v_dimension_scores := p_execution->'dimension_scores';
  v_dimension_rationales := p_execution->'dimension_rationales';
  v_usage := p_execution->'usage';
  v_i2_status := p_execution->>'i2_computation_status';

  if not public.orotitan_benchmark_dimensions_valid(v_dimension_scores) then
    raise exception 'invalid benchmark dimension_scores' using errcode = '22023';
  end if;

  if v_dimension_rationales is null or jsonb_typeof(v_dimension_rationales) <> 'object' then
    raise exception 'dimension_rationales must be an object' using errcode = '22023';
  end if;

  if v_i2_status not in ('PASS', 'NOT_COMPUTABLE') then
    raise exception 'invalid i2_computation_status' using errcode = '22023';
  end if;

  v_oqs_raw := case when p_execution->>'oqs_raw' is null then null else (p_execution->>'oqs_raw')::numeric end;
  v_weak_link_cap := case when p_execution->>'weak_link_cap' is null then null else (p_execution->>'weak_link_cap')::numeric end;
  v_oqs := case when p_execution->>'oqs' is null then null else (p_execution->>'oqs')::numeric end;

  if (v_i2_status = 'PASS' and (v_oqs_raw is null or v_weak_link_cap is null or v_oqs is null))
     or (v_i2_status = 'NOT_COMPUTABLE' and (v_oqs_raw is not null or v_weak_link_cap is not null or v_oqs is not null)) then
    raise exception 'I2 status and OQS nullability disagree' using errcode = '22023';
  end if;

  select *
    into v_existing
    from public.orotitan_engine_benchmark_executions
   where campaign_id = p_execution->>'campaign_id'
     and phase_id = p_execution->>'phase_id'
     and case_id = p_execution->>'case_id'
     and repetition_index = (p_execution->>'repetition_index')::integer
     and execution_profile_id = p_execution->>'execution_profile_id';

  if found then
    if v_existing.output_sha256 = p_execution->>'output_sha256' then
      return jsonb_build_object(
        'status', 'IDEMPOTENT',
        'execution_id', v_existing.execution_id,
        'output_sha256', v_existing.output_sha256
      );
    end if;

    raise exception 'benchmark execution idempotency conflict'
      using errcode = '23505',
            detail = format(
              'existing output_sha256=%s incoming output_sha256=%s',
              v_existing.output_sha256,
              p_execution->>'output_sha256'
            );
  end if;

  insert into public.orotitan_engine_benchmark_executions (
    campaign_id,
    phase_id,
    case_id,
    repetition_index,
    execution_profile_id,
    engine_fingerprint,
    model_provider,
    model_name,
    model_version,
    reasoning_config,
    gateway_provider_only,
    max_output_tokens,
    runner_version,
    prompt_artifact_version,
    sanitizer_version,
    input_package_sha256,
    started_at,
    finished_at,
    duration_ms,
    dimension_scores,
    dimension_rationales,
    oqs_raw,
    weak_link_cap,
    oqs,
    i2_computation_status,
    finish_reason,
    usage,
    output_sha256
  ) values (
    p_execution->>'campaign_id',
    p_execution->>'phase_id',
    p_execution->>'case_id',
    (p_execution->>'repetition_index')::integer,
    p_execution->>'execution_profile_id',
    p_execution->>'engine_fingerprint',
    p_execution->>'model_provider',
    p_execution->>'model_name',
    p_execution->>'model_version',
    p_execution->>'reasoning_config',
    p_execution->>'gateway_provider_only',
    (p_execution->>'max_output_tokens')::integer,
    p_execution->>'runner_version',
    p_execution->>'prompt_artifact_version',
    p_execution->>'sanitizer_version',
    p_execution->>'input_package_sha256',
    (p_execution->>'started_at')::timestamptz,
    (p_execution->>'finished_at')::timestamptz,
    (p_execution->>'duration_ms')::bigint,
    v_dimension_scores,
    v_dimension_rationales,
    v_oqs_raw,
    v_weak_link_cap,
    v_oqs,
    v_i2_status,
    p_execution->>'finish_reason',
    v_usage,
    p_execution->>'output_sha256'
  )
  returning execution_id into v_execution_id;

  return jsonb_build_object(
    'status', 'INSERTED',
    'execution_id', v_execution_id,
    'output_sha256', p_execution->>'output_sha256'
  );
end;
$$;

revoke all on function public.orotitan_benchmark_dimensions_valid(jsonb)
  from public, anon, authenticated, service_role;
revoke all on function public.reject_orotitan_benchmark_execution_mutation()
  from public, anon, authenticated, service_role;
revoke all on function public.record_orotitan_benchmark_execution(jsonb)
  from public, anon, authenticated, service_role;
grant execute on function public.record_orotitan_benchmark_execution(jsonb) to service_role;

commit;
