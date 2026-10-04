\set ON_ERROR_STOP on

do $$
begin
  if to_regclass('public.orotitan_engine_benchmark_executions') is null then
    raise exception 'benchmark execution table missing';
  end if;

  if not has_table_privilege('service_role', 'public.orotitan_engine_benchmark_executions', 'SELECT') then
    raise exception 'service_role SELECT missing';
  end if;
  if has_table_privilege('service_role', 'public.orotitan_engine_benchmark_executions', 'INSERT')
     or has_table_privilege('service_role', 'public.orotitan_engine_benchmark_executions', 'UPDATE')
     or has_table_privilege('service_role', 'public.orotitan_engine_benchmark_executions', 'DELETE') then
    raise exception 'service_role must not mutate benchmark table directly';
  end if;
  if has_function_privilege('anon', 'public.record_orotitan_benchmark_execution(jsonb)', 'EXECUTE')
     or has_function_privilege('authenticated', 'public.record_orotitan_benchmark_execution(jsonb)', 'EXECUTE')
     or has_function_privilege('public', 'public.record_orotitan_benchmark_execution(jsonb)', 'EXECUTE') then
    raise exception 'benchmark RPC leaked to non-service roles';
  end if;
  if not has_function_privilege('service_role', 'public.record_orotitan_benchmark_execution(jsonb)', 'EXECUTE') then
    raise exception 'service_role benchmark RPC execute missing';
  end if;
end;
$$;

set role service_role;

select public.record_orotitan_benchmark_execution(
  jsonb_build_object(
    'campaign_id', 'OROTITAN_V2_REPRODUCIBILITY_CAMPAIGN_V0.2',
    'phase_id', 'PHASE_A1_SCORING_JUDGMENT_REPLAY',
    'case_id', 'V2REF-010',
    'repetition_index', 1,
    'execution_profile_id', repeat('1', 64),
    'engine_fingerprint', repeat('2', 64),
    'model_provider', 'openai',
    'model_name', 'gpt-test',
    'model_version', 'gpt-test',
    'reasoning_config', 'medium',
    'gateway_provider_only', 'openai',
    'max_output_tokens', 6000,
    'runner_version', 'OROTITAN_V2_SCORING_REPLAY_RUNNER_V0.2',
    'prompt_artifact_version', 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1',
    'sanitizer_version', '1.0.0',
    'input_package_sha256', repeat('3', 64),
    'started_at', '2026-10-04T15:00:00Z',
    'finished_at', '2026-10-04T15:00:02Z',
    'duration_ms', 2000,
    'dimension_scores', jsonb_build_object(
      'MOAT', 85,
      'RUNWAY', 85,
      'RETURN_QUALITY', 80,
      'CASH_ECONOMICS', 80,
      'CAPITAL_ALLOCATION', 75,
      'MANAGEMENT_GOVERNANCE', 80,
      'RESILIENCE_RISK', 80
    ),
    'dimension_rationales', jsonb_build_object(
      'MOAT', 'test',
      'RUNWAY', 'test',
      'RETURN_QUALITY', 'test',
      'CASH_ECONOMICS', 'test',
      'CAPITAL_ALLOCATION', 'test',
      'MANAGEMENT_GOVERNANCE', 'test',
      'RESILIENCE_RISK', 'test'
    ),
    'oqs_raw', 81,
    'weak_link_cap', 100,
    'oqs', 81,
    'i2_computation_status', 'PASS',
    'finish_reason', 'stop',
    'usage', jsonb_build_object('inputTokens', 100, 'outputTokens', 50),
    'output_sha256', repeat('a', 64)
  )
);

select public.record_orotitan_benchmark_execution(
  jsonb_build_object(
    'campaign_id', 'OROTITAN_V2_REPRODUCIBILITY_CAMPAIGN_V0.2',
    'phase_id', 'PHASE_A1_SCORING_JUDGMENT_REPLAY',
    'case_id', 'V2REF-010',
    'repetition_index', 1,
    'execution_profile_id', repeat('1', 64),
    'engine_fingerprint', repeat('2', 64),
    'model_provider', 'openai',
    'model_name', 'gpt-test',
    'model_version', 'gpt-test',
    'reasoning_config', 'medium',
    'gateway_provider_only', 'openai',
    'max_output_tokens', 6000,
    'runner_version', 'OROTITAN_V2_SCORING_REPLAY_RUNNER_V0.2',
    'prompt_artifact_version', 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1',
    'sanitizer_version', '1.0.0',
    'input_package_sha256', repeat('3', 64),
    'started_at', '2026-10-04T15:00:00Z',
    'finished_at', '2026-10-04T15:00:02Z',
    'duration_ms', 2000,
    'dimension_scores', jsonb_build_object(
      'MOAT', 85,
      'RUNWAY', 85,
      'RETURN_QUALITY', 80,
      'CASH_ECONOMICS', 80,
      'CAPITAL_ALLOCATION', 75,
      'MANAGEMENT_GOVERNANCE', 80,
      'RESILIENCE_RISK', 80
    ),
    'dimension_rationales', jsonb_build_object(
      'MOAT', 'test',
      'RUNWAY', 'test',
      'RETURN_QUALITY', 'test',
      'CASH_ECONOMICS', 'test',
      'CAPITAL_ALLOCATION', 'test',
      'MANAGEMENT_GOVERNANCE', 'test',
      'RESILIENCE_RISK', 'test'
    ),
    'oqs_raw', 81,
    'weak_link_cap', 100,
    'oqs', 81,
    'i2_computation_status', 'PASS',
    'finish_reason', 'stop',
    'usage', jsonb_build_object('inputTokens', 100, 'outputTokens', 50),
    'output_sha256', repeat('a', 64)
  )
);

reset role;

do $$
declare
  v_count integer;
begin
  select count(*) into v_count from public.orotitan_engine_benchmark_executions;
  if v_count <> 1 then
    raise exception 'idempotent retry created duplicate row: %', v_count;
  end if;
end;
$$;

do $$
begin
  begin
    perform public.record_orotitan_benchmark_execution(
      jsonb_build_object(
        'campaign_id', 'OROTITAN_V2_REPRODUCIBILITY_CAMPAIGN_V0.2',
        'phase_id', 'PHASE_A1_SCORING_JUDGMENT_REPLAY',
        'case_id', 'V2REF-010',
        'repetition_index', 1,
        'execution_profile_id', repeat('1', 64),
        'engine_fingerprint', repeat('2', 64),
        'model_provider', 'openai',
        'model_name', 'gpt-test',
        'model_version', 'gpt-test',
        'reasoning_config', 'PROVIDER_DEFAULT',
        'prompt_artifact_version', 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1',
        'input_package_sha256', repeat('3', 64),
        'started_at', '2026-10-04T15:00:00Z',
        'finished_at', '2026-10-04T15:00:02Z',
        'duration_ms', 2000,
        'dimension_scores', jsonb_build_object(
          'MOAT', 85,
          'RUNWAY', 85,
          'RETURN_QUALITY', 80,
          'CASH_ECONOMICS', 80,
          'CAPITAL_ALLOCATION', 75,
          'MANAGEMENT_GOVERNANCE', 80,
          'RESILIENCE_RISK', 80
        ),
        'dimension_rationales', '{}'::jsonb,
        'oqs_raw', 81,
        'weak_link_cap', 100,
        'oqs', 81,
        'i2_computation_status', 'PASS',
        'finish_reason', 'stop',
        'usage', '{}'::jsonb,
        'output_sha256', repeat('b', 64)
      )
    );
    raise exception 'divergent idempotency collision was accepted';
  exception
    when unique_violation then
      null;
  end;
end;
$$;

do $$
begin
  begin
    update public.orotitan_engine_benchmark_executions set finish_reason = 'changed';
    raise exception 'benchmark update was accepted';
  exception
    when sqlstate '55000' then
      null;
  end;

  begin
    delete from public.orotitan_engine_benchmark_executions;
    raise exception 'benchmark delete was accepted';
  exception
    when sqlstate '55000' then
      null;
  end;
end;
$$;

set role service_role;
do $$
begin
  begin
    insert into public.orotitan_engine_benchmark_executions (
      campaign_id, phase_id, case_id, repetition_index, execution_profile_id,
      engine_fingerprint, model_provider, model_name, model_version, reasoning_config,
      gateway_provider_only, max_output_tokens, runner_version,
      prompt_artifact_version, sanitizer_version, input_package_sha256,
      started_at, finished_at, duration_ms,
      dimension_scores, dimension_rationales, oqs_raw, weak_link_cap, oqs,
      i2_computation_status, finish_reason, usage, output_sha256
    ) values (
      'X', 'Y', 'V2REF-001', 99, repeat('4',64), repeat('5',64),
      'openai', 'x', 'x', 'x', 'medium', 'openai', 6000,
      'OROTITAN_V2_SCORING_REPLAY_RUNNER_V0.2', 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1', '1.0.0',
      repeat('6',64), now(), now(), 0,
      jsonb_build_object(
        'MOAT', 80, 'RUNWAY', 80, 'RETURN_QUALITY', 80, 'CASH_ECONOMICS', 80,
        'CAPITAL_ALLOCATION', 80, 'MANAGEMENT_GOVERNANCE', 80, 'RESILIENCE_RISK', 80
      ),
      '{}'::jsonb, 80, 100, 80, 'PASS', 'stop', '{}'::jsonb, repeat('c',64)
    );
    raise exception 'direct service_role insert was accepted';
  exception
    when insufficient_privilege then
      null;
  end;
end;
$$;
reset role;

do $$
begin
  begin
    perform public.record_orotitan_benchmark_execution(
      jsonb_build_object(
        'campaign_id', 'OROTITAN_V2_REPRODUCIBILITY_CAMPAIGN_V0.2',
        'phase_id', 'PHASE_A1_SCORING_JUDGMENT_REPLAY',
        'case_id', 'V2REF-011',
        'repetition_index', 1,
        'execution_profile_id', repeat('7', 64),
        'engine_fingerprint', repeat('8', 64),
        'model_provider', 'openai',
        'model_name', 'gpt-test',
        'model_version', 'gpt-test',
        'reasoning_config', 'PROVIDER_DEFAULT',
        'prompt_artifact_version', 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1',
        'input_package_sha256', repeat('9', 64),
        'started_at', '2026-10-04T15:00:00Z',
        'finished_at', '2026-10-04T15:00:02Z',
        'duration_ms', 2000,
        'dimension_scores', jsonb_build_object(
          'MOAT', 83,
          'RUNWAY', 85,
          'RETURN_QUALITY', 80,
          'CASH_ECONOMICS', 80,
          'CAPITAL_ALLOCATION', 75,
          'MANAGEMENT_GOVERNANCE', 80,
          'RESILIENCE_RISK', 80
        ),
        'dimension_rationales', '{}'::jsonb,
        'oqs_raw', 81,
        'weak_link_cap', 100,
        'oqs', 81,
        'i2_computation_status', 'PASS',
        'finish_reason', 'stop',
        'usage', '{}'::jsonb,
        'output_sha256', repeat('d', 64)
      )
    );
    raise exception 'non-5-point dimension score was accepted';
  exception
    when invalid_parameter_value then
      null;
  end;
end;
$$;

select 'benchmark execution registry migration: PASS' as result;
