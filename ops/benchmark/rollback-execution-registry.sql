-- OPERATOR-ONLY, NEVER auto-apply. No Supabase-history updates.
-- Forward source pinned: migrations/20261004_orotitan_engine_benchmark_execution_registry.sql
-- Git commit a10e2e61fa9815c865762e84437296f08c08b990.
-- Requires explicit operator GO, verified recovery and quiesced benchmark runner.
\set ON_ERROR_STOP on
begin;
set local lock_timeout='2s';
lock table public.orotitan_engine_benchmark_executions,
           public.orotitan_engine_benchmark_claims in access exclusive mode nowait;
-- Independent transactional, session-local constraint baselines from pinned forward DDL.
-- Same unqualified table names intentionally reproduce PostgreSQL-generated constraint names.
-- ON COMMIT DROP ensures no durable reference objects are created.
create temporary table orotitan_engine_benchmark_executions (
  execution_id uuid primary key default gen_random_uuid(),
  campaign_id text not null,
  phase_id text not null,
  case_id text not null,
  repetition_index integer not null check (repetition_index >= 1),
  execution_profile_id text not null check (execution_profile_id ~ '^[a-f0-9]{64}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 -- RESTRICT does not track PL/pgSQL textual/dynamic-SQL references.
 -- Conservatively refuse identifiable external routine-body references,
 -- including references to the pinned helper functions.
 if exists (
   select 1 from pg_proc p
   where p.oid not in (
     'public.orotitan_benchmark_dimensions_valid(jsonb)'::regprocedure,
     'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure,
     'public.record_orotitan_benchmark_execution(jsonb)'::regprocedure,
     'public.claim_orotitan_benchmark_execution(text,text,text,integer,text)'::regprocedure
   )
   and (
     lower(coalesce(p.prosrc,'')) like '%orotitan_engine_benchmark_%'
     or lower(coalesce(p.prosrc,'')) like '%orotitan_benchmark_dimensions_valid%'
     or lower(coalesce(p.prosrc,'')) like '%reject_orotitan_benchmark_execution_mutation%'
     or lower(coalesce(p.prosrc,'')) like '%record_orotitan_benchmark_execution%'
     or lower(coalesce(p.prosrc,'')) like '%claim_orotitan_benchmark_execution%'
   )
 ) then
   raise exception 'ROLLBACK_REFUSED_EXTERNAL_ROUTINE_REFERENCE' using errcode='55000';
 end if;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
  engine_fingerprint text not null check (engine_fingerprint ~ '^[a-f0-9]{64}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
  model_provider text not null,
  model_name text not null,
  model_version text not null,
  reasoning_config text not null,
  gateway_provider_only text not null,
  max_output_tokens integer not null check (max_output_tokens >= 1),
  runner_version text not null,
  prompt_artifact_version text not null,
  sanitizer_version text not null,
  input_package_sha256 text not null check (input_package_sha256 ~ '^[a-f0-9]{64}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
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
  output_sha256 text not null check (output_sha256 ~ '^[a-f0-9]{64}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
  created_at timestamptz not null default now(),
  constraint orotitan_engine_benchmark_case_id_check check (case_id ~ '^V2REF-[0-9]{3}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
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
) on commit drop;

create temporary table orotitan_engine_benchmark_claims (
  campaign_id text not null,
  phase_id text not null,
  case_id text not null check (case_id ~ '^V2REF-[0-9]{3}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
  repetition_index integer not null check (repetition_index >= 1),
  execution_profile_id text not null check (execution_profile_id ~ '^[a-f0-9]{64}do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
),
  claimed_at timestamptz not null default now(),
  primary key (campaign_id, phase_id, case_id, repetition_index, execution_profile_id)
) on commit drop;

alter table pg_temp.orotitan_engine_benchmark_executions
  add constraint orotitan_engine_benchmark_dimensions_check
  check (public.orotitan_benchmark_dimensions_valid(dimension_scores));
do $rollback_guard$
declare
 e oid := 'public.orotitan_engine_benchmark_executions'::regclass;
 c oid := 'public.orotitan_engine_benchmark_claims'::regclass;
 owner_id oid;
 actual jsonb;
 expected record;
begin
 select relowner into owner_id from pg_class where oid=e;
 if pg_get_userbyid(owner_id)<>current_user or
   (select pg_get_userbyid(relowner) from pg_class where oid=c)<>current_user then
   raise exception 'ROLLBACK_REFUSED_OWNER' using errcode='55000';
 end if;
 if not exists(select 1 from pg_class where oid=e and relkind='r' and relrowsecurity and not relforcerowsecurity)
 or not exists(select 1 from pg_class where oid=c and relkind='r' and relrowsecurity and not relforcerowsecurity) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: table kind or RLS' using errcode='55000';
 end if;
 if exists(select 1 from public.orotitan_engine_benchmark_executions)
 or exists(select 1 from public.orotitan_engine_benchmark_claims) then
   raise exception 'ROLLBACK_REFUSED_DATA: nonempty benchmark registry' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=e and attnum>0 and not attisdropped;
 if actual is distinct from '[["execution_id","uuid",true],["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["engine_fingerprint","text",true],["model_provider","text",true],["model_name","text",true],["model_version","text",true],["reasoning_config","text",true],["gateway_provider_only","text",true],["max_output_tokens","integer",true],["runner_version","text",true],["prompt_artifact_version","text",true],["sanitizer_version","text",true],["input_package_sha256","text",true],["started_at","timestamp with time zone",true],["finished_at","timestamp with time zone",true],["duration_ms","bigint",true],["dimension_scores","jsonb",true],["dimension_rationales","jsonb",true],["oqs_raw","numeric",false],["weak_link_cap","numeric",false],["oqs","numeric",false],["i2_computation_status","text",true],["finish_reason","text",true],["usage","jsonb",false],["output_sha256","text",true],["created_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: execution columns' using errcode='55000';
 end if;
 select jsonb_agg(jsonb_build_array(attname,format_type(atttypid,atttypmod),attnotnull) order by attnum)
 into actual from pg_attribute where attrelid=c and attnum>0 and not attisdropped;
 if actual is distinct from '[["campaign_id","text",true],["phase_id","text",true],["case_id","text",true],["repetition_index","integer",true],["execution_profile_id","text",true],["claimed_at","timestamp with time zone",true]]'::jsonb then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: claims columns' using errcode='55000';
 end if;
 if (select count(*) from pg_attrdef where adrelid in (e,c))<>3
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='execution_id')<>'gen_random_uuid()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=e and a.attname='created_at')<>'now()'
 or (select pg_get_expr(d.adbin,d.adrelid) from pg_attrdef d join pg_attribute a on a.attrelid=d.adrelid and a.attnum=d.adnum
     where d.adrelid=c and a.attname='claimed_at')<>'now()' then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: defaults' using errcode='55000';
 end if;
 -- Compare the complete pinned constraint identity, including PostgreSQL's normalized definitions.
 -- Name/type/count-only tests are insufficient: a replaced CHECK could otherwise pass.
 if (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=e) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_executions'::regclass)
 or (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid=c) is distinct from
    (select jsonb_agg(jsonb_build_array(conname::text,contype::text,conkey,confkey,
      condeferrable,condeferred,convalidated,pg_get_constraintdef(oid,true)) order by conname)
   from pg_constraint where conrelid='pg_temp.orotitan_engine_benchmark_claims'::regclass) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: constraints' using errcode='55000';
 end if;
 if (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=e)
   is distinct from array['orotitan_engine_benchmark_campaign_case_idx','orotitan_engine_benchmark_executions_pkey','orotitan_engine_benchmark_unique_execution']::text[]
 or (select array_agg(i.relname::text order by i.relname) from pg_index x join pg_class i on i.oid=x.indexrelid where x.indrelid=c)
   is distinct from array['orotitan_engine_benchmark_claims_pkey']::text[] then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: indexes' using errcode='55000';
 end if;
 if (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=e and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_executions_immutable']::text[]
 or (select array_agg(tgname::text order by tgname) from pg_trigger where tgrelid=c and not tgisinternal)
   is distinct from array['orotitan_engine_benchmark_claims_immutable']::text[]
 or exists(select 1 from pg_trigger where tgrelid in (e,c) and not tgisinternal and
   (tgfoid<>'public.reject_orotitan_benchmark_execution_mutation()'::regprocedure or tgtype<>27 or tgenabled<>'O')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: triggers' using errcode='55000';
 end if;
 if exists(select 1 from pg_policies where schemaname='public' and tablename in
 ('orotitan_engine_benchmark_executions','orotitan_engine_benchmark_claims')) then
   raise exception 'ROLLBACK_REFUSED_IDENTITY: policies' using errcode='55000';
 end if;
 if not has_table_privilege('service_role',e,'SELECT') or not has_table_privilege('service_role',c,'SELECT')
 or has_table_privilege('service_role',e,'INSERT') or has_table_privilege('service_role',e,'UPDATE')
 or has_table_privilege('service_role',e,'DELETE') or has_table_privilege('service_role',c,'INSERT')
 or has_table_privilege('service_role',c,'UPDATE') or has_table_privilege('service_role',c,'DELETE')
 or exists(select 1 from pg_class t cross join lateral aclexplode(coalesce(t.relacl,acldefault('r',t.relowner))) a
    where t.oid in(e,c) and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: tables' using errcode='55000';
 end if;
 for expected in
 select * from (values
('orotitan_benchmark_dimensions_valid(jsonb)',$expected_0$
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
$expected_0$,false,'i'),
('reject_orotitan_benchmark_execution_mutation()',$expected_1$
begin
  raise exception 'OroTitan benchmark execution rows are append-only'
    using errcode = '55000';
end;
$expected_1$,false,'v'),
('record_orotitan_benchmark_execution(jsonb)',$expected_2$
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
$expected_2$,true,'v'),
('claim_orotitan_benchmark_execution(text,text,text,integer,text)',$expected_3$
declare v_inserted integer;
begin
  if nullif(p_campaign_id,'') is null or nullif(p_phase_id,'') is null
     or p_case_id is null or p_case_id !~ '^V2REF-[0-9]{3}$'
     or p_repetition_index is null or p_repetition_index < 1
     or p_execution_profile_id is null or p_execution_profile_id !~ '^[a-f0-9]{64}$' then
    raise exception 'invalid benchmark claim key' using errcode = '22023';
  end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  insert into public.orotitan_engine_benchmark_claims
    (campaign_id,phase_id,case_id,repetition_index,execution_profile_id)
  values (p_campaign_id,p_phase_id,p_case_id,p_repetition_index,p_execution_profile_id)
  on conflict do nothing;
  get diagnostics v_inserted = row_count;
  if v_inserted=1 then return jsonb_build_object('status','CLAIMED'); end if;
  if exists(select 1 from public.orotitan_engine_benchmark_executions e
    where e.campaign_id=p_campaign_id and e.phase_id=p_phase_id and e.case_id=p_case_id
      and e.repetition_index=p_repetition_index and e.execution_profile_id=p_execution_profile_id) then
    return jsonb_build_object('status','EXISTING');
  end if;
  return jsonb_build_object('status','IN_PROGRESS');
end;
$expected_3$,true,'v')
 ) x(signature,source,security_definer,volatility) loop
   if not exists(select 1 from pg_proc p where p.oid=to_regprocedure('public.'||expected.signature)
   and p.pronamespace='public'::regnamespace and p.proowner=owner_id
   and p.prolang=(select oid from pg_language where lanname='plpgsql')
   and p.prosrc=expected.source and p.prosecdef=expected.security_definer and p.provolatile=expected.volatility
   and p.proconfig=array['search_path=pg_catalog, public']::text[]) then
     raise exception 'ROLLBACK_REFUSED_IDENTITY: function %', expected.signature using errcode='55000';
   end if;
 end loop;
 if (select count(*) from pg_proc where pronamespace='public'::regnamespace
 and proname in('orotitan_benchmark_dimensions_valid','reject_orotitan_benchmark_execution_mutation',
 'record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution'))<>4
 or not has_function_privilege('service_role','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or not has_function_privilege('service_role','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('anon','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('authenticated','public.record_orotitan_benchmark_execution(jsonb)','EXECUTE')
 or has_function_privilege('anon','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or has_function_privilege('authenticated','public.claim_orotitan_benchmark_execution(text,text,text,integer,text)','EXECUTE')
 or exists(select 1 from pg_proc p cross join lateral aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 where p.pronamespace='public'::regnamespace and p.proname in('orotitan_benchmark_dimensions_valid',
 'reject_orotitan_benchmark_execution_mutation','record_orotitan_benchmark_execution','claim_orotitan_benchmark_execution')
 and a.grantee not in(owner_id,'service_role'::regrole::oid)) then
   raise exception 'ROLLBACK_REFUSED_ACL: functions' using errcode='55000';
 end if;
end;
$rollback_guard$;
-- RESTRICT catches unplanned external dependencies; any failure rolls back
-- the whole transaction including preceding DDL.
drop table public.orotitan_engine_benchmark_claims restrict;
drop table public.orotitan_engine_benchmark_executions restrict;
drop function public.claim_orotitan_benchmark_execution(text,text,text,integer,text) restrict;
drop function public.record_orotitan_benchmark_execution(jsonb) restrict;
drop function public.reject_orotitan_benchmark_execution_mutation() restrict;
drop function public.orotitan_benchmark_dimensions_valid(jsonb) restrict;
commit;
