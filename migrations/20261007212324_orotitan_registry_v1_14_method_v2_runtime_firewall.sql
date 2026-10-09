-- Registry V1.14: inactive Method-V2 runtime firewall. LOCAL/CODE ONLY.
-- No runtime role creation grants, activation, publisher or snapshot mutation.
begin;

create function public.orotitan_method_v2_runtime_contract_pins()
returns jsonb language sql immutable set search_path = pg_catalog, public
as $pins$ select '{"analysis_standard":{"name":"01_ANALYSIS_STANDARD_V1","version":"1.0","content_sha256":"9283a0df395d4596c93cf6cfad9644ce114b343ee80e6e0a81e8f94f15d1e3df","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/contract-pin-pack-v1/archives/analysis_standard/01_ANALYSIS_STANDARD_V1.archive.json","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"4702be68ca6f08138add2b10773140572fdd1c71","locator_format":"OROTITAN_MULTIPART_GZIP_V1"}},"deep_dive_stage":{"name":"OROTITAN_METHOD_V2_DEEP_DIVE_STAGE_CONTRACT_SUCCESSOR_V1.0","version":"1.0","content_sha256":"7cc34e39a8c8e5895c63ba51781b41aa2a81011219e14b8e127e0c85a76f2b3d","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_DEEP_DIVE_STAGE_CONTRACT_SUCCESSOR_V1.0.md","commit_sha":"3ba9cb7e83368155abcca9d44429df490bb08b9e","blob_sha":"8d046c2bc1f0a1b9cc05f24bd44b7d2c9f580466","locator_format":"RAW_CANONICAL_V1"}},"execution_patch":{"name":"OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1","version":"1.0.1","content_sha256":"6e45f39c03e79c34911850c84904b36f3d95266288a5528127a59d0b8d675973","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"449bb43b664eae9d8c9fd98440e5511c6fbf4be3","locator_format":"RAW_CANONICAL_V1"}},"i2":{"name":"I2_CANONICAL_COMPUTATION","version":"1.0","content_sha256":"fa5f6c10851182660c8b11526266620556cc01d5e8b42c769819cbbcf3d3ed35","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I2_CANONICAL_COMPUTATION.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"93633adec3409144fcb03d5ec7b17ecfe22ff736","locator_format":"RAW_CANONICAL_V1"}},"i3b":{"name":"I3B_VALIDATED_SNAPSHOT_WRITER","version":"1.0","content_sha256":"7db0ca33867200087ba39716ba1618150cc057e975ede4b4075d403e7d3ccb25","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"docs/orotitan-equity/I3B_VALIDATED_SNAPSHOT_WRITER.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"65f5c0c013362fdd7ba64a8975a8d25b74d408d6","locator_format":"RAW_CANONICAL_V1"}},"integration_spec":{"name":"04_INTEGRATION_SPEC_V1_PATCHED","version":"1.0","content_sha256":"cac78e505d354a124f5fdddb726baf31ecf7f0b6b90653fe578a5c5cca9a8238","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/04_INTEGRATION_SPEC_V1_PATCHED.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"e41aee33c714128acacc1342af0949d4d38b6442","locator_format":"RAW_CANONICAL_V1"}},"integration_stage":{"name":"OROTITAN_METHOD_V2_INTEGRATION_ADMISSION_SUCCESSOR_V1.0","version":"1.0","content_sha256":"c88c915823b04112eede6acb5b0ec136b8149d7f550cd3c9e6edea82c6876ae3","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_INTEGRATION_ADMISSION_SUCCESSOR_V1.0.md","commit_sha":"3ba9cb7e83368155abcca9d44429df490bb08b9e","blob_sha":"63c98611c4adc2345fca144e5daec7cbdbe7a650","locator_format":"RAW_CANONICAL_V1"}},"investment_policy":{"name":"OROTITAN_INVESTMENT_POLICY_V1.0.0","version":"1.0.0","content_sha256":"66cc29ccaccf4ccf65f58c2959bf95bcf676e67567aa1dd57863b2cb9a2936c8","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/OROTITAN_INVESTMENT_POLICY_V1.0.0.md","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"fd0121bbb14e35370ddd700d9e9845eaf1ec9f75","locator_format":"RAW_CANONICAL_V1"}},"master_prompt":{"name":"OROTITAN_METHOD_V2_MASTER_PROMPT_SEQUENCING_SUCCESSOR_V1.0","version":"1.0","content_sha256":"efe42012bcf8f41be4942ff216d49b7a5010e67c0a1101d82e5c4a4330fde51c","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_MASTER_PROMPT_SEQUENCING_SUCCESSOR_V1.0.md","commit_sha":"3ba9cb7e83368155abcca9d44429df490bb08b9e","blob_sha":"abfeb6b759acff4c854d648548cc01cf1a0208ba","locator_format":"RAW_CANONICAL_V1"}},"pilotage":{"name":"OROTITAN_METHOD_V2_PILOTAGE_SUCCESSOR_V1.0","version":"1.0","content_sha256":"a332708924cb2c684de8d87b21aafa52d74b658467e11b8f2baf9adc8a25df5c","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_PILOTAGE_SUCCESSOR_V1.0.md","commit_sha":"3ba9cb7e83368155abcca9d44429df490bb08b9e","blob_sha":"b940df2a0fa3435147a46bd4e89cb39b9f9d71b2","locator_format":"RAW_CANONICAL_V1"}},"process":{"name":"OROTITAN_METHOD_V2_EXECUTION_PROCESS_SUCCESSOR_V1.0","version":"1.0","content_sha256":"4bb6672949c704349316f40c4a1280fcd65e2e96d40a619b9e500a7eba89d088","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_PROCESS_SUCCESSOR_V1.0.md","commit_sha":"3ba9cb7e83368155abcca9d44429df490bb08b9e","blob_sha":"e5280c1d442bee3898d99d9228e7b8d1ab1b482f","locator_format":"RAW_CANONICAL_V1"}},"research_stage":{"name":"OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0","version":"1.0","content_sha256":"b9a5930c91d90e498e41c1640f912fb5e38fd0ad607899a5e2ca0a616f3d47a2","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/execution-freeze/OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0.md.gz","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"3bf94124835fe17b31e7b11410274ed585f92aa5","locator_format":"OROTITAN_GZIP_V1"}},"screener_schema":{"name":"04_SCREENER_SCHEMA_V1_PATCHED","version":"1.0.0","content_sha256":"bf407ca217553521586ba5f6002180ff6522700b4671986079ea6ed577604ede","locator":{"backend":"GITHUB_IMMUTABLE","repository":"robzer13/indice_nexus","path":"contracts/orotitan-equity/v1/04_SCREENER_SCHEMA_V1_PATCHED.json","commit_sha":"8aba7cee6a9b38204785c16976e65e9010f5d959","blob_sha":"22e13b5fb058371eca863613a5f1ac8e6582da00","locator_format":"RAW_CANONICAL_V1"}}}'::jsonb $pins$;
revoke all on function public.orotitan_method_v2_runtime_contract_pins() from public, anon, authenticated, service_role;

do $$ begin
  if public.orotitan_contract_set_sha256(public.orotitan_method_v2_runtime_contract_pins())
    is distinct from '23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea'
    or (select count(*) from jsonb_object_keys(public.orotitan_method_v2_runtime_contract_pins())) <> 13 then
    raise exception 'METHOD_V2_FROZEN_PIN_PACK_MISMATCH';
  end if;
end $$;

create table public.orotitan_method_v2_runtime_control (
  singleton boolean primary key default true check (singleton),
  admission_mode text not null default 'INSTALLED_INACTIVE'
    check (admission_mode in ('INSTALLED_INACTIVE','CANARY_ONLY','ACTIVE_FOR_NEW_RUNS','DISABLED')),
  admission_scope text not null default 'INITIAL_ONLY'
    check (admission_scope in ('INITIAL_ONLY','INITIAL_AND_REFRESH')),
  publication_mode text not null default 'DISABLED'
    check (publication_mode in ('DISABLED','CANARY_ONLY','NORMAL')),
  methodology_authority_sha256 text not null default '1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2'
    check (methodology_authority_sha256 = '1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2'),
  contract_set_sha256 text not null default '23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea'
    check (contract_set_sha256 = '23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea'),
  runtime_binding_sha256 text not null default 'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0'
    check (runtime_binding_sha256 = 'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0'),
  runtime_commit_sha text check (runtime_commit_sha ~ '^[0-9a-f]{40}$'),
  check (admission_mode in ('INSTALLED_INACTIVE','DISABLED') or runtime_commit_sha is not null)
);
insert into public.orotitan_method_v2_runtime_control(singleton) values(true);
create trigger orotitan_method_v2_control_no_delete
before delete or truncate on public.orotitan_method_v2_runtime_control
for each statement execute function public.reject_orotitan_history_delete();

create table public.orotitan_method_v2_canary_allowlist (
  dossier_id uuid not null,
  issuer_id uuid not null references public.issuers(issuer_id),
  security_id uuid not null,
  primary key (dossier_id, issuer_id, security_id),
  foreign key (dossier_id, issuer_id) references public.research_dossiers(dossier_id, issuer_id),
  foreign key (security_id, issuer_id) references public.securities(security_id, issuer_id)
);
-- Empty by construction; runtime roles cannot self-enroll.
alter table public.orotitan_method_v2_runtime_control enable row level security;
alter table public.orotitan_method_v2_canary_allowlist enable row level security;
revoke all on public.orotitan_method_v2_runtime_control, public.orotitan_method_v2_canary_allowlist
  from public, anon, authenticated, service_role;
grant select on public.orotitan_method_v2_runtime_control, public.orotitan_method_v2_canary_allowlist to service_role;

alter table public.orotitan_runs add column runtime_binding_sha256 text,
  add column runtime_commit_sha text,
  add constraint orotitan_method_v2_runtime_birth_check check (
    runtime_binding_sha256 is null or (
      methodology_generation = 'METHOD_V2'
      and runtime_binding_sha256 = 'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0'
      and runtime_commit_sha is not null and runtime_commit_sha ~ '^[0-9a-f]{40}$'
      and parent_run_id is null and issuer_id is not null and security_id is not null and dossier_id is not null
      and contract_pins = public.orotitan_method_v2_runtime_contract_pins()
      and contract_set_sha256 = '23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea'
      and process_version = public.orotitan_method_v2_runtime_contract_pins() #>> '{process,version}'
      and pilotage_contract_version = public.orotitan_method_v2_runtime_contract_pins() #>> '{pilotage,version}'
      and entry_path = 'IMPOSED_COMPANY'
    )
  );
create unique index orotitan_one_nonterminal_fresh_method_v2_dossier_idx
on public.orotitan_runs(dossier_id)
where methodology_generation = 'METHOD_V2' and parent_run_id is null
  and runtime_binding_sha256 = 'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0'
  and run_status in ('CREATED','ACTIVE','PAUSED','BLOCKED','READY_TO_PUBLISH');

create function public.enforce_orotitan_method_v2_runtime_identity()
returns trigger language plpgsql set search_path = pg_catalog, public as $$
begin
  if new.runtime_binding_sha256 is distinct from old.runtime_binding_sha256
    or new.runtime_commit_sha is distinct from old.runtime_commit_sha then
    raise exception 'METHOD_V2_RUNTIME_IDENTITY_IMMUTABLE' using errcode='23514';
  end if;
  return new;
end $$;
create trigger orotitan_method_v2_runtime_identity_immutable before update on public.orotitan_runs
for each row execute function public.enforce_orotitan_method_v2_runtime_identity();
revoke all on function public.enforce_orotitan_method_v2_runtime_identity() from public, anon, authenticated, service_role;

create function public.enforce_orotitan_legacy_new_run_firewall()
returns trigger language plpgsql security definer set search_path = pg_catalog, public as $$
begin
  if new.methodology_generation = 'METHOD_V1' and new.parent_run_id is null
     and new.run_scope = 'COMPANY_ANALYSIS' and exists (
       select 1 from public.orotitan_method_v2_runtime_control
       where admission_mode in ('CANARY_ONLY','ACTIVE_FOR_NEW_RUNS')) then
    raise exception 'LEGACY_NEW_RUN_FIREWALL' using errcode='23514';
  end if;
  return new;
end $$;
create trigger orotitan_legacy_new_run_firewall before insert on public.orotitan_runs
for each row execute function public.enforce_orotitan_legacy_new_run_firewall();
revoke all on function public.enforce_orotitan_legacy_new_run_firewall() from public, anon, authenticated, service_role;

create function public.create_orotitan_method_v2_runtime_run(
  p_creation_idempotency_key text, p_issuer_id uuid, p_security_id uuid, p_dossier_id uuid,
  p_expected_current_snapshot_id uuid, p_data_cutoff date,
  p_request_fingerprint_sha256 text, p_runtime_binding_sha256 text
) returns jsonb language plpgsql security definer set search_path = pg_catalog, public as $$
declare
  c public.orotitan_method_v2_runtime_control%rowtype;
  d public.research_dossiers%rowtype;
  s public.research_snapshots%rowtype;
  r public.orotitan_runs%rowtype;
  pins jsonb := public.orotitan_method_v2_runtime_contract_pins();
  v_run_id uuid := gen_random_uuid();
  v_event_id uuid;
  v_type text;
  v_mode text;
  v_request jsonb := jsonb_build_object('issuer_id',p_issuer_id,'security_id',p_security_id,
    'dossier_id',p_dossier_id,'expected_current_snapshot_id',p_expected_current_snapshot_id,
    'data_cutoff',p_data_cutoff,'runtime_binding_sha256',p_runtime_binding_sha256);
begin
  if p_creation_idempotency_key is null or length(btrim(p_creation_idempotency_key)) = 0
    or p_request_fingerprint_sha256 is null or p_request_fingerprint_sha256 !~ '^[0-9a-f]{64}$'
    or p_issuer_id is null or p_security_id is null or p_dossier_id is null or p_data_cutoff is null then
    raise exception 'METHOD_V2_INVALID_REQUEST' using errcode='22023';
  end if;
  if p_runtime_binding_sha256 is distinct from 'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0' then
    raise exception 'METHOD_V2_RUNTIME_BINDING_MISMATCH' using errcode='23514';
  end if;
  -- Serialize creation keys across dossiers as well as exact requests on one dossier.
  perform pg_advisory_xact_lock(hashtextextended('method-v2-runtime:' || p_creation_idempotency_key,0));
  select * into r from public.orotitan_runs where creation_idempotency_key=p_creation_idempotency_key;
  if found then
    if r.methodology_generation is distinct from 'METHOD_V2'
      or r.runtime_binding_sha256 is distinct from p_runtime_binding_sha256
      or not exists (select 1 from public.orotitan_run_events e where e.run_id=r.run_id
        and e.event_type='RUN_CREATED' and e.idempotency_key=p_creation_idempotency_key
        and e.request_fingerprint_sha256=p_request_fingerprint_sha256
        and e.payload->'runtime_creation_request'=v_request) then
      raise exception 'IDEMPOTENCY_CONFLICT: Method-V2 runtime creation' using errcode='23514';
    end if;
    return jsonb_build_object('run_id',r.run_id,'state_version',r.state_version,
      'run_status',r.run_status,'idempotent_replay',true);
  end if;
  select * into d from public.research_dossiers where dossier_id=p_dossier_id for update;
  if not found or not d.active or d.issuer_id is distinct from p_issuer_id then
    raise exception 'METHOD_V2_DOSSIER_IDENTITY_MISMATCH' using errcode='23514';
  end if;
  perform 1 from public.securities where security_id=p_security_id and issuer_id=p_issuer_id for share;
  if not found then raise exception 'METHOD_V2_SECURITY_IDENTITY_MISMATCH' using errcode='23514'; end if;
  if d.current_snapshot_id is distinct from p_expected_current_snapshot_id then
    raise exception 'METHOD_V2_STALE_CURRENT_SNAPSHOT' using errcode='40001';
  end if;
  select * into c from public.orotitan_method_v2_runtime_control where singleton for share;
  if not found or c.admission_mode not in ('CANARY_ONLY','ACTIVE_FOR_NEW_RUNS') then
    raise exception 'METHOD_V2_RUNTIME_NOT_ADMITTED' using errcode='23514';
  end if;
  if c.runtime_binding_sha256 is distinct from p_runtime_binding_sha256 or c.runtime_commit_sha is null then
    raise exception 'METHOD_V2_RUNTIME_DEPLOYMENT_UNBOUND' using errcode='23514';
  end if;
  if c.admission_mode='CANARY_ONLY' and not exists (
    select 1 from public.orotitan_method_v2_canary_allowlist a
    where a.dossier_id=p_dossier_id and a.issuer_id=p_issuer_id and a.security_id=p_security_id) then
    raise exception 'METHOD_V2_CANARY_SCOPE_MISMATCH' using errcode='23514';
  end if;
  if d.current_snapshot_id is null then
    v_type := 'INITIAL'; v_mode := 'ANALYZE';
  else
    if c.admission_scope <> 'INITIAL_AND_REFRESH' then
      raise exception 'METHOD_V2_REFRESH_NOT_ADMITTED' using errcode='23514';
    end if;
    select * into s from public.research_snapshots where snapshot_id=d.current_snapshot_id for share;
    if not found or s.dossier_id is distinct from p_dossier_id or s.issuer_id is distinct from p_issuer_id
      or s.security_id is distinct from p_security_id or p_data_cutoff <= s.data_cutoff then
      raise exception 'METHOD_V2_REFRESH_BASELINE_MISMATCH' using errcode='23514';
    end if;
    v_type := 'REFRESH'; v_mode := 'REFRESH';
  end if;
  if exists (select 1 from public.orotitan_runs where dossier_id=p_dossier_id
    and methodology_generation='METHOD_V2' and parent_run_id is null
    and runtime_binding_sha256='b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0'
    and run_status in ('CREATED','ACTIVE','PAUSED','BLOCKED','READY_TO_PUBLISH')) then
    raise exception 'METHOD_V2_FRESH_RUN_ALREADY_EXISTS' using errcode='23514';
  end if;
  insert into public.orotitan_runs(run_id,creation_idempotency_key,issuer_id,security_id,dossier_id,
    parent_run_id,baseline_snapshot_id,entry_path,canonical_mode,run_type,data_cutoff,
    process_version,pilotage_contract_version,contract_pins,contract_set_sha256,
    methodology_generation,methodology_authority_sha256,runtime_binding_sha256,runtime_commit_sha)
  values(v_run_id,p_creation_idempotency_key,p_issuer_id,p_security_id,p_dossier_id,null,d.current_snapshot_id,
    'IMPOSED_COMPANY',v_mode,v_type,p_data_cutoff,pins #>> '{process,version}',pins #>> '{pilotage,version}',
    pins,c.contract_set_sha256,'METHOD_V2',c.methodology_authority_sha256,c.runtime_binding_sha256,c.runtime_commit_sha);
  v_event_id := public.orotitan_insert_event(v_run_id,null,'RUN_CREATED',p_creation_idempotency_key,
    p_request_fingerprint_sha256,'PILOTAGE',jsonb_build_object('runtime_creation_request',v_request));
  return jsonb_build_object('run_id',v_run_id,'state_version',1,'run_status','CREATED',
    'event_id',v_event_id,'idempotent_replay',false);
end $$;
revoke all on function public.create_orotitan_method_v2_runtime_run(text,uuid,uuid,uuid,uuid,date,text,text)
  from public, anon, authenticated, service_role;

-- Trusted verifier output only. No runtime/API role may attest or mutate this proof.
-- The persisted-byte verifier produces exact identities after pure admission;
-- no caller PASS field, analytical score, or alternative validator is stored.
create table public.orotitan_method_v2_challenge_proofs (
  run_id uuid not null references public.orotitan_runs(run_id),
  stage_revision integer not null check (stage_revision >= 1),
  question_ledger_id uuid not null, question_ledger_version integer not null,
  question_ledger_sha256 text not null check (question_ledger_sha256 ~ '^[0-9a-f]{64}$'),
  challenge_report_id uuid not null, challenge_report_version integer not null,
  challenge_report_sha256 text not null check (challenge_report_sha256 ~ '^[0-9a-f]{64}$'),
  fundamentals_lock_id uuid not null, fundamentals_lock_version integer not null,
  fundamentals_lock_sha256 text not null check (fundamentals_lock_sha256 ~ '^[0-9a-f]{64}$'),
  valuation_lock_id uuid not null, valuation_lock_version integer not null,
  valuation_lock_sha256 text not null check (valuation_lock_sha256 ~ '^[0-9a-f]{64}$'),
  certification_id uuid not null, certification_version integer not null,
  certification_sha256 text not null check (certification_sha256 ~ '^[0-9a-f]{64}$'),
  certification_profile_sha256 text not null check (certification_profile_sha256 = 'a9b3eff1930a9a7e6164cfb1f0248c98c8041975ef9adc77f9d152062df0a53c'),
  runtime_binding_sha256 text not null check (runtime_binding_sha256 = 'b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0'),
  validator_identity text not null check (validator_identity = 'verifyPersistedMethodV2Challenge:1.1'),
  primary key (run_id,stage_revision,challenge_report_id,challenge_report_version),
  foreign key (run_id,question_ledger_id,question_ledger_version) references public.orotitan_artifacts(run_id,artifact_id,version),
  foreign key (run_id,challenge_report_id,challenge_report_version) references public.orotitan_artifacts(run_id,artifact_id,version),
  foreign key (run_id,fundamentals_lock_id,fundamentals_lock_version) references public.orotitan_artifacts(run_id,artifact_id,version),
  foreign key (run_id,valuation_lock_id,valuation_lock_version) references public.orotitan_artifacts(run_id,artifact_id,version),
  foreign key (run_id,certification_id,certification_version) references public.orotitan_artifacts(run_id,artifact_id,version)
);
alter table public.orotitan_method_v2_challenge_proofs enable row level security;
revoke all on public.orotitan_method_v2_challenge_proofs from public,anon,authenticated,service_role;
grant select on public.orotitan_method_v2_challenge_proofs to service_role;
create trigger orotitan_method_v2_proofs_immutable before update or delete on public.orotitan_method_v2_challenge_proofs
for each row execute function public.reject_orotitan_event_mutation();

create function public.assert_orotitan_method_v2_deep_dive_lineage(
  p_run_id uuid, p_revision integer, p_manifest_id uuid, p_manifest_version integer
) returns void language plpgsql security definer set search_path = pg_catalog, public as $$
declare
  proof public.orotitan_method_v2_challenge_proofs%rowtype;
  q public.orotitan_artifacts%rowtype;
  ch public.orotitan_artifacts%rowtype;
  cert public.orotitan_artifacts%rowtype;
  f public.orotitan_artifacts%rowtype;
  v public.orotitan_artifacts%rowtype;
  required text[] := array['DEEP_DIVE_REPORT','EVIDENCE_LEDGER','CONFLICT_LEDGER','CALCULATION_LEDGER',
    'MATERIAL_ASSUMPTION_REGISTER','ANALYTICAL_BLOCK_OUTPUTS','CROSS_BLOCK_RECONCILIATION_RECORD',
    'RED_TEAM_PREMORTEM_RECORD','VALUATION_ARTIFACT','CERTIFICATION_ARTIFACT',
    'OROTITAN_TERMINAL_GATE_ARTIFACT','READINESS_NEXT_ACTION_ARTIFACT',
    'PRE_CERTIFICATION_QUESTION_LEDGER','PRE_CERTIFICATION_CHALLENGE_REPORT'];
  kind text;
begin
  foreach kind in array required loop
    if not exists (select 1 from public.orotitan_artifacts a where a.run_id=p_run_id and a.stage_code='DEEP_DIVE'
      and a.manifest_artifact_id=p_manifest_id and a.manifest_version=p_manifest_version
      and a.artifact_type=kind and a.authority_class='AUTHORITATIVE_STAGE_OUTPUT'
      and a.authority_state='AUTHORITATIVE' and a.artifact_status='SEALED' and a.availability_state='AVAILABLE') then
      raise exception 'METHOD_V2_DEEP_DIVE_REQUIRED_OUTPUT: %',kind using errcode='23514';
    end if;
  end loop;
  -- Exactly one current ledger, report and Certification; ambiguity fails closed.
  begin
    select * into strict q from public.orotitan_artifacts where run_id=p_run_id and stage_code='DEEP_DIVE'
      and manifest_artifact_id=p_manifest_id and manifest_version=p_manifest_version and artifact_type='PRE_CERTIFICATION_QUESTION_LEDGER';
    select * into strict ch from public.orotitan_artifacts where run_id=p_run_id and stage_code='DEEP_DIVE'
      and manifest_artifact_id=p_manifest_id and manifest_version=p_manifest_version and artifact_type='PRE_CERTIFICATION_CHALLENGE_REPORT';
    select * into strict cert from public.orotitan_artifacts where run_id=p_run_id and stage_code='DEEP_DIVE'
      and manifest_artifact_id=p_manifest_id and manifest_version=p_manifest_version and artifact_type='CERTIFICATION_ARTIFACT';
    select * into strict proof from public.orotitan_method_v2_challenge_proofs where run_id=p_run_id
      and stage_revision=p_revision and question_ledger_id=q.artifact_id and question_ledger_version=q.version
      and challenge_report_id=ch.artifact_id and challenge_report_version=ch.version
      and question_ledger_sha256=q.content_sha256 and challenge_report_sha256=ch.content_sha256
      -- An edge alone is insufficient: the owner verifier must have reread this
      -- exact Certification payload and validated its complete Challenge binding.
      and certification_id=cert.artifact_id and certification_version=cert.version
      and certification_sha256=cert.content_sha256;
  exception when no_data_found or too_many_rows then
    raise exception 'METHOD_V2_CHALLENGE_PERSISTED_PROOF_REQUIRED' using errcode='23514';
  end;
  select * into f from public.orotitan_artifacts where run_id=p_run_id
    and artifact_id=proof.fundamentals_lock_id and version=proof.fundamentals_lock_version;
  select * into v from public.orotitan_artifacts where run_id=p_run_id
    and artifact_id=proof.valuation_lock_id and version=proof.valuation_lock_version;
  if f.stage_code is distinct from 'DEEP_DIVE' or v.stage_code is distinct from 'DEEP_DIVE'
    or f.artifact_type is distinct from 'FUNDAMENTALS_LOCK' or v.artifact_type is distinct from 'VALUATION_LOCK'
    or f.content_sha256 is distinct from proof.fundamentals_lock_sha256 or v.content_sha256 is distinct from proof.valuation_lock_sha256
    or f.artifact_status is distinct from 'SEALED' or v.artifact_status is distinct from 'SEALED'
    or f.availability_state is distinct from 'AVAILABLE' or v.availability_state is distinct from 'AVAILABLE'
    or f.authority_class is distinct from 'AUTHORITATIVE_STAGE_OUTPUT' or v.authority_class is distinct from 'AUTHORITATIVE_STAGE_OUTPUT'
    or f.authority_state is distinct from 'AUTHORITATIVE' or v.authority_state is distinct from 'AUTHORITATIVE'
    or f.manifest_artifact_id is distinct from p_manifest_id or v.manifest_artifact_id is distinct from p_manifest_id
    or f.manifest_version is distinct from p_manifest_version or v.manifest_version is distinct from p_manifest_version then
    raise exception 'METHOD_V2_CHALLENGE_LOCK_LINEAGE_MISMATCH' using errcode='23514';
  end if;
  if not exists (select 1 from public.orotitan_artifact_edges where child_run_id=p_run_id and parent_run_id=p_run_id
    and child_artifact_id=ch.artifact_id and child_version=ch.version and parent_artifact_id=q.artifact_id
    and parent_version=q.version and relation_type='CONSUMES')
    or not exists (select 1 from public.orotitan_artifact_edges where child_run_id=p_run_id and parent_run_id=p_run_id
    and child_artifact_id=ch.artifact_id and child_version=ch.version and parent_artifact_id=v.artifact_id
    and parent_version=v.version and relation_type='CONSUMES')
    or not exists (select 1 from public.orotitan_artifact_edges where child_run_id=p_run_id and parent_run_id=p_run_id
    and child_artifact_id=cert.artifact_id and child_version=cert.version and parent_artifact_id=ch.artifact_id
    and parent_version=ch.version and relation_type='CONSUMES') then
    raise exception 'METHOD_V2_CHALLENGE_CERTIFICATION_LINEAGE_MISMATCH' using errcode='23514';
  end if;
end $$;
revoke all on function public.assert_orotitan_method_v2_deep_dive_lineage(uuid,integer,uuid,integer)
from public,anon,authenticated,service_role;

create function public.enforce_orotitan_method_v2_stage_firewall()
returns trigger language plpgsql security definer set search_path = pg_catalog, public as $$
declare d public.orotitan_run_stages%rowtype;
begin
  if not exists(select 1 from public.orotitan_runs where run_id=new.run_id and methodology_generation='METHOD_V2'
    and runtime_binding_sha256='b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0') then
    return new;
  end if;
  if new.stage_code='DEEP_DIVE' and (new.lifecycle_status='COMPLETE' or new.handoff_gate_state='YES') then
    if new.active_manifest_kind is distinct from 'FINAL' then
      raise exception 'METHOD_V2_DEEP_DIVE_FINAL_REQUIRED' using errcode='23514';
    end if;
    perform public.assert_orotitan_method_v2_deep_dive_lineage(new.run_id,new.stage_revision,
      new.active_manifest_artifact_id,new.active_manifest_version);
  elsif new.stage_code='INTEGRATION' then
    -- The inherited Research/Deep Dive reopen RPC invalidates an existing downstream row.
    -- This inert UPDATE admits no readiness, active manifest or progression.
    if tg_op='UPDATE' and new.lifecycle_status='BLOCKED'
      and new.contract_status_code='UPSTREAM_STAGE_REOPENED'
      and new.handoff_gate_state='NOT_EVALUATED'
      and new.active_manifest_artifact_id is null and new.active_manifest_version is null
      and new.active_manifest_kind is null and new.completed_at is null
      and new.blocker_summary in (
        '[{"code":"UPSTREAM_STAGE_REOPENED","summary":"Deep Dive was reopened"}]'::jsonb,
        '[{"code":"UPSTREAM_STAGE_REOPENED","summary":"Research was reopened"}]'::jsonb)
      and new.state_version=old.state_version+1
      and new.stage_revision=old.stage_revision+(case when old.lifecycle_status='COMPLETE' then 1 else 0 end) then
      return new;
    end if;
    select * into d from public.orotitan_run_stages where run_id=new.run_id and stage_code='DEEP_DIVE' for share;
    if not found or d.lifecycle_status is distinct from 'COMPLETE' or d.active_manifest_kind is distinct from 'FINAL'
      or d.handoff_gate_state is distinct from 'YES' then
      raise exception 'METHOD_V2_INTEGRATION_NOT_ADMITTED' using errcode='23514';
    end if;
    perform public.assert_orotitan_method_v2_deep_dive_lineage(new.run_id,d.stage_revision,
      d.active_manifest_artifact_id,d.active_manifest_version);
  end if;
  return new;
end $$;
create trigger zz_orotitan_method_v2_stage_firewall before insert or update on public.orotitan_run_stages
for each row execute function public.enforce_orotitan_method_v2_stage_firewall();
revoke all on function public.enforce_orotitan_method_v2_stage_firewall() from public,anon,authenticated,service_role;

commit;
