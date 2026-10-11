\set ON_ERROR_STOP on
-- Actual V1.13 creation/binding surfaces, before runtime columns exist.
do $$
declare p public.orotitan_runs%rowtype; result jsonb; id uuid; n integer;
begin
  select * into strict p from public.orotitan_runs where creation_idempotency_key='method-generation:parent';
  for n in 1..2 loop
    result := public.create_orotitan_method_v2_run('v14:grandfather:'||n,p.issuer_id,p.entry_path,p.canonical_mode,
      'INITIAL',p.data_cutoff,null,null,p.process_version,p.pilotage_contract_version,p.contract_pins,
      p.contract_set_sha256,repeat('d',64),'1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2');
    id := (result->>'run_id')::uuid;
    perform public.bind_orotitan_run_identity(id,(result->>'state_version')::bigint,p.security_id,p.dossier_id,
      null,null,'v14:grandfather:bind:'||n,repeat('e',64));
  end loop;
  -- Ordinary historical Integration rows: no V1.14 artifacts/proof/edges.
  for p in select * from public.orotitan_runs
    where creation_idempotency_key in ('v14:grandfather:1','v14:grandfather:2','method-generation:parent') loop
    insert into public.orotitan_run_stages(run_id,stage_code,stage_contract_name,stage_contract_version,
      stage_contract_sha256,handoff_gate_name,lifecycle_status,started_at)
    values(p.run_id,'INTEGRATION',p.contract_pins #>> '{integration_stage,name}',
      p.contract_pins #>> '{integration_stage,version}',p.contract_pins #>> '{integration_stage,content_sha256}',
      'READY_TO_PUBLISH','IN_PROGRESS',now());
  end loop;
end $$;
-- Capture every historical run, including physical identity, before V1.14.
create table public.runtime_firewall_test_history as
  select run_id,ctid::text physical_row,to_jsonb(r) original_row from public.orotitan_runs r;
