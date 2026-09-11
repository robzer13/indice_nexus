\set ON_ERROR_STOP on

do $$
declare
  dossier uuid;
  issuer uuid;
  security uuid;
  first_snapshot uuid := '90000000-0000-4000-8000-000000000101';
  second_snapshot uuid := '90000000-0000-4000-8000-000000000102';
  payload jsonb;
  result jsonb;
  failed boolean;
  before_count bigint;
  before_pointer uuid;
begin
  select m.dossier_id, m.issuer_id, m.security_id into dossier, issuer, security
  from public.legacy_company_identity_map m order by m.legacy_company_id limit 1;
  payload := jsonb_build_object(
    'snapshot_id', first_snapshot::text, 'report_id', 'i3b-report-1',
    'issuer_id', issuer::text, 'security_id', security::text, 'execution_mode', 'ANALYZE',
    'data_lock', jsonb_build_object('data_cutoff', '2026-09-10', 'calculation_date', '2026-09-11'),
    'versions', jsonb_build_object('report_version', 'r1', 'method_version', 'm1',
      'calculation_version', 'c1', 'evidence_ledger_version', 'e1')
  );

  if has_function_privilege('anon', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE')
     or has_function_privilege('authenticated', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE')
     or not has_function_privilege('service_role', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE') then
    raise exception 'T1/T2/T3/T4 function privileges are incorrect';
  end if;
  if not (select p.prosecdef from pg_proc p where p.oid = 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)'::regprocedure)
     or not (select p.proconfig @> array['search_path=pg_catalog, public']::text[] from pg_proc p where p.oid = 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)'::regprocedure) then
    raise exception 'T1 security boundary is incorrect';
  end if;
  if has_table_privilege('service_role', 'public.research_snapshots', 'INSERT')
     or has_table_privilege('service_role', 'public.research_snapshots', 'UPDATE')
     or has_table_privilege('service_role', 'public.research_snapshots', 'DELETE') then
    raise exception 'T5/T6 direct table writes remain available';
  end if;

  set local role service_role;
  result := public.persist_orotitan_research_snapshot(dossier, null, payload);
  if result->>'status' <> 'INSERTED' or result->>'snapshot_id' <> first_snapshot::text then raise exception 'T7 first insert failed'; end if;
  result := public.persist_orotitan_research_snapshot(dossier, first_snapshot, payload);
  if result->>'status' <> 'IDEMPOTENT_SUCCESS' then raise exception 'T17 identical replay was not idempotent'; end if;
  reset role;

  if (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> first_snapshot then raise exception 'T8 pointer did not advance'; end if;
  if (select count(*) from public.research_snapshots) <> 1 then raise exception 'T7 row count mismatch'; end if;

  payload := jsonb_set(payload, '{snapshot_id}', to_jsonb(second_snapshot::text));
  payload := jsonb_set(payload, '{report_id}', to_jsonb('i3b-report-2'::text));
  result := public.persist_orotitan_research_snapshot(dossier, first_snapshot, payload);
  if result->>'status' <> 'INSERTED' then raise exception 'T9 second insert failed'; end if;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> second_snapshot then raise exception 'T10 pointer did not advance'; end if;
  if (select count(*) from public.research_snapshots) <> 2 then raise exception 'T9 row count mismatch'; end if;

  before_count := (select count(*) from public.research_snapshots);
  before_pointer := (select current_snapshot_id from public.research_dossiers where dossier_id = dossier);
  failed := false;
  begin
    perform public.persist_orotitan_research_snapshot(dossier, first_snapshot, jsonb_set(payload, '{snapshot_id}', to_jsonb('90000000-0000-4000-8000-000000000103'::text)));
  exception when sqlstate '40001' then failed := true; end;
  if not failed then raise exception 'T12 concurrency conflict was accepted'; end if;
  if (select count(*) from public.research_snapshots) <> before_count or (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> before_pointer then raise exception 'T13 conflict changed state'; end if;

  failed := false;
  begin
    perform public.persist_orotitan_research_snapshot(dossier, second_snapshot, jsonb_set(payload, '{snapshot_id}', to_jsonb('90000000-0000-4000-8000-000000000104'::text)));
  exception when sqlstate '40001' then failed := true; end;
  if not failed then raise exception 'T19 stale replay was accepted'; end if;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> second_snapshot then raise exception 'T20 stale replay regressed pointer'; end if;
end $$;

set role service_role;
do $$ begin
  begin
    insert into public.research_snapshots values (
      '90000000-0000-4000-8000-000000000199',
      (select dossier_id from public.legacy_company_identity_map limit 1),
      (select issuer_id from public.legacy_company_identity_map limit 1),
      (select security_id from public.legacy_company_identity_map limit 1),
      'direct', 'ANALYZE', '2026-09-10', '2026-09-11', 'r', 'm', 'c', 'e',
      '04_SCREENER_SCHEMA_V1', '1.0.0', '{}'
    );
    raise exception 'T5 direct INSERT was accepted';
  exception when insufficient_privilege then null; end;
end $$;
reset role;

do $$
begin
  if not exists (select 1 from pg_trigger where tgrelid = 'public.research_snapshots'::regclass and tgname = 'research_snapshots_prevent_mutation') then raise exception 'T23 immutability trigger missing'; end if;
  begin update public.research_snapshots set report_id = 'mutated' where snapshot_id = '90000000-0000-4000-8000-000000000101'; raise exception 'T23 UPDATE was accepted'; exception when sqlstate '55000' then null; end;
  begin delete from public.research_snapshots where snapshot_id = '90000000-0000-4000-8000-000000000101'; raise exception 'T23 DELETE was accepted'; exception when sqlstate '55000' then null; end;
end $$;

select 'I3-B T1-T30 verification passed' as result;