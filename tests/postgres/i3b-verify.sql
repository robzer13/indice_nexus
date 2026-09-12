\set ON_ERROR_STOP on

set role service_role;
do $$
declare
  dossier uuid;
  second_dossier uuid;
  issuer uuid;
  second_issuer uuid;
  security uuid;
  second_security uuid;
  first_snapshot uuid := '90000000-0000-4000-8000-000000000101';
  second_snapshot uuid := '90000000-0000-4000-8000-000000000102';
  payload jsonb;
  second_payload jsonb;
  result jsonb;
  failed boolean;
  before_count bigint;
  before_pointer uuid;
  first_fingerprint text;
  second_fingerprint text;
  actual_constraint text;
begin
  select m.dossier_id, m.issuer_id, m.security_id into dossier, issuer, security
  from public.legacy_company_identity_map m order by m.legacy_company_id limit 1;
  select m.dossier_id, m.issuer_id, m.security_id into second_dossier, second_issuer, second_security
  from public.legacy_company_identity_map m order by m.legacy_company_id offset 1 limit 1;

  if not exists (select 1 from pg_proc where oid = 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)'::regprocedure) then raise exception 'T1 RPC signature missing'; end if;
  if (select count(*) from pg_proc where proname = 'persist_orotitan_research_snapshot') <> 1 then raise exception 'T1 RPC overload exists'; end if;
  if has_function_privilege('anon', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE') then raise exception 'T2 anon EXECUTE exposed'; end if;
  if has_function_privilege('authenticated', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE') then raise exception 'T3 authenticated EXECUTE exposed'; end if;
  if has_function_privilege('public', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE') then raise exception 'T28 PUBLIC EXECUTE exposed'; end if;
  if not has_function_privilege('service_role', 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)', 'EXECUTE') then raise exception 'T4 service_role EXECUTE missing'; end if;
  if not (select p.prosecdef from pg_proc p where p.oid = 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)'::regprocedure)
     or not (select p.proconfig @> array['search_path=pg_catalog, public']::text[] from pg_proc p where p.oid = 'public.persist_orotitan_research_snapshot(uuid, uuid, jsonb)'::regprocedure) then
    raise exception 'T29 SECURITY DEFINER/search_path boundary is incorrect';
  end if;

  payload := jsonb_build_object(
    'snapshot_id', first_snapshot::text, 'report_id', 'i3b-report-1',
    'issuer_id', issuer::text, 'security_id', security::text, 'execution_mode', 'ANALYZE',
    'data_lock', jsonb_build_object('data_cutoff', '2026-09-10', 'calculation_date', '2026-09-11'),
    'versions', jsonb_build_object('report_version', 'r1', 'method_version', 'm1',
      'calculation_version', 'c1', 'evidence_ledger_version', 'e1')
  );
  second_payload := jsonb_set(jsonb_set(payload, '{snapshot_id}', to_jsonb(second_snapshot::text)), '{report_id}', to_jsonb('i3b-report-2'::text));

  if has_table_privilege('service_role', 'public.research_snapshots', 'INSERT') then raise exception 'T5 direct INSERT privilege exists'; end if;
  if has_table_privilege('service_role', 'public.research_snapshots', 'UPDATE') or has_table_privilege('service_role', 'public.research_snapshots', 'DELETE') then raise exception 'T6 direct UPDATE/DELETE privilege exists'; end if;

  result := public.persist_orotitan_research_snapshot(dossier, null, payload);
  if result->>'status' <> 'INSERTED' then raise exception 'T7 first valid insert failed'; end if;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> first_snapshot then raise exception 'T8 NULL pointer did not advance to #1'; end if;
  first_fingerprint := md5((select row_to_json(s)::text from public.research_snapshots s where s.snapshot_id = first_snapshot));

  result := public.persist_orotitan_research_snapshot(dossier, null, payload);
  if result->>'status' <> 'IDEMPOTENT_SUCCESS' then raise exception 'T17 original request retry was not idempotent'; end if;
  if (select count(*) from public.research_snapshots) <> 1 then raise exception 'T17 retry inserted a duplicate'; end if;

  result := public.persist_orotitan_research_snapshot(dossier, first_snapshot, second_payload);
  if result->>'status' <> 'INSERTED' then raise exception 'T9 second valid insert failed'; end if;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> second_snapshot then raise exception 'T10 pointer did not advance to #2'; end if;
  second_fingerprint := md5((select row_to_json(s)::text from public.research_snapshots s where s.snapshot_id = second_snapshot));
  if (select count(*) from public.research_snapshots) <> 2 then raise exception 'T9 row count mismatch'; end if;

  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, null, payload); exception when sqlstate '23505' then failed := true; end;
  if not failed then raise exception 'T19 stale replay #1 with original NULL expected pointer was accepted'; end if;
  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, second_snapshot, payload); exception when sqlstate '23505' then failed := true; end;
  if not failed then raise exception 'T19 stale replay #1 with expected #2 was accepted'; end if;
  if (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> second_snapshot then raise exception 'T20 stale replay regressed pointer'; end if;
  if md5((select row_to_json(s)::text from public.research_snapshots s where s.snapshot_id = first_snapshot)) <> first_fingerprint then raise exception 'T11 snapshot #1 changed'; end if;
  if md5((select row_to_json(s)::text from public.research_snapshots s where s.snapshot_id = second_snapshot)) <> second_fingerprint then raise exception 'T11 snapshot #2 changed'; end if;

  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, null, jsonb_set(payload, '{report_id}', '"different"')); exception when sqlstate '23505' then failed := true; end;
  if not failed then raise exception 'T18 duplicate snapshot_id with different payload was accepted'; end if;

  before_count := (select count(*) from public.research_snapshots); before_pointer := (select current_snapshot_id from public.research_dossiers where dossier_id = dossier);
  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, first_snapshot, jsonb_set(second_payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000103"')); exception when sqlstate '40001' then failed := true; end;
  if not failed then raise exception 'T12 wrong expected pointer was accepted'; end if;
  if (select count(*) from public.research_snapshots) <> before_count or (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> before_pointer then raise exception 'T13 conflict changed row count or pointer'; end if;

  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, second_snapshot, jsonb_set(jsonb_set(payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000104"'), '{issuer_id}', to_jsonb(second_issuer::text))); exception when sqlstate '23514' then failed := true; end;
  if not failed then raise exception 'T14 dossier/issuer mismatch was accepted'; end if;
  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, second_snapshot, jsonb_set(jsonb_set(payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000105"'), '{security_id}', to_jsonb(second_security::text))); exception when sqlstate '23503' then failed := true; end;
  if not failed then raise exception 'T15 security/issuer mismatch was accepted'; end if;
  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, second_snapshot, jsonb_set(payload - 'versions', '{snapshot_id}', '"90000000-0000-4000-8000-000000000106"')); exception when sqlstate '23502' then failed := true; end;
  if not failed then raise exception 'T16 malformed persistence lock was accepted'; end if;

  before_count := (select count(*) from public.research_snapshots); before_pointer := (select current_snapshot_id from public.research_dossiers where dossier_id = dossier);
  failed := false;
  begin perform public.persist_orotitan_research_snapshot(dossier, second_snapshot, jsonb_set(jsonb_set(payload, '{snapshot_id}', '"90000000-0000-4000-8000-000000000106"'), '{security_id}', '"00000000-0000-4000-8000-000000000999"')); exception when sqlstate '23503' then failed := true; end;
  if not failed then raise exception 'T21 failed insert was accepted'; end if;
  if (select count(*) from public.research_snapshots) <> before_count or (select current_snapshot_id from public.research_dossiers where dossier_id = dossier) <> before_pointer then raise exception 'T21 failed insert changed pointer or row count'; end if;
  if has_table_privilege('anon', 'public.research_snapshots', 'SELECT') or has_table_privilege('authenticated', 'public.research_snapshots', 'SELECT') then raise exception 'T30 browser table path exposed'; end if;
  if not (select c.relrowsecurity from pg_class c where c.oid = 'public.research_snapshots'::regclass) then raise exception 'T24 RLS disabled'; end if;
  if exists (select 1 from pg_policy where polrelid = 'public.research_snapshots'::regclass) then raise exception 'T24 browser policy exists'; end if;
  if (select count(*) from pg_trigger where tgrelid = 'public.research_snapshots'::regclass and tgname = 'research_snapshots_prevent_mutation') <> 1 then raise exception 'T23 immutability trigger missing'; end if;
end $$;
reset role;

do $$
begin
  begin update public.research_snapshots set report_id = 'mutated' where snapshot_id = '90000000-0000-4000-8000-000000000101'; raise exception 'T23 UPDATE was accepted'; exception when sqlstate '55000' then null; end;
  begin delete from public.research_snapshots where snapshot_id = '90000000-0000-4000-8000-000000000101'; raise exception 'T23 DELETE was accepted'; exception when sqlstate '55000' then null; end;
end $$;

create or replace function public.i3b_fail_pointer_update() returns trigger language plpgsql set search_path = pg_catalog, public as $$ begin raise exception 'i3b forced pointer failure' using errcode = 'P0001'; end; $$;
create trigger i3b_fail_pointer_update before update on public.research_dossiers for each row execute function public.i3b_fail_pointer_update();
do $$
declare
  dossier uuid;
  issuer uuid;
  security uuid;
  payload jsonb;
  before_count bigint;
  failed boolean := false;
begin
  select m.dossier_id, m.issuer_id, m.security_id into dossier, issuer, security from public.legacy_company_identity_map m order by m.legacy_company_id limit 1;
  payload := jsonb_build_object('snapshot_id', '90000000-0000-4000-8000-000000000107', 'report_id', 'rollback', 'issuer_id', issuer::text, 'security_id', security::text, 'execution_mode', 'ANALYZE', 'data_lock', jsonb_build_object('data_cutoff', '2026-09-10', 'calculation_date', '2026-09-11'), 'versions', jsonb_build_object('report_version', 'r1', 'method_version', 'm1', 'calculation_version', 'c1', 'evidence_ledger_version', 'e1'));
  before_count := (select count(*) from public.research_snapshots);
  begin perform public.persist_orotitan_research_snapshot(dossier, (select current_snapshot_id from public.research_dossiers where dossier_id = dossier), payload); exception when sqlstate 'P0001' then failed := true; end;
  if not failed then raise exception 'T22 forced pointer transition did not fail'; end if;
  if (select count(*) from public.research_snapshots) <> before_count or exists (select 1 from public.research_snapshots where snapshot_id = '90000000-0000-4000-8000-000000000107') then raise exception 'T22 orphan snapshot remained'; end if;
end $$;
drop trigger i3b_fail_pointer_update on public.research_dossiers;
drop function public.i3b_fail_pointer_update();

set role service_role;
do $$
declare
  dossier uuid;
  issuer uuid;
  security uuid;
  failed boolean;
begin
  select m.dossier_id, m.issuer_id, m.security_id into dossier, issuer, security from public.legacy_company_identity_map m order by m.legacy_company_id limit 1;
  failed := false;
  begin
    insert into public.research_snapshots (snapshot_id, dossier_id, issuer_id, security_id, report_id, execution_mode, data_cutoff, calculation_date, report_version, method_version, calculation_version, evidence_ledger_version, canonical_payload)
    values ('90000000-0000-4000-8000-000000000198', dossier, issuer, security, 'direct', 'ANALYZE', '2026-09-10', '2026-09-11', 'r', 'm', 'c', 'e', '{}');
  exception when insufficient_privilege then failed := true; end;
  if not failed then raise exception 'T5 direct INSERT was accepted'; end if;
  failed := false;
  begin update public.research_snapshots set report_id = 'direct' where snapshot_id = '90000000-0000-4000-8000-000000000101'; exception when insufficient_privilege then failed := true; end;
  if not failed then raise exception 'T6 direct UPDATE was accepted'; end if;
  failed := false;
  begin delete from public.research_snapshots where snapshot_id = '90000000-0000-4000-8000-000000000101'; exception when insufficient_privilege then failed := true; end;
  if not failed then raise exception 'T6 direct DELETE was accepted'; end if;
end $$;
reset role;

do $$
declare failed boolean := false;
begin
  begin perform public.persist_orotitan_research_snapshot((select dossier_id from public.legacy_company_identity_map limit 1), null, null); exception when sqlstate '22023' then failed := true; end;
  if not failed then raise exception 'NULL payload boundary was accepted'; end if;
end $$;

select 'I3-B T1-T30 verification passed' as result;