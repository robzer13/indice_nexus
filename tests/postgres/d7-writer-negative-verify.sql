-- D7 negative PostgreSQL matrix. Runs only on disposable orotitan_d7_<pid>.
\set ON_ERROR_STOP on

do $$
begin
 if current_database() !~ '^orotitan_d7_[0-9]+$' or session_user <> 'postgres'
 then raise exception 'D7_SANDBOX_DATABASE_GUARD_FAILED'; end if;
 if to_regprocedure('public.persist_orotitan_research_snapshot_v3_1(uuid,uuid,jsonb)') is not null
 then raise exception 'D7_PUBLIC_V31_WRITER_MUST_NOT_EXIST'; end if;
 if to_regprocedure('public.persist_orotitan_research_snapshot_v2(uuid,uuid,jsonb)') is null
 then raise exception 'D7_LEGACY_V2_WRITER_DISAPPEARED'; end if;
 if to_regprocedure('public.persist_orotitan_research_snapshot(uuid,uuid,jsonb)') is null
 then raise exception 'D7_LEGACY_V1_WRITER_DISAPPEARED'; end if;
 if has_function_privilege('anon', 'd7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)', 'EXECUTE')
    or has_function_privilege('authenticated', 'd7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)', 'EXECUTE')
    or has_function_privilege('public', 'd7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)', 'EXECUTE')
    or not has_function_privilege('service_role', 'd7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)', 'EXECUTE')
 then raise exception 'D7_FUNCTION_PRIVILEGES_NOT_LEAST_PRIVILEGE'; end if;
 if has_table_privilege('service_role','d7_sandbox.approved_inputs','INSERT')
    or has_table_privilege('service_role','d7_sandbox.approved_inputs','UPDATE')
    or has_table_privilege('service_role','d7_sandbox.approved_inputs','DELETE')
 then raise exception 'D7_API_ROLE_CAN_MINT_ATTESTATION'; end if;
 if has_table_privilege('service_role','public.research_snapshots','INSERT')
    or has_table_privilege('service_role','public.research_snapshots','UPDATE')
    or has_table_privilege('service_role','public.research_snapshots','DELETE')
 then raise exception 'D7_API_ROLE_CAN_BYPASS_WRITER'; end if;
 if not (select p.prosecdef and p.proconfig @> array['search_path=pg_catalog, public, d7_sandbox']::text[]
  from pg_proc p where p.oid='d7_sandbox.persist_v31_candidate(uuid,uuid,uuid,jsonb)'::regprocedure)
 then raise exception 'D7_FUNCTION_SEARCH_PATH_NOT_FIXED'; end if;
end $$;

-- Only the privileged local test harness can create proofs. The values are
-- synthetic and MUST NEVER be considered genuine analyst admissions.
create table d7_sandbox.fixtures (
 name text primary key,
 run_id uuid not null,
 dossier_id uuid not null,
 issuer_id uuid not null,
 security_id uuid not null,
 payload jsonb not null
);
insert into d7_sandbox.fixtures(name,run_id,dossier_id,issuer_id,security_id,payload)
select 'p1','a1000000-0000-4000-8000-000000000001'::uuid,
 m.dossier_id,m.issuer_id,m.security_id,
 jsonb_build_object(
   'snapshot_id','93000000-0000-4000-8000-000000000101',
   'issuer_id',m.issuer_id::text,'security_id',m.security_id::text,
   'report_id','d7-sandbox-1','execution_mode','ANALYZE',
   'data_lock',jsonb_build_object(
     'data_cutoff','2026-09-18','calculation_date','2026-09-18',
     'reference_price',12,'reference_price_date','2026-09-18'),
   'versions',jsonb_build_object(
     'report_version','r3','method_version','m3','calculation_version','c3',
     'evidence_ledger_version','e3'),
   'v2_product',jsonb_build_object(
     'classification',jsonb_build_object('taxonomy_version','OROTITAN_TAXONOMY_V2.0'),
     'business_summary',jsonb_build_object('business_description_short','Test-only business.'),
     'investment_thesis',jsonb_build_object('quality_case','Test','valuation_case','Test','key_risk','Test')),
   'v3_methodology',jsonb_build_object(
     'economic_share_count',jsonb_build_object(
       'methodology_version','OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0',
       'representation','EXACT','share_count_as_of_date','2026-09-18',
       'valuation_admission','YES','propagation_status','SCALAR_EXACT',
       'exact_count',1000000,'provenance_refs',jsonb_build_array('TEST_ONLY_EVIDENCE')),
     'valuation_timing',jsonb_build_object(
       'methodology_version','OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0',
       'valuation_date','2026-09-18','time_origin','2026-09-18',
       'reference_price_date','2026-09-18','economic_share_count_date','2026-09-18',
       'share_count_as_of_date','2026-09-18','ev_to_equity_bridge_date','2026-09-18',
       'market_cap_date','2026-09-18','per_share_output_date','2026-09-18'))
 )
from public.legacy_company_identity_map m order by m.legacy_company_id limit 1;
insert into d7_sandbox.fixtures
select 'p2',run_id,dossier_id,issuer_id,security_id,
 jsonb_set(jsonb_set(payload,'{snapshot_id}','"93000000-0000-4000-8000-000000000102"'),'{report_id}','"d7-sandbox-2"')
from d7_sandbox.fixtures where name='p1';
insert into d7_sandbox.fixtures
select 'p3',run_id,dossier_id,issuer_id,security_id,
 jsonb_set(jsonb_set(payload,'{snapshot_id}','"93000000-0000-4000-8000-000000000103"'),'{report_id}','"d7-sandbox-rollback"')
from d7_sandbox.fixtures where name='p1';
insert into d7_sandbox.fixtures
select 'bad_security',f.run_id,f.dossier_id,f.issuer_id,m.security_id,
 jsonb_set(
   jsonb_set(jsonb_set(f.payload,'{snapshot_id}','"93000000-0000-4000-8000-000000000104"'),
     '{report_id}','"d7-foreign-security"'),
   '{security_id}',to_jsonb(m.security_id::text))
from d7_sandbox.fixtures f cross join lateral (
 select m.security_id from public.legacy_company_identity_map m
 where m.issuer_id<>f.issuer_id order by m.legacy_company_id limit 1
) m where f.name='p1';
insert into d7_sandbox.fixtures
select 'bad_ev',run_id,dossier_id,issuer_id,security_id,
 jsonb_set(
   jsonb_set(jsonb_set(payload,'{snapshot_id}','"93000000-0000-4000-8000-000000000105"'),
     '{report_id}','"d7-wrong-ev"'),
   '{v3_methodology,valuation_timing,ev_to_equity_bridge_date}','"NOT_APPLICABLE"')
from d7_sandbox.fixtures where name='p1';

insert into d7_sandbox.approved_inputs
(candidate_snapshot_id,run_id,dossier_id,issuer_id,security_id,payload_sha256,cash_flow_basis,approval_source)
select (payload->>'snapshot_id')::uuid,run_id,dossier_id,issuer_id,security_id,
 encode(public.digest(convert_to(payload::text,'UTF8'),'sha256'),'hex'),
 'FCFF', 'D7_TEST_ONLY_APPROVED'
from d7_sandbox.fixtures;

-- Physical V3.1 schema-pair firewall must reject mixed V3.1/2.0.0 rows
-- even when invoked by the superuser inside the disposable database.
do $physical$
declare
 f record;
 rejected boolean := false;
begin
 select * into f from d7_sandbox.fixtures where name='p1';
 begin
   insert into public.research_snapshots(
     snapshot_id,dossier_id,issuer_id,security_id,report_id,execution_mode,
     data_cutoff,calculation_date,report_version,method_version,
     calculation_version,evidence_ledger_version,contract_version,schema_version,canonical_payload)
   values(
     '93000000-0000-4000-8000-000000000199',f.dossier_id,f.issuer_id,f.security_id,
     'illegal-version-pair','ANALYZE','2026-09-18','2026-09-18',
     'r3','m3','c3','e3','04_SCREENER_SCHEMA_V3_1','2.0.0',
     jsonb_set(f.payload,'{snapshot_id}','"93000000-0000-4000-8000-000000000199"')
   );
 exception when check_violation then rejected := true;
 end;
 if not rejected then raise exception 'D7 mixed V3.1/2.0.0 physical schema pair was accepted'; end if;
 if exists(select 1 from public.research_snapshots where snapshot_id='93000000-0000-4000-8000-000000000199')
 then raise exception 'D7 physical schema firewall persisted an invalid row'; end if;
end;
$physical$;

grant select on d7_sandbox.fixtures to service_role;

-- Used only by this harness; unlike a client-supplied PASS, proofs are
-- inserted by the isolated postgres superuser and pinned to exact payload SHA.
create function d7_sandbox.expect_rejection(
 p_label text, p_sqlstate text, p_run uuid, p_dossier uuid,
 p_expected uuid, p_payload jsonb
) returns void language plpgsql security invoker
set search_path=pg_catalog,d7_sandbox
as $fn$
declare accepted boolean := false;
begin
 begin
   perform d7_sandbox.persist_v31_candidate(p_run,p_dossier,p_expected,p_payload);
   accepted := true;
 exception when others then
   if SQLSTATE <> p_sqlstate then
     raise exception 'D7 %: expected SQLSTATE %, got % (%).',p_label,p_sqlstate,SQLSTATE,SQLERRM;
   end if;
 end;
 if accepted then raise exception 'D7 % unexpectedly accepted',p_label; end if;
end;
$fn$;
revoke all on function d7_sandbox.expect_rejection(text,text,uuid,uuid,uuid,jsonb)
 from public,anon,authenticated;
grant execute on function d7_sandbox.expect_rejection(text,text,uuid,uuid,uuid,jsonb)
 to service_role;

set role service_role;
do $cases$
declare
 f record;
 f2 record;
 f3 record;
 bad record;
 bad_ev record;
 pointer uuid;
 before_count bigint;
 rejected_v2 boolean;
 resp jsonb;
 foreign_issuer uuid;
begin
 select * into f from d7_sandbox.fixtures where name='p1';
 select * into f2 from d7_sandbox.fixtures where name='p2';
 select * into f3 from d7_sandbox.fixtures where name='p3';
 select * into bad from d7_sandbox.fixtures where name='bad_security';
 select * into bad_ev from d7_sandbox.fixtures where name='bad_ev';
 if f.dossier_id is null or bad.security_id is null then raise exception 'D7 missing fixtures'; end if;

 -- Before any positive flow, all unauthorized routes must refuse.
 perform d7_sandbox.expect_rejection('missing attest','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{snapshot_id}','"93000000-0000-4000-8000-000000000188"'));
 perform d7_sandbox.expect_rejection('forged PASS','23514',f.run_id,f.dossier_id,null,
   f.payload::jsonb||'{"I2_PASS":true,"I3B_PASS":true,"GO_PUBLISH":true}'::jsonb);
 perform d7_sandbox.expect_rejection('wrong run proof','23514','b1000000-0000-4000-8000-000000000001',f.dossier_id,null,f.payload);
 perform d7_sandbox.expect_rejection('wrong FCFF EV bridge','23514',bad_ev.run_id,bad_ev.dossier_id,null,bad_ev.payload);
 perform d7_sandbox.expect_rejection('missing overlay','23514',f.run_id,f.dossier_id,null,(f.payload::jsonb - 'v3_methodology'));
 perform d7_sandbox.expect_rejection('wrong issuer','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{issuer_id}','"b1000000-0000-4000-8000-000000000002"'));
 perform d7_sandbox.expect_rejection('incorrect reference date','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,valuation_timing,reference_price_date}','"2026-09-17"'));
 perform d7_sandbox.expect_rejection('future reference date','23514',f.run_id,f.dossier_id,null,
   jsonb_set(jsonb_set(f.payload,'{data_lock,reference_price_date}','"2026-09-19"'),
     '{v3_methodology,valuation_timing,reference_price_date}','"2026-09-19"'));
 perform d7_sandbox.expect_rejection('stale denominator','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,economic_share_count,share_count_as_of_date}','"2026-09-17"'));
 perform d7_sandbox.expect_rejection('negative count','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,economic_share_count,exact_count}','0'::jsonb));
 perform d7_sandbox.expect_rejection('hidden bound','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,economic_share_count,lower_bound}','1'::jsonb,true));
 perform d7_sandbox.expect_rejection('unknown denominator','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,economic_share_count,representation}','"UNKNOWN"'));
 perform d7_sandbox.expect_rejection('reversed bounded range','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,economic_share_count}',
     (f.payload::jsonb->'v3_methodology'->'economic_share_count' - 'exact_count')
     || '{"representation":"BOUNDED","propagation_status":"RANGE_PROPAGATED","lower_bound":200,"upper_bound":100,"bound_completeness":"COMPLETE","unbounded_movement_classes":0}'::jsonb));

 perform d7_sandbox.expect_rejection('date tuple misaligned','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,valuation_timing,per_share_output_date}','"2026-09-17"'));
 perform d7_sandbox.expect_rejection('market cap date mismatch','23514',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{v3_methodology,valuation_timing,market_cap_date}','"2026-09-17"'));

 before_count:=(select count(*) from public.research_snapshots);
 resp:=d7_sandbox.persist_v31_candidate(f.run_id,f.dossier_id,null,f.payload);
 if resp->>'status'<>'INSERTED_SANDBOX' then raise exception 'D7 T01 sandbox baseline did not insert'; end if;
 pointer:=(select current_snapshot_id from public.research_dossiers where dossier_id=f.dossier_id);
 if pointer is distinct from (f.payload->>'snapshot_id')::uuid then raise exception 'D7 T01 pointer did not advance'; end if;
 resp:=d7_sandbox.persist_v31_candidate(f.run_id,f.dossier_id,null,f.payload);
 if resp->>'status'<>'IDEMPOTENT_SANDBOX' then raise exception 'D7 idempotent replay failed'; end if;
 if (select count(*) from public.research_snapshots)<>before_count+1 then raise exception 'D7 identical retry inserted duplicate'; end if;

 perform d7_sandbox.expect_rejection('divergent ID replay','23505',f.run_id,f.dossier_id,null,
   jsonb_set(f.payload,'{report_id}','"forged-same-id"'));
 perform d7_sandbox.expect_rejection('stale expected CAS','40001',f2.run_id,f2.dossier_id,null,f2.payload);
 resp:=d7_sandbox.persist_v31_candidate(f2.run_id,f2.dossier_id,pointer,f2.payload);
 if resp->>'status'<>'INSERTED_SANDBOX' then raise exception 'D7 T02 successor failed'; end if;
 resp:=d7_sandbox.persist_v31_candidate(f2.run_id,f2.dossier_id,pointer,f2.payload);
 if resp->>'status'<>'IDEMPOTENT_SANDBOX' then raise exception 'D7 T02 idempotence failed'; end if;

 perform d7_sandbox.expect_rejection('stale historical replay','23505',f.run_id,f.dossier_id,null,f.payload);
 pointer:=(select current_snapshot_id from public.research_dossiers where dossier_id=f.dossier_id);
 perform d7_sandbox.expect_rejection('foreign issuer/security FK','23503',bad.run_id,bad.dossier_id,pointer,bad.payload);
 rejected_v2 := false;
 begin
   -- The V2 writer must not silently downgrade a V3.1 payload into V2.
   perform public.persist_orotitan_research_snapshot_v2(
     f3.dossier_id,pointer,f3.payload);
 exception when check_violation then rejected_v2 := true;
 end;
 if not rejected_v2 then raise exception 'D7 V2 writer accepted hidden V3.1 overlay'; end if;
 if (select count(*) from public.research_snapshots)<>before_count+2 then raise exception 'D7 rejected cases inserted rows'; end if;
 if (select count(*) from public.research_snapshots where contract_version='04_SCREENER_SCHEMA_V3_1')<>2
 then raise exception 'D7 sandbox V3.1 row count incorrect'; end if;

 -- V1/V2 writers remain physically registered and historical versions intact.
 if not exists (select 1 from public.research_snapshots
   where snapshot_id=(f.payload->>'snapshot_id')::uuid
     and contract_version='04_SCREENER_SCHEMA_V3_1' and schema_version='3.1.0') then
   raise exception 'D7 contract/schema V3.1 test pair invalid';
 end if;
end;
$cases$;
reset role;

-- Triggered pointer failure must roll back the snapshot insert atomically.
create function d7_sandbox.force_pointer_failure() returns trigger
language plpgsql set search_path=pg_catalog,d7_sandbox
as $$ begin raise exception 'D7_TEST_FORCED_ROLLBACK' using errcode='P0001'; end; $$;
create trigger d7_sandbox_pointer_fail
before update on public.research_dossiers
for each row execute function d7_sandbox.force_pointer_failure();

set role service_role;
do $rollback$
declare
 f record;
 before_count bigint;
 current_pointer uuid;
begin
 select * into f from d7_sandbox.fixtures where name='p3';
 before_count:=(select count(*) from public.research_snapshots);
 current_pointer:=(select current_snapshot_id from public.research_dossiers where dossier_id=f.dossier_id);
 perform d7_sandbox.expect_rejection('atomic pointer rollback','P0001',f.run_id,f.dossier_id,current_pointer,f.payload);
 if (select count(*) from public.research_snapshots)<>before_count then raise exception 'D7 failed transaction inserted orphan snapshot'; end if;
 if (select current_snapshot_id from public.research_dossiers where dossier_id=f.dossier_id)
    is distinct from current_pointer then raise exception 'D7 failed transaction altered pointer'; end if;
end;
$rollback$;
reset role;
drop trigger d7_sandbox_pointer_fail on public.research_dossiers;
drop function d7_sandbox.force_pointer_failure();

select json_build_object(
 'result','D7_LOCAL_SANDBOX_TESTS_PASS',
 'v31_public_writer_installed',to_regprocedure('public.persist_orotitan_research_snapshot_v3_1(uuid,uuid,jsonb)') is not null,
 'v1_writer_preserved',to_regprocedure('public.persist_orotitan_research_snapshot(uuid,uuid,jsonb)') is not null,
 'v2_writer_preserved',to_regprocedure('public.persist_orotitan_research_snapshot_v2(uuid,uuid,jsonb)') is not null,
 'sandbox_snapshot_count',(select count(*) from public.research_snapshots)
);
