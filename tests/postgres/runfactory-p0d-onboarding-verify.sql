\set ON_ERROR_STOP on
-- PostgreSQL integration, disposable fixture only.
do $matrix$
declare
 e jsonb:=jsonb_build_object('issuer_source_url','https://example.com/issuer/source',
  'listing_source_url','https://example.org/exchange/listing',
  'method','OPERATOR_PRIMARY_SOURCE_REVIEW',
  'reviewed_by','ADMIN_SESSION_MANUAL_ATTESTATION_V1',
  'observed_at','2026-10-31T12:00:00Z');
 a jsonb;
 b jsonb;
 issuer uuid;
 security uuid;
 dossier uuid;
begin
 -- T1: one atomic CREATE of issuer/security/dossier, no run/stage/publication.
 a:=public.orotitan_onboard_identity_shadow_v1(
 'orotitan:p0d:test:loreal:initial','FR0000120321','OR','XPAR',
 'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
 null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
 if a->>'outcome'<>'CREATED' or a->>'replayed'<>'false' or
    a->>'run_created'<>'false' or a->>'registry_stage_advanced'<>'false' or
    a->>'source_automatically_verified'<>'false' then
  raise exception 'T1 created response failure: %',a;
 end if;
 issuer:=(a->>'issuer_id')::uuid;
 security:=(a->>'security_id')::uuid;
 dossier:=(a->>'dossier_id')::uuid;
 if (select count(*) from public.issuers where issuer_id=issuer)<>1 or
    (select count(*) from public.securities where security_id=security)<>1 or
    (select count(*) from public.research_dossiers
     where dossier_id=dossier and active and current_snapshot_id is null)<>1 then
  raise exception 'T1 identity persistence incorrect';
 end if;
 -- T2: exact same key and payload -> no new rows, stable IDs.
 b:=public.orotitan_onboard_identity_shadow_v1(
 'orotitan:p0d:test:loreal:initial','FR0000120321','OR','XPAR',
 'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
 null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
 if b->>'outcome'<>'REPLAYED' or b->>'issuer_id'<>issuer::text or
    b->>'security_id'<>security::text or b->>'dossier_id'<>dossier::text then
  raise exception 'T2 replay inconsistent';
 end if;
 -- T3: a second key for the same canonical listing reuses identity.
 b:=public.orotitan_onboard_identity_shadow_v1(
 'orotitan:p0d:test:loreal:second-key','FR0000120321','OR','XPAR',
 'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
 null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
 if b->>'outcome'<>'REUSED' or b->>'security_id'<>security::text then
  raise exception 'T3 reuse failed';
 end if;
 -- T4: same idempotency key changed data must fail closed.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:initial','FR0000120321','OR','XPAR',
   'L’Oréal','L’Oréal Changed SA','FR','EUR','EUR',true,'MAJOR',2,
   null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T4';
 exception when others then
  if sqlerrm<>'OROTITAN_ONBOARDING_IDEMPOTENCY_CONFLICT' then raise; end if;
 end;
 -- T5: same listing, different ISIN must fail closed.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:conflict','NL0010273215','OR','XPAR',
   'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
   null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T5';
 exception when others then
  if sqlerrm<>'OROTITAN_ONBOARDING_LISTING_ISIN_CONFLICT' then raise; end if;
 end;
 -- T6: cross-listing same ISIN, same issuer, non-primary security accepted.
 b:=public.orotitan_onboard_identity_shadow_v1(
 'orotitan:p0d:test:loreal:crosslisting','FR0000120321','ORL','XAMS',
 'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',false,'MAJOR',2,
 null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
 if b->>'outcome'<>'LISTING_ADDED' or b->>'issuer_id'<>issuer::text or
    b->>'dossier_id'<>dossier::text or
    (select count(*) from public.securities where issuer_id=issuer)<>2 then
  raise exception 'T6 legitimate cross-listing failed: %',b;
 end if;
 -- T7: cannot create a second primary listing for same issuer.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:secondprimary','FR0000120321','ORL2','XBRU',
   'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
   null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T7';
 exception when others then
  if sqlerrm<>'OROTITAN_ONBOARDING_PRIMARY_LISTING_CONFLICT' then raise; end if;
 end;
 -- T8: currency mismatch on reused security must not be rewritten.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:badcurrency','FR0000120321','OR','XPAR',
   'L’Oréal','L’Oréal S.A.','FR','EUR','USD',true,'MAJOR',2,
   null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T8';
 exception when others then
  if sqlerrm<>'OROTITAN_ONBOARDING_EXISTING_SECURITY_METADATA_CONFLICT'
   then raise; end if;
 end;
 -- T9: invalid checksum independently caught by PLpgSQL.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:badchecksum','FR0000120322','BAD','XPAR',
   'Example','Example SA','FR','EUR','EUR',true,'MAJOR',2,
   null,null,e,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T9';
 exception when others then
  if sqlerrm<>'OROTITAN_ISIN_CHECK_DIGIT_INVALID' then raise; end if;
 end;
 -- T10: no operator attestation.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:noapproval','FR0000120321','OR','XPAR',
   'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
   null,null,e,'NO');
  raise exception 'MISSING_T10';
 exception when others then
  if sqlerrm<>'OROTITAN_ONBOARDING_ATTESTATION_REQUIRED' then raise; end if;
 end;
 -- T11: supply a missing primary source.
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:loreal:source','FR0000120321','OR','XPAR',
   'L’Oréal','L’Oréal S.A.','FR','EUR','EUR',true,'MAJOR',2,
   null,null,e-'listing_source_url','APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T11';
 exception when others then
  if sqlerrm<>'OROTITAN_PRIMARY_SOURCE_ATTESTATION_INCOMPLETE'
   then raise; end if;
 end;
 if (select count(*) from public.issuers where issuer_id=issuer)<>1 or
    (select count(*) from public.research_dossiers where issuer_id=issuer)<>1 or
    (select count(*) from public.securities where issuer_id=issuer)<>2 or
    (select count(*) from public.orotitan_identity_onboarding_receipts_v1
     where issuer_id=issuer)<>3 then
  raise exception 'T12 tests caused unexpected mutation';
 end if;
 -- T13: role separation and RLS.
 if pg_catalog.has_function_privilege('anon',
  'public.orotitan_onboard_identity_shadow_v1(text,text,text,text,text,text,text,text,text,boolean,text,integer,uuid,uuid,jsonb,text)',
  'EXECUTE') or
    pg_catalog.has_function_privilege('authenticated',
  'public.orotitan_onboard_identity_shadow_v1(text,text,text,text,text,text,text,text,text,boolean,text,integer,uuid,uuid,jsonb,text)',
  'EXECUTE') or not
    pg_catalog.has_function_privilege('service_role',
  'public.orotitan_onboard_identity_shadow_v1(text,text,text,text,text,text,text,text,text,boolean,text,integer,uuid,uuid,jsonb,text)',
  'EXECUTE') or not
    (select relrowsecurity from pg_class where oid=
     'public.orotitan_identity_onboarding_receipts_v1'::regclass) then
  raise exception 'T13 access control / RLS failure';
 end if;
 raise notice 'P0D T1-T13 PASS: created, replayed, reused, crosslisted, collisions/privileges guarded';
end $matrix$;

-- T14: atomic rollback when a legacy crosswalk cannot be reconciled.
do $matrix$
declare
 e jsonb:=jsonb_build_object('issuer_source_url','https://example.com/issuer/source',
  'listing_source_url','https://example.org/exchange/listing',
  'method','OPERATOR_PRIMARY_SOURCE_REVIEW',
  'reviewed_by','ADMIN_SESSION_MANUAL_ATTESTATION_V1',
  'observed_at','2026-10-31T12:00:00Z');
 b integer;
begin
 select count(*) into b from public.issuers;
 begin
  perform public.orotitan_onboard_identity_shadow_v1(
   'orotitan:p0d:test:legacy:rollback','NL0010273215','WRONG','XPAR',
   'Rollback Fixture','Rollback Fixture SA','FR','EUR','EUR',true,'MAJOR',2,
   '00000000-0000-4000-8000-000000000002',null,e,
   'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
  raise exception 'MISSING_T14';
 exception when others then
  if sqlerrm<>'OROTITAN_ONBOARDING_LEGACY_IDENTITY_MISMATCH' then raise; end if;
 end;
 if (select count(*) from public.issuers)<>b or
    exists (select 1 from public.securities where ticker='WRONG') or
    exists (select 1 from public.orotitan_identity_onboarding_receipts_v1
     where request_key='orotitan:p0d:test:legacy:rollback') then
   raise exception 'T14 transaction rollback failure';
 end if;
 -- T15: one correct legacy crosswalk persisted atomically.
 perform public.orotitan_onboard_identity_shadow_v1(
  'orotitan:p0d:test:legacy:correct','NL0010273215','BET','XPAR',
  'Beta Display','Beta Display SA','FR','EUR','EUR',true,'MAJOR',2,
  '00000000-0000-4000-8000-000000000002',null,e,
  'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
 if (select count(*) from public.legacy_company_identity_map where
  legacy_company_id='00000000-0000-4000-8000-000000000002')<>1 then
  raise exception 'T15 crosswalk not created';
 end if;
 raise notice 'P0D T14-T15 PASS: atomic rollback and legacy crosswalk';
end $matrix$;
