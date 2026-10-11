-- STAGED ONLY. Do not apply to production without a separately authorized GO.
-- Exact canonical quote-identity backfill for the 2026-10-11 audited cohort.
-- No issuer, security, dossier, research snapshot or market price is modified.
begin;

create temporary table _canonical_quote_identity_candidates on commit drop as
select c.id as legacy_company_id, d.issuer_id, s.security_id, d.dossier_id
from public.research_dossiers d
join public.research_snapshots s
  on s.snapshot_id = d.current_snapshot_id
 and s.dossier_id = d.dossier_id and s.issuer_id = d.issuer_id
join public.securities sec
  on sec.security_id = s.security_id and sec.issuer_id = s.issuer_id
join public.companies c
  on c.id = d.issuer_id
 and c.active is true
 and c.ticker = sec.ticker
 and c.exchange = sec.exchange
 and c.currency = sec.trading_currency
 and c.quote_unit = sec.quote_unit
 and c.market_data_symbol = sec.market_data_symbol
 and c.market_data_symbol is not null
 and c.market_data_multiplier = sec.market_data_multiplier
 and c.price_decimals = sec.price_decimals
where d.active is true and d.current_snapshot_id is not null;

-- Pinned to the exact 32 quadruplets audited in production on 2026-10-11.
-- A same-size substitution (including a newly matching Wise or ISRG row)
-- is NOT permitted. Changing this allowlist requires a separate audit/GO.
create temporary table _canonical_quote_audited_cohort on commit drop as
select audited.legacy_company_id, audited.issuer_id, audited.security_id, audited.dossier_id
from (values
  ('0a0c53bb-ac99-4d6a-8169-a00d60ee66b1'::uuid, '0a0c53bb-ac99-4d6a-8169-a00d60ee66b1'::uuid, '9fa42e6f-384d-4d78-b30f-f66590913d2d'::uuid, '14ad94ae-5605-46ad-a4b4-afeca7452743'::uuid),
  ('1fb5ee9e-9fc2-4a0e-8685-e0ecc3057716'::uuid, '1fb5ee9e-9fc2-4a0e-8685-e0ecc3057716'::uuid, '6d82977c-f4bd-4ac2-a2ba-227abb640471'::uuid, '468bc508-2a36-4005-88b4-e5e6c12f61af'::uuid),
  ('26944490-eee2-4061-874e-3adc99ad1bb8'::uuid, '26944490-eee2-4061-874e-3adc99ad1bb8'::uuid, '9587e2df-41e7-430d-b48f-51a45270ea91'::uuid, '2bf366bd-85a5-47ad-89fa-9de7b14f5210'::uuid),
  ('2a7e0122-825c-46c1-b423-8bd9c185eb81'::uuid, '2a7e0122-825c-46c1-b423-8bd9c185eb81'::uuid, '1dec8fb7-9acd-4dac-9b4a-9b43290c79b5'::uuid, 'c55dbd62-1ab7-49c7-9a2b-be7cb65984e9'::uuid),
  ('2bd2641f-67e8-438d-addc-5c9f8a1f6163'::uuid, '2bd2641f-67e8-438d-addc-5c9f8a1f6163'::uuid, '2202c604-24d9-2737-b8dd-7b192de1fc11'::uuid, '3e26750a-aedc-bd4f-c5a5-843c22bfe7db'::uuid),
  ('2bf74c6c-574b-4c80-bc82-eaca05bef1dd'::uuid, '2bf74c6c-574b-4c80-bc82-eaca05bef1dd'::uuid, '894f452d-703f-4d79-a3a5-1bc25aafb304'::uuid, '59ea049c-c356-4dbc-80ee-ec12cd266dfc'::uuid),
  ('2e696ee6-f538-4db1-8c8a-1d282b1e2861'::uuid, '2e696ee6-f538-4db1-8c8a-1d282b1e2861'::uuid, '27d40e90-21d4-4a99-8cd7-a13bb40e90a7'::uuid, '2f820dc7-62bc-4452-8b9d-2447d9c72871'::uuid),
  ('321bae14-d7a8-40d4-9863-203e10217112'::uuid, '321bae14-d7a8-40d4-9863-203e10217112'::uuid, '7d31c334-aa2b-66d2-11ed-9e0c4fb6e1dc'::uuid, 'f3fa5883-307d-a78f-64e4-02a49879bf44'::uuid),
  ('386ecaaf-a9a9-4653-9ba5-2333b8934700'::uuid, '386ecaaf-a9a9-4653-9ba5-2333b8934700'::uuid, 'aee0ff49-412a-4af3-b402-b1b0ccd023d6'::uuid, 'adf00fa1-d954-4a4f-9898-7dc6e2fd3a4a'::uuid),
  ('386ed207-8bc6-4ad7-8ced-7d8f107f47ea'::uuid, '386ed207-8bc6-4ad7-8ced-7d8f107f47ea'::uuid, 'b127e1fc-ecff-4f4e-ac32-35f74663da3f'::uuid, '212b4abe-8e48-4149-898e-eabbebe8e1c4'::uuid),
  ('3f61c86b-cf13-468c-b175-59d61d1161e7'::uuid, '3f61c86b-cf13-468c-b175-59d61d1161e7'::uuid, '766fac58-96a3-407a-9302-1484abff41f5'::uuid, '8b55eb92-912b-4941-96d1-3b2eb9602666'::uuid),
  ('46dc987d-7f9e-4125-a516-c781ab49e1a0'::uuid, '46dc987d-7f9e-4125-a516-c781ab49e1a0'::uuid, '076617ff-e951-4cdc-8782-9d5e0a4392bf'::uuid, '25fd991f-f4d2-4478-a374-55f12fc0dcc1'::uuid),
  ('4beab7cb-26b1-464b-af08-b7ef343aebc6'::uuid, '4beab7cb-26b1-464b-af08-b7ef343aebc6'::uuid, '6b91c58a-8f54-4a4b-9727-446e7800363d'::uuid, 'b0674e4a-e22a-494a-9626-87b1c6499dd5'::uuid),
  ('508aa42e-8cb7-4806-870d-972c39a8df2c'::uuid, '508aa42e-8cb7-4806-870d-972c39a8df2c'::uuid, 'f14244b2-581d-4040-8a7d-c11bf55114ae'::uuid, 'b6c8ec7f-4d82-4927-83b3-cd007e638054'::uuid),
  ('516ab04b-f3e5-4853-88ad-a386b7c2dab6'::uuid, '516ab04b-f3e5-4853-88ad-a386b7c2dab6'::uuid, '925f910f-08fb-4431-bf54-81b07295fe8c'::uuid, 'e1c503d6-c2e2-4fc3-abb4-73bcc9698646'::uuid),
  ('5250363b-3274-4772-81d4-7599c8e87caf'::uuid, '5250363b-3274-4772-81d4-7599c8e87caf'::uuid, '2f474fa7-0cc0-4f72-8ab9-7432f28f251d'::uuid, '1ed7d3ab-0c10-44d1-87c9-31d368cf3f38'::uuid),
  ('68332633-0f03-426d-801f-8923219407dd'::uuid, '68332633-0f03-426d-801f-8923219407dd'::uuid, '247e7f79-7a34-4e3d-9542-c217274526e9'::uuid, 'c7277756-d903-4fea-9c0f-2305364f4156'::uuid),
  ('722258ca-f70b-4146-83e3-ad2aff98737d'::uuid, '722258ca-f70b-4146-83e3-ad2aff98737d'::uuid, '69641ba2-5138-4c3e-994e-4ef1fc2ec8b5'::uuid, '1bad37ef-d7a9-4383-b8ef-adf4a60772a1'::uuid),
  ('72ffd040-20e0-4287-bd59-5246e71db7b9'::uuid, '72ffd040-20e0-4287-bd59-5246e71db7b9'::uuid, 'b522fe6b-9901-49a5-b3da-42b93a4ec8de'::uuid, '58ae1917-d094-4533-b8eb-1848e191b87a'::uuid),
  ('76060379-e9bf-4182-948c-f7875efb6683'::uuid, '76060379-e9bf-4182-948c-f7875efb6683'::uuid, '5497554e-6f38-4ce0-9800-9a9fc74c7973'::uuid, '5c731282-1cc6-437a-b3ad-410d902d8808'::uuid),
  ('7a8cd7ef-8623-4ddd-9074-5b0fd81787ed'::uuid, '7a8cd7ef-8623-4ddd-9074-5b0fd81787ed'::uuid, '05e580c2-96e5-47d3-812a-2d251eac29bd'::uuid, 'c1cc5b24-6e32-498e-91f3-d85654c6d39e'::uuid),
  ('7ae72c78-92a7-495d-abaf-926096a926fb'::uuid, '7ae72c78-92a7-495d-abaf-926096a926fb'::uuid, '84b71262-fb37-4198-90f0-9d85f4442e44'::uuid, '40b47c57-225d-4a5c-911d-6d0b1c750d70'::uuid),
  ('88a380bc-4dfc-4291-b03b-932a8e232872'::uuid, '88a380bc-4dfc-4291-b03b-932a8e232872'::uuid, '5ff7e189-30f1-478d-8985-00b4c2dd79b5'::uuid, 'c9e1bcee-842e-4441-bc77-53fb7f6955ab'::uuid),
  ('91ca71c1-137c-4935-a75e-b38b9688cb4d'::uuid, '91ca71c1-137c-4935-a75e-b38b9688cb4d'::uuid, 'c2bf1235-4ed6-4f10-ac47-eb6853f261ef'::uuid, 'd7c0f86d-68ba-4bb0-9002-bd4195f28af0'::uuid),
  ('9a88dc72-5c41-416d-b0e0-b92d0e5973cf'::uuid, '9a88dc72-5c41-416d-b0e0-b92d0e5973cf'::uuid, '951b0718-d3fb-4ce1-be06-b9bd3810ee50'::uuid, 'ac768009-3e48-4067-9b6d-bbaf60f74900'::uuid),
  ('9ba202e1-7434-4988-884f-6b7489121728'::uuid, '9ba202e1-7434-4988-884f-6b7489121728'::uuid, '6d44adba-a1a7-4ac2-8283-ae73d3b0f377'::uuid, 'b3074408-8741-4abf-8451-3b90e29eab17'::uuid),
  ('a0d2b534-e9a2-4093-9ab2-58d6ea6b31c5'::uuid, 'a0d2b534-e9a2-4093-9ab2-58d6ea6b31c5'::uuid, 'b77eedc2-13f6-4faf-a4fc-17e09e10ff05'::uuid, 'e47a0165-9d80-4d5e-9279-db206846afac'::uuid),
  ('ac496408-bbd0-4bfa-bcda-443e2bfd8f97'::uuid, 'ac496408-bbd0-4bfa-bcda-443e2bfd8f97'::uuid, 'bd6c9473-bfa4-41fa-9022-b8787bfedb00'::uuid, '263c824c-4b46-45fe-aa63-c18fffecc49f'::uuid),
  ('bfd7be07-7a82-4f78-a6a1-d04e6a4d9454'::uuid, 'bfd7be07-7a82-4f78-a6a1-d04e6a4d9454'::uuid, '28174abd-d2e7-415d-8491-a8ee07325e4d'::uuid, 'a8b77e70-4179-4da7-b70c-63c545fec1bc'::uuid),
  ('d37ccfaa-4dad-47c3-9b44-a752522cd7d5'::uuid, 'd37ccfaa-4dad-47c3-9b44-a752522cd7d5'::uuid, '5ca6f4a2-a796-55fd-58f8-0798a4b04fbe'::uuid, '17f57723-a218-8327-841a-04c5f5513174'::uuid),
  ('f871b24a-9f44-4773-a268-4fc6a1f02400'::uuid, 'f871b24a-9f44-4773-a268-4fc6a1f02400'::uuid, 'e15698ca-050b-4c70-ab7a-2bc3ef5f58a4'::uuid, 'dcffe010-2fc2-4f3f-8bd4-97e5d612c2ee'::uuid),
  ('fc521b1d-9462-4060-9eee-c958726cef1b'::uuid, 'fc521b1d-9462-4060-9eee-c958726cef1b'::uuid, '7006178d-1704-4c62-b0dd-971f3407016a'::uuid, 'ab46d1b4-4576-4264-8416-0fe59bdefdc5'::uuid)
) as audited(legacy_company_id, issuer_id, security_id, dossier_id);


do $$
begin
  -- Freeze the audited cohort. Drift must be reviewed, never guessed.
  if (select count(*) from public.research_dossiers
      where active is true and current_snapshot_id is not null) <> 34 then
    raise exception 'CANONICAL_QUOTE_COHORT_DRIFT';
  end if;
  if (select count(*) from _canonical_quote_identity_candidates) <> 32 then
    raise exception 'CANONICAL_QUOTE_CANDIDATE_COUNT_DRIFT';
  end if;
  -- Cardinality alone does not prove the approved identity cohort.
  -- FULL OUTER JOIN of all four UUIDs rejects dropped, substituted,
  -- newly matching, or moved issuer/security/dossier identities.
  if (select count(*) from _canonical_quote_audited_cohort) <> 32 or exists (
    select 1
    from _canonical_quote_identity_candidates x
    full outer join _canonical_quote_audited_cohort audited
      on audited.legacy_company_id = x.legacy_company_id
     and audited.issuer_id = x.issuer_id
     and audited.security_id = x.security_id
     and audited.dossier_id = x.dossier_id
    where x.legacy_company_id is null or audited.legacy_company_id is null
  ) then
    raise exception 'CANONICAL_QUOTE_AUDITED_IDENTITY_DRIFT';
  end if;
  if exists (
    select 1 from _canonical_quote_identity_candidates
    group by legacy_company_id having count(*) <> 1
  ) or exists (
    select 1 from _canonical_quote_identity_candidates
    group by security_id having count(*) <> 1
  ) or exists (
    select 1 from _canonical_quote_identity_candidates
    group by dossier_id having count(*) <> 1
  ) then
    raise exception 'CANONICAL_QUOTE_AMBIGUOUS_IDENTITY';
  end if;
  -- Priority holdings must have an exact one-to-one candidate identity.
  if not exists (
    select 1 from _canonical_quote_identity_candidates x
    join public.securities sec on sec.security_id=x.security_id
    where sec.ticker='MUV2' and sec.exchange='Xetra'
      and sec.trading_currency='EUR' and sec.market_data_symbol='MUV2:XETR'
  ) or not exists (
    select 1 from _canonical_quote_identity_candidates x
    join public.securities sec on sec.security_id=x.security_id
    where sec.ticker='RI' and sec.exchange='Euronext Paris'
      and sec.trading_currency='EUR' and sec.market_data_symbol='RI:EPA'
  ) then
    raise exception 'CANONICAL_QUOTE_PRIORITY_IDENTITY_UNPROVEN';
  end if;
  -- Fail closed on existing crosswalk ownership conflicts in either direction.
  if exists (
    select 1 from _canonical_quote_identity_candidates x
    join public.legacy_company_identity_map m
      on m.legacy_company_id=x.legacy_company_id
      or m.security_id=x.security_id or m.dossier_id=x.dossier_id
    where m.legacy_company_id is distinct from x.legacy_company_id
       or m.issuer_id is distinct from x.issuer_id
       or m.security_id is distinct from x.security_id
       or m.dossier_id is distinct from x.dossier_id
  ) then
    raise exception 'CANONICAL_QUOTE_CROSSWALK_CONFLICT';
  end if;
end $$;

insert into public.legacy_company_identity_map
  (legacy_company_id, issuer_id, security_id, dossier_id)
select x.legacy_company_id,x.issuer_id,x.security_id,x.dossier_id
from _canonical_quote_identity_candidates x
where not exists (
  select 1 from public.legacy_company_identity_map m
  where m.legacy_company_id=x.legacy_company_id
);

do $$
begin
  if exists (
    select 1 from _canonical_quote_identity_candidates x
    left join public.legacy_company_identity_map m
      on m.legacy_company_id=x.legacy_company_id
     and m.issuer_id=x.issuer_id
     and m.security_id=x.security_id
     and m.dossier_id=x.dossier_id
    where m.legacy_company_id is null
  ) then
    raise exception 'CANONICAL_QUOTE_POSTCONDITION_FAILED';
  end if;
end $$;

commit;
