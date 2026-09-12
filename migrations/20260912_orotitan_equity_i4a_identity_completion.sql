begin;

-- I4-A canonical identity completion for the eight legacy-mapped issuers.
-- Production migration version: 20260912070836
-- Fresh-reanalysis policy: identity only. No legacy analytical data, score,
-- valuation, certification state, market-price row, or research snapshot is modified.

do $$
begin
  if (select count(*) from public.issuers) <> 8
     or (select count(*) from public.securities) <> 8
     or (select count(*) from public.research_dossiers) <> 8 then
    raise exception 'I4-A precondition failed: canonical identity counts drifted';
  end if;

  if exists (select 1 from public.research_snapshots)
     or exists (select 1 from public.research_dossiers where current_snapshot_id is not null) then
    raise exception 'I4-A precondition failed: canonical research already exists';
  end if;

  if (select count(*) from public.securities where isin = 'UNKNOWN') <> 8
     or (select count(*) from public.securities where primary_listing is null) <> 8
     or (select count(*) from public.issuers where reporting_currency is null) <> 8
     or (select count(*) from public.issuers where legal_name is null) <> 8 then
    raise exception 'I4-A precondition failed: identity fields were already partially resolved';
  end if;

  if not exists (select 1 from public.issuers where issuer_id='2bd2641f-67e8-438d-addc-5c9f8a1f6163' and display_name='RATIONAL AG')
     or not exists (select 1 from public.issuers where issuer_id='321bae14-d7a8-40d4-9863-203e10217112' and display_name='Hermès International')
     or not exists (select 1 from public.issuers where issuer_id='38ae603d-5645-4045-9ca9-d3ce909bd0d7' and display_name='SEI Investments Company')
     or not exists (select 1 from public.issuers where issuer_id='7a2a0fe0-3666-46ba-9a2c-3b44346c452a' and display_name='Auto Trader Group plc')
     or not exists (select 1 from public.issuers where issuer_id='884076df-14a7-45c1-af48-aa0f54262f7a' and display_name='Baltic Classifieds Group')
     or not exists (select 1 from public.issuers where issuer_id='9a879b0b-8957-4557-b2b8-4955d9889c37' and display_name='Medistim ASA')
     or not exists (select 1 from public.issuers where issuer_id='b2ba8f58-b968-4fc3-a7a7-6bfe5e6ae552' and display_name='Scout24 SE')
     or not exists (select 1 from public.issuers where issuer_id='d37ccfaa-4dad-47c3-9b44-a752522cd7d5' and display_name='Qualys, Inc.') then
    raise exception 'I4-A precondition failed: issuer identity mapping drifted';
  end if;
end $$;

update public.issuers i
set display_name = v.display_name,
    legal_name = v.legal_name,
    country = v.country,
    reporting_currency = v.reporting_currency,
    updated_at = now()
from (values
  ('2bd2641f-67e8-438d-addc-5c9f8a1f6163'::uuid, 'RATIONAL AG'::text, 'RATIONAL Aktiengesellschaft'::text, 'Germany'::text, 'EUR'::text),
  ('321bae14-d7a8-40d4-9863-203e10217112'::uuid, 'Hermès International', 'Hermès International', 'France', 'EUR'),
  ('38ae603d-5645-4045-9ca9-d3ce909bd0d7'::uuid, 'SEI Investments Company', 'SEI Investments Company', 'United States', 'USD'),
  ('7a2a0fe0-3666-46ba-9a2c-3b44346c452a'::uuid, 'Autotrader Group plc', 'AUTOTRADER GROUP PLC', 'United Kingdom', 'GBP'),
  ('884076df-14a7-45c1-af48-aa0f54262f7a'::uuid, 'Baltic Classifieds Group PLC', 'Baltic Classifieds Group PLC', 'United Kingdom', 'EUR'),
  ('9a879b0b-8957-4557-b2b8-4955d9889c37'::uuid, 'Medistim ASA', 'Medistim ASA', 'Norway', 'NOK'),
  ('b2ba8f58-b968-4fc3-a7a7-6bfe5e6ae552'::uuid, 'Scout24 SE', 'Scout24 SE', 'Germany', 'EUR'),
  ('d37ccfaa-4dad-47c3-9b44-a752522cd7d5'::uuid, 'Qualys, Inc.', 'Qualys, Inc.', 'United States', 'USD')
) as v(issuer_id, display_name, legal_name, country, reporting_currency)
where i.issuer_id = v.issuer_id;

update public.securities s
set isin = v.isin,
    country = v.country,
    trading_currency = v.trading_currency,
    primary_listing = true,
    updated_at = now()
from (values
  ('2202c604-24d9-2737-b8dd-7b192de1fc11'::uuid, 'DE0007010803'::text, 'Germany'::text, 'EUR'::text),
  ('7d31c334-aa2b-66d2-11ed-9e0c4fb6e1dc'::uuid, 'FR0000052292', 'France', 'EUR'),
  ('e58246d9-a7c0-1809-0490-ba2e30d67c7b'::uuid, 'US7841171033', 'United States', 'USD'),
  ('91438711-4074-2773-32f9-123162dd80f0'::uuid, 'GB00BVYVFW23', 'United Kingdom', 'GBP'),
  ('dbe64199-a1dc-2e55-d41b-b5431eb3b755'::uuid, 'GB00BN44P254', 'United Kingdom', 'EUR'),
  ('bd396734-c383-e069-2725-736bf0b3ac7a'::uuid, 'NO0010159684', 'Norway', 'NOK'),
  ('e2ddbc2c-6c65-5e95-405c-aee0e75467e6'::uuid, 'DE000A12DM80', 'Germany', 'EUR'),
  ('5ca6f4a2-a796-55fd-58f8-0798a4b04fbe'::uuid, 'US74758T3032', 'United States', 'USD')
) as v(security_id, isin, country, trading_currency)
where s.security_id = v.security_id;

do $$
begin
  if (select count(*) from public.issuers where legal_name is not null and reporting_currency is not null) <> 8 then
    raise exception 'I4-A postcondition failed: issuer completion incomplete';
  end if;

  if (select count(*) from public.securities where isin <> 'UNKNOWN' and country is not null and primary_listing is true) <> 8 then
    raise exception 'I4-A postcondition failed: security completion incomplete';
  end if;

  if not exists (
    select 1 from public.issuers i
    join public.securities s on s.issuer_id=i.issuer_id
    where i.issuer_id='7a2a0fe0-3666-46ba-9a2c-3b44346c452a'
      and i.display_name='Autotrader Group plc'
      and i.legal_name='AUTOTRADER GROUP PLC'
      and s.isin='GB00BVYVFW23'
      and s.ticker='AUTO'
      and s.trading_currency='GBP'
      and s.quote_unit='MINOR'
      and s.primary_listing is true
  ) then
    raise exception 'I4-A postcondition failed: Autotrader identity mismatch';
  end if;

  if not exists (
    select 1 from public.issuers i
    join public.securities s on s.issuer_id=i.issuer_id
    where i.issuer_id='884076df-14a7-45c1-af48-aa0f54262f7a'
      and i.display_name='Baltic Classifieds Group PLC'
      and i.country='United Kingdom'
      and i.reporting_currency='EUR'
      and s.isin='GB00BN44P254'
      and s.ticker='BCG'
      and s.trading_currency='EUR'
      and s.primary_listing is true
  ) then
    raise exception 'I4-A postcondition failed: BCG identity mismatch';
  end if;

  if exists (select 1 from public.research_snapshots)
     or exists (select 1 from public.research_dossiers where current_snapshot_id is not null) then
    raise exception 'I4-A postcondition failed: research state was modified';
  end if;
end $$;

commit;
