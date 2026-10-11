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
