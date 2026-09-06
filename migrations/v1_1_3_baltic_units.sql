-- OroTitan V1.1.3 corrective migration for databases that already ran V1.1.2.
-- Yahoo Finance BCG.L is already returned in major GBP units, so no x0.01 normalization is required.
-- No snapshot and no historical market price is modified.
update public.companies
set
  currency = 'GBP',
  quote_unit = 'MAJOR',
  price_decimals = 3,
  market_data_multiplier = 1
where slug = 'baltic-classifieds'
  and market_data_symbol = 'BCG:LSE'
  and market_data_multiplier = 0.01;
