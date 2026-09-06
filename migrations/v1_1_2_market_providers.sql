-- OroTitan V1.1.2 market-provider normalization.
-- Yahoo Finance returns Baltic Classifieds BCG.L directly in major GBP units.
-- No snapshot is modified. Only company market metadata is corrected.
update public.companies
set
  currency = 'GBP',
  quote_unit = 'MAJOR',
  price_decimals = 3,
  market_data_multiplier = 1
where slug = 'baltic-classifieds'
  and market_data_symbol = 'BCG:LSE';
