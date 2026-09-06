-- OroTitan V1.1.1 market-data metadata correction.
-- Twelve Data quotes domestic LSE equities such as Auto Trader directly in GBp.
update public.companies
set market_data_multiplier = 1
where slug = 'auto-trader'
  and market_data_symbol = 'AUTO:LSE'
  and market_data_multiplier = 100;
