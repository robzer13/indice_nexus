import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';

export interface MarketDataCompany {
  id: string;
  slug: string;
  name: string;
  market_data_symbol: string;
  market_data_multiplier: number;
}

export async function getMarketDataCompanies(): Promise<MarketDataCompany[]> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('companies')
    .select('id,slug,name,market_data_symbol,market_data_multiplier')
    .eq('active', true)
    .not('market_data_symbol', 'is', null)
    .order('name');
  if (error) throw new Error(`Unable to load market-data companies: ${error.message}`);
  return (data ?? []).filter((row) => row.market_data_symbol) as MarketDataCompany[];
}

export async function insertMarketPrice(input: {
  companyId: string;
  price: number;
  asOf: string;
  source: string;
  raw: unknown;
}): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { error } = await supabase.from('market_prices').insert({
    company_id: input.companyId,
    price: input.price,
    as_of: input.asOf,
    source: input.source,
    raw: input.raw,
  });
  if (error) throw new Error(`Unable to insert market price: ${error.message}`);
}
