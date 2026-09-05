import { PriceDisplay } from '@/components/price-display';
import type { CompanyState } from '@/lib/domain/types';

export function FairValueRange({ company }: { company: CompanyState }) {
  const props = { currency: company.currency, quoteUnit: company.quote_unit, priceDecimals: company.price_decimals };
  return <div className="grid gap-3 sm:grid-cols-3"><div className="rounded-lg border border-slate-800 p-3"><div className="text-xs text-slate-500">FV basse</div><div className="mt-1 text-slate-200"><PriceDisplay value={company.fair_value_low} {...props} /></div></div><div className="rounded-lg border border-cyan-900/50 bg-cyan-950/20 p-3"><div className="text-xs text-cyan-400">FV centrale</div><div className="mt-1 text-lg font-semibold text-white"><PriceDisplay value={company.fair_value_base} {...props} /></div></div><div className="rounded-lg border border-slate-800 p-3"><div className="text-xs text-slate-500">FV haute</div><div className="mt-1 text-slate-200"><PriceDisplay value={company.fair_value_high} {...props} /></div></div></div>;
}
