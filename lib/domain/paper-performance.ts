/**
 * Financial arithmetic for future audited NAV, FX, corporate-action and
 * benchmark imports. No data feed is assumed or fabricated here.
 */
export interface DailyNavPoint {
  date: string; // YYYY-MM-DD
  navEur: number;
  externalNetContributionEur: number; // flows invested between prior and current NAV observations
}

export function computeTimeWeightedReturn(points: DailyNavPoint[]): number | null {
  if (points.length < 2) return null;
  let multiplier = 1;
  for (let i = 0; i < points.length; i++) {
    const { date, navEur, externalNetContributionEur } = points[i];
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || !Number.isFinite(Date.parse(date+'T00:00:00Z'))
      || !Number.isFinite(navEur) || navEur <= 0 || !Number.isFinite(externalNetContributionEur))
      return null;
    if (i === 0) continue;
    const previous = points[i-1];
    if (date <= previous.date) return null;
    // Cash flows are assumed immediately before the ending observation.
    const subperiod = (navEur - externalNetContributionEur) / previous.navEur;
    if (!Number.isFinite(subperiod) || subperiod <= 0) return null;
    multiplier *= subperiod;
  }
  return multiplier - 1;
}

export function maximumDrawdown(values: number[]): number | null {
  if (!values.length || values.some(x => !Number.isFinite(x) || x <= 0)) return null;
  let peak = values[0], maximum = 0;
  for (const n of values) { peak = Math.max(peak, n); maximum = Math.max(maximum, (peak - n) / peak); }
  return maximum;
}

export interface VerifiedFxQuote {
  currency: string;
  eurPerUnit: number;
  observedAt: string;
  source: string;
}
/** Source-derived EUR/1-unit quote; caller must prove origin and temporal alignment. */
export function translateToEur(amount: number, rate: VerifiedFxQuote, valuationAsOf: string): number | null {
  const ts = Date.parse(rate.observedAt), asOf = Date.parse(valuationAsOf);
  if (!Number.isFinite(amount) || !Number.isFinite(rate.eurPerUnit) || rate.eurPerUnit <= 0
    || !/^[A-Z]{3}$/.test(rate.currency) || !rate.source.trim()
    || !Number.isFinite(ts) || !Number.isFinite(asOf) || ts > asOf || asOf - ts > 96 * 3600000) return null;
  return amount * rate.eurPerUnit;
}

export function cashDividendEur(
  eligibleShares: number, grossPerShare: number, withholdingFraction: number, fxEurPerUnit: number,
): number | null {
  if (![eligibleShares,grossPerShare,withholdingFraction,fxEurPerUnit].every(Number.isFinite)
    || eligibleShares < 0 || grossPerShare < 0 || withholdingFraction < 0 || withholdingFraction > 1
    || fxEurPerUnit <= 0) return null;
  return eligibleShares * grossPerShare * (1 - withholdingFraction) * fxEurPerUnit;
}

export function relativePerformance(portfolioTwr: number | null, benchmarkTwr: number | null): number | null {
  return portfolioTwr === null || benchmarkTwr === null ? null : portfolioTwr - benchmarkTwr;
}
