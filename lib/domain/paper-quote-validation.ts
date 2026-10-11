import { getFreshness } from './freshness';
/** Reject future, invalid, stale and nonpositive prices from paper NAV. */
export function isAdmissiblePaperValuationQuote(
  price: number, asOf: string | null, now: Date = new Date(),
): boolean {
  const quoteTime = asOf === null ? NaN : Date.parse(asOf);
  const nowTime = now.getTime();
  return Number.isFinite(price) && price > 0
    && Number.isFinite(quoteTime) && Number.isFinite(nowTime)
    && quoteTime <= nowTime + 5 * 60_000
    && !getFreshness(asOf, now).stale;
}
