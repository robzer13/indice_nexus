/**
 * Pure deterministic policy for paper-only scheduled execution.
 * Execution itself is performed only by the V1.1 PostgreSQL serializable
 * portfolio lock / exactly-once RPC using a quote after the signal.
 */
export type PaperAutoPolicy = {
  enabled: boolean;
  initialCapitalEur: number;
  availableCashEur: number;
  maxInitialCapitalWeight: number;
  simulatedFeeEur: number;
};
export type PaperAutoDecision =
  | { permitted: true; quantity: number }
  | { permitted: false; reason: string };

function validAmount(value: number) {
  return Number.isFinite(value) && value >= 0;
}

/** V1.2 uses whole shares and deliberately disallows automatic reinforcement. */
export function planPaperAutoFill(
  action: string, quotePriceEur: number, heldQuantity: number, policy: PaperAutoPolicy,
): PaperAutoDecision {
  if (!policy.enabled) return { permitted: false, reason: 'AUTOMATION_DISABLED' };
  if (!Number.isFinite(quotePriceEur) || quotePriceEur <= 0)
    return { permitted: false, reason: 'INVALID_QUOTE' };
  if (!validAmount(heldQuantity) || !validAmount(policy.availableCashEur)
    || !validAmount(policy.initialCapitalEur) || policy.initialCapitalEur === 0
    || !validAmount(policy.simulatedFeeEur) || policy.simulatedFeeEur > 100
    || !Number.isFinite(policy.maxInitialCapitalWeight)
    || policy.maxInitialCapitalWeight <= 0 || policy.maxInitialCapitalWeight > 0.1) {
    return { permitted: false, reason: 'INVALID_AUTOMATION_POLICY' };
  }
  if (action === 'SELL_CANDIDATE') {
    if (heldQuantity <= 0) return { permitted: false, reason: 'NO_HELD_SHARES' };
    if (!Number.isInteger(heldQuantity))
      return { permitted: false, reason: 'FRACTIONAL_POSITION_REQUIRES_REVIEW' };
    return { permitted: true, quantity: heldQuantity };
  }
  if (action !== 'BUY_CANDIDATE') return { permitted: false, reason: 'NON_ACTIONABLE_SIGNAL' };
  if (heldQuantity !== 0) return { permitted: false, reason: 'REINFORCEMENT_REQUIRES_REVIEW' };

  // Risk budget is based on INITIAL virtual capital, not unrealized gains.
  const budget = Math.min(policy.availableCashEur, policy.initialCapitalEur * policy.maxInitialCapitalWeight);
  const quantity = Math.floor((budget - policy.simulatedFeeEur) / quotePriceEur);
  if (!Number.isSafeInteger(quantity) || quantity <= 0)
    return { permitted: false, reason: 'INSUFFICIENT_BUDGET' };
  return { permitted: true, quantity };
}

/**
 * Admission gate for automation: only actual-exchange Yahoo observations.
 * Twelve Data /price historically stores request time as quote time; block
 * such data rather than fabricate exchange chronology.
 */
export function trustedExchangeQuote(quote: {
  source: string;
  as_of: string;
  created_at: string;
  raw: unknown;
}, now: string): boolean {
  if (quote.source !== 'YAHOO_FINANCE') return false;
  const raw = quote.raw;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return false;
  const observed = (raw as Record<string, unknown>).regularMarketTime;
  const observedMs = typeof observed === 'string' ? Date.parse(observed) : NaN;
  const quoteMs = Date.parse(quote.as_of), storedMs = Date.parse(quote.created_at), nowMs = Date.parse(now);
  return Number.isFinite(observedMs) && Number.isFinite(quoteMs)
    && Number.isFinite(storedMs) && Number.isFinite(nowMs)
    && Math.abs(observedMs - quoteMs) <= 1000
    && quoteMs <= nowMs + 300000 && quoteMs >= nowMs - 96 * 3600000
    && storedMs <= nowMs + 300000;
}
