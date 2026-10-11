import { ANALYSIS_STALE_DAYS } from './data-health';
import { MARKET_PRICE_STALE_HOURS } from './freshness';

/**
 * Simulation-only decision support. These actions are not exchange orders,
 * investment recommendations, or Certification/Registry transitions.
 * The caller must persist the evidence and decision timestamp separately.
 */
export const PAPER_LONG_TERM_RULESET = 'PAPER_LONG_TERM_V1' as const;

export type PaperInvestmentProfile = 'CORE' | 'VALUE' | 'CYCLICAL_VALUE' | 'TURNAROUND';
export type PaperThesisState = 'ACTIVE' | 'INVALIDATED' | 'UNKNOWN';
export type PaperMilestoneState = 'CONFIRMED' | 'UNCONFIRMED' | 'NOT_APPLICABLE';

export type PaperSignalAction =
  | 'BUY_CANDIDATE'
  | 'SELL_CANDIDATE'
  | 'HOLD'
  | 'WAIT'
  | 'REVIEW_REQUIRED'
  | 'DATA_BLOCKED';

export type PaperSignalReason =
  | 'INVALID_POSITION'
  | 'MISSING_SNAPSHOT'
  | 'MISSING_OR_INVALID_PRICE'
  | 'MISSING_OR_INVALID_QUOTE_TIME'
  | 'FUTURE_QUOTE'
  | 'STALE_QUOTE'
  | 'STALE_OR_INVALID_ANALYSIS'
  | 'THESIS_INVALIDATED'
  | 'THESIS_UNVERIFIED'
  | 'SNAPSHOT_INELIGIBLE'
  | 'CORE_QUALITY_NOT_CONFIRMED'
  | 'CORE_QUALITY_DETERIORATED'
  | 'MISSING_EXIT_TARGET'
  | 'INVALID_EXIT_TARGET'
  | 'TARGET_REACHED'
  | 'MILESTONE_REQUIRES_REVIEW'
  | 'NO_ENTRY_PRICE'
  | 'INVALID_ENTRY_PRICE'
  | 'TARGET_BELOW_ENTRY'
  | 'ENTRY_PRICE_REACHED'
  | 'ABOVE_ENTRY_PRICE'
  | 'LONG_TERM_HOLD';

export interface PaperSignalInput {
  /** Versioned, explicitly selected *position* policy, not an analytical archetype. */
  strategy: {
    profile: PaperInvestmentProfile;
    thesisState: PaperThesisState;
    entryPrice?: number | null;
    exitTargetPrice?: number | null;
    milestoneState?: PaperMilestoneState;
  };
  snapshot: {
    id: string | null;
    analysisDate: string | null; // YYYY-MM-DD
    canonicalEntryPrice: number | null; // published required-return price H
    eligibleForInvestment: boolean;
    coreQualityConfirmed: boolean | null;
  };
  quote: {
    price: number | null;
    asOf: string | null;
  };
  positionQuantity: number;
  now: string; // injected clock, never inferred from market price timestamps
}

export interface PaperSignal {
  ruleset: typeof PAPER_LONG_TERM_RULESET;
  profile: PaperInvestmentProfile;
  action: PaperSignalAction;
  reason: PaperSignalReason;
  snapshotId: string | null;
  quoteAsOf: string | null;
  referencePrice: number | null;
  /** No order is placed by this function; execution requires separate authorization. */
  executable: false;
}

const FUTURE_QUOTE_TOLERANCE_MS = 5 * 60 * 1000;
const DAY_MS = 86_400_000;

function positive(value: number | null | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

export function evaluateLongTermPaperSignal(input: PaperSignalInput): PaperSignal {
  const { strategy, snapshot, quote, positionQuantity } = input;
  const decision = (action: PaperSignalAction, reason: PaperSignalReason): PaperSignal => ({
    ruleset: PAPER_LONG_TERM_RULESET,
    profile: strategy.profile,
    action,
    reason,
    snapshotId: snapshot.id,
    quoteAsOf: quote.asOf,
    referencePrice: quote.price,
    executable: false,
  });

  if (!Number.isFinite(positionQuantity) || positionQuantity < 0) {
    return decision('DATA_BLOCKED', 'INVALID_POSITION');
  }
  if (!snapshot.id) return decision('DATA_BLOCKED', 'MISSING_SNAPSHOT');
  if (!positive(quote.price)) return decision('DATA_BLOCKED', 'MISSING_OR_INVALID_PRICE');

  const quoteTime = quote.asOf === null ? NaN : Date.parse(quote.asOf);
  const now = Date.parse(input.now);
  if (!Number.isFinite(now) || !Number.isFinite(quoteTime)) {
    return decision('DATA_BLOCKED', 'MISSING_OR_INVALID_QUOTE_TIME');
  }
  if (quoteTime > now + FUTURE_QUOTE_TOLERANCE_MS) {
    return decision('DATA_BLOCKED', 'FUTURE_QUOTE');
  }
  if (now - quoteTime > MARKET_PRICE_STALE_HOURS * 3_600_000) {
    return decision('DATA_BLOCKED', 'STALE_QUOTE');
  }

  const analysisTime = snapshot.analysisDate && /^\d{4}-\d{2}-\d{2}$/.test(snapshot.analysisDate)
    ? Date.parse(snapshot.analysisDate + 'T00:00:00Z')
    : NaN;
  if (!Number.isFinite(analysisTime)
      || analysisTime > now + DAY_MS
      || now - analysisTime > ANALYSIS_STALE_DAYS * DAY_MS) {
    return decision(positionQuantity > 0 ? 'REVIEW_REQUIRED' : 'DATA_BLOCKED', 'STALE_OR_INVALID_ANALYSIS');
  }

  if (strategy.thesisState === 'INVALIDATED') {
    return decision(positionQuantity > 0 ? 'SELL_CANDIDATE' : 'WAIT', 'THESIS_INVALIDATED');
  }
  if (strategy.thesisState !== 'ACTIVE') {
    return decision('REVIEW_REQUIRED', 'THESIS_UNVERIFIED');
  }
  if (!snapshot.eligibleForInvestment) {
    return decision(positionQuantity > 0 ? 'REVIEW_REQUIRED' : 'WAIT', 'SNAPSHOT_INELIGIBLE');
  }

  if (strategy.profile === 'CORE') {
    if (snapshot.coreQualityConfirmed !== true) {
      return decision(positionQuantity > 0 ? 'REVIEW_REQUIRED' : 'WAIT',
        positionQuantity > 0 ? 'CORE_QUALITY_DETERIORATED' : 'CORE_QUALITY_NOT_CONFIRMED');
    }
    if (positionQuantity > 0) return decision('HOLD', 'LONG_TERM_HOLD');
  } else {
    if (strategy.exitTargetPrice == null) {
      return decision('REVIEW_REQUIRED', 'MISSING_EXIT_TARGET');
    }
    if (!positive(strategy.exitTargetPrice)) {
      return decision('DATA_BLOCKED', 'INVALID_EXIT_TARGET');
    }
    if (positionQuantity > 0) {
      if (quote.price >= strategy.exitTargetPrice) {
        if ((strategy.profile === 'CYCLICAL_VALUE' || strategy.profile === 'TURNAROUND')
          && strategy.milestoneState !== 'CONFIRMED') {
          return decision('REVIEW_REQUIRED', 'MILESTONE_REQUIRES_REVIEW');
        }
        return decision('SELL_CANDIDATE', 'TARGET_REACHED');
      }
      return decision('HOLD', 'LONG_TERM_HOLD');
    }
  }

  const entry = strategy.entryPrice === undefined || strategy.entryPrice === null
    ? snapshot.canonicalEntryPrice
    : strategy.entryPrice;
  if (entry == null) return decision('WAIT', 'NO_ENTRY_PRICE');
  if (!positive(entry)) return decision('DATA_BLOCKED', 'INVALID_ENTRY_PRICE');

  if (strategy.profile !== 'CORE' && positive(strategy.exitTargetPrice)
    && strategy.exitTargetPrice <= entry) {
    return decision('DATA_BLOCKED', 'TARGET_BELOW_ENTRY');
  }
  return quote.price <= entry
    ? decision('BUY_CANDIDATE', 'ENTRY_PRICE_REACHED')
    : decision('WAIT', 'ABOVE_ENTRY_PRICE');
}
