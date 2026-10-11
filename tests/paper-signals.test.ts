import assert from 'node:assert/strict';
import test from 'node:test';
import {
  PAPER_LONG_TERM_RULESET,
  evaluateLongTermPaperSignal,
  type PaperSignalInput,
} from '../lib/domain/paper-signals';

const baseline: PaperSignalInput = {
  strategy: { profile: 'VALUE', thesisState: 'ACTIVE', entryPrice: 4.5, exitTargetPrice: 12 },
  snapshot: {
    id: 'canonical-published-snapshot',
    analysisDate: '2026-10-01',
    canonicalEntryPrice: 4.5,
    eligibleForInvestment: true,
    coreQualityConfirmed: false,
  },
  quote: { price: 4.5, asOf: '2026-10-10T18:00:00Z' },
  positionQuantity: 0,
  now: '2026-10-11T10:00:00Z',
};

function evaluate(changes: Partial<PaperSignalInput> = {}) {
  return evaluateLongTermPaperSignal({
    ...baseline,
    ...changes,
    strategy: { ...baseline.strategy, ...changes.strategy },
    snapshot: { ...baseline.snapshot, ...changes.snapshot },
    quote: { ...baseline.quote, ...changes.quote },
  });
}

test('explicit VALUE entry signal is a candidate, never an order', () => {
  const result = evaluate();
  assert.equal(result.action, 'BUY_CANDIDATE');
  assert.equal(result.reason, 'ENTRY_PRICE_REACHED');
  assert.equal(result.ruleset, PAPER_LONG_TERM_RULESET);
  assert.equal(result.executable, false);
  assert.equal(result.snapshotId, baseline.snapshot.id);
});

test('VALUE position bought at 4.50 has a 12.00 exit candidate at 12.00', () => {
  const result = evaluate({ positionQuantity: 10, quote: { price: 12, asOf: baseline.quote.asOf } });
  assert.equal(result.action, 'SELL_CANDIDATE');
  assert.equal(result.reason, 'TARGET_REACHED');
});

test('VALUE held below exit target remains HOLD', () => {
  assert.equal(evaluate({ positionQuantity: 10, quote: { price: 11.99, asOf: baseline.quote.asOf } }).action, 'HOLD');
});

test('CORE has no mechanical take-profit target even when price doubles', () => {
  const signal = evaluate({
    strategy: { profile: 'CORE', thesisState: 'ACTIVE', exitTargetPrice: 12 },
    snapshot: { coreQualityConfirmed: true, id: baseline.snapshot.id, analysisDate: baseline.snapshot.analysisDate,
      canonicalEntryPrice: baseline.snapshot.canonicalEntryPrice, eligibleForInvestment: true },
    quote: { price: 50, asOf: baseline.quote.asOf },
    positionQuantity: 10,
  });
  assert.equal(signal.action, 'HOLD');
});

test('CORE buying requires explicit quality confirmation', () => {
  assert.equal(evaluate({ strategy: { profile: 'CORE', thesisState: 'ACTIVE' } }).action, 'WAIT');
  assert.equal(evaluate({
    strategy: { profile: 'CORE', thesisState: 'ACTIVE' },
    snapshot: { ...baseline.snapshot, coreQualityConfirmed: true },
  }).action, 'BUY_CANDIDATE');
});

test('CORE with deteriorated quality needs review rather than an invented sell order', () => {
  assert.equal(evaluate({
    strategy: { profile: 'CORE', thesisState: 'ACTIVE' },
    positionQuantity: 10,
  }).action, 'REVIEW_REQUIRED');
});

test('cyclical exit requires both target and manually confirmed milestone', () => {
  const cycle = { profile: 'CYCLICAL_VALUE' as const, thesisState: 'ACTIVE' as const, exitTargetPrice: 12 };
  assert.equal(evaluate({ strategy: cycle, positionQuantity: 10, quote: { price: 12, asOf: baseline.quote.asOf } }).action, 'REVIEW_REQUIRED');
  assert.equal(evaluate({
    strategy: { ...cycle, milestoneState: 'CONFIRMED' }, positionQuantity: 10,
    quote: { price: 12, asOf: baseline.quote.asOf },
  }).action, 'SELL_CANDIDATE');
});

test('turnaround exit requires milestone confirmation', () => {
  assert.equal(evaluate({
    strategy: { profile: 'TURNAROUND', thesisState: 'ACTIVE', exitTargetPrice: 12 },
    positionQuantity: 2, quote: { price: 12, asOf: baseline.quote.asOf },
  }).action, 'REVIEW_REQUIRED');
});

test('thesis invalidation is a sell candidate only when a position exists', () => {
  assert.equal(evaluate({ strategy: { ...baseline.strategy, thesisState: 'INVALIDATED' }, positionQuantity: 1 }).action, 'SELL_CANDIDATE');
  assert.equal(evaluate({ strategy: { ...baseline.strategy, thesisState: 'INVALIDATED' } }).action, 'WAIT');
});

test('unverified thesis never emits buy/sell', () => {
  assert.equal(evaluate({ strategy: { ...baseline.strategy, thesisState: 'UNKNOWN' } }).action, 'REVIEW_REQUIRED');
});

test('missing, old, future and invalid market quotes fail closed', () => {
  assert.equal(evaluate({ quote: { price: null, asOf: baseline.quote.asOf } }).reason, 'MISSING_OR_INVALID_PRICE');
  assert.equal(evaluate({ quote: { price: 4.5, asOf: null } }).reason, 'MISSING_OR_INVALID_QUOTE_TIME');
  assert.equal(evaluate({ quote: { price: 4.5, asOf: '2026-09-10T10:00:00Z' } }).reason, 'STALE_QUOTE');
  assert.equal(evaluate({ quote: { price: 4.5, asOf: '2026-10-12T10:00:00Z' } }).reason, 'FUTURE_QUOTE');
});

test('missing or outdated analysis blocks fresh purchase and demands review of existing position', () => {
  assert.equal(evaluate({ snapshot: { ...baseline.snapshot, id: null } }).reason, 'MISSING_SNAPSHOT');
  assert.equal(evaluate({ snapshot: { ...baseline.snapshot, analysisDate: '2025-01-01' } }).action, 'DATA_BLOCKED');
  assert.equal(evaluate({ snapshot: { ...baseline.snapshot, analysisDate: '2025-01-01' }, positionQuantity: 1 }).action, 'REVIEW_REQUIRED');
});

test('ineligible published analysis never becomes a buy recommendation', () => {
  assert.equal(evaluate({ snapshot: { ...baseline.snapshot, eligibleForInvestment: false } }).action, 'WAIT');
});

test('VALUE needs a valid objective and entry/exit ordering', () => {
  assert.equal(evaluate({ strategy: { profile: 'VALUE', thesisState: 'ACTIVE', exitTargetPrice: null } }).action, 'REVIEW_REQUIRED');
  assert.equal(evaluate({ strategy: { ...baseline.strategy, exitTargetPrice: 0 } }).action, 'DATA_BLOCKED');
  assert.equal(evaluate({ strategy: { ...baseline.strategy, exitTargetPrice: 3 } }).reason, 'TARGET_BELOW_ENTRY');
});

test('invalid position quantities are blocked', () => {
  assert.equal(evaluate({ positionQuantity: -1 }).reason, 'INVALID_POSITION');
  assert.equal(evaluate({ positionQuantity: Number.NaN }).reason, 'INVALID_POSITION');
});

test('neither signal evaluation nor ticker prices retroactively define executions', () => {
  const signal = evaluate({ positionQuantity: 10, quote: { price: 12, asOf: baseline.quote.asOf } });
  assert.equal(signal.action, 'SELL_CANDIDATE');
  assert.equal(signal.executable, false);
  assert.equal('orderId' in signal, false);
  assert.equal('filledAt' in signal, false);
});
