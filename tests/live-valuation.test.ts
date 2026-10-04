import assert from 'node:assert/strict';
import test from 'node:test';
import { computeLiveValuation, repriceAnnualizedReturnPct } from '../lib/domain/live-valuation';

test('repriceAnnualizedReturnPct preserves the certified terminal value', () => {
  const result = repriceAnnualizedReturnPct({
    currentPrice: 187.930759359571,
    referencePrice: 393.33,
    referenceAnnualizedReturnPct: -5.105543129034096,
    horizonYears: 5,
  });
  assert.ok(result !== null);
  assert.ok(Math.abs(result - 10) < 1e-9);
});

test('live valuation reproduces the published ISRG scores at the reference price', () => {
  const result = computeLiveValuation({
    currentPrice: 393.33,
    referencePrice: 393.33,
    horizonYears: 5,
    primaryExpectedReturnPct: -5.105543129034096,
    normalizationExpectedReturnPct: -5.105543129034096,
    requiredReturnPct: 10,
    marginOfSafety: 'NONE',
    valuationReliability: 'LOW',
    scorePermission: 'CONDITIONAL',
    investmentConclusionStatus: 'CERTIFIED_WITH_LIMITATIONS',
    oqs: 81,
  });
  assert.equal(result.available, true);
  assert.equal(result.valuationScore, 0);
  assert.equal(result.investmentScore, 15);
});

test('live valuation improves when the price reaches the required-return threshold', () => {
  const result = computeLiveValuation({
    currentPrice: 187.930759359571,
    referencePrice: 393.33,
    horizonYears: 5,
    primaryExpectedReturnPct: -5.105543129034096,
    normalizationExpectedReturnPct: -5.105543129034096,
    requiredReturnPct: 10,
    marginOfSafety: 'NONE',
    valuationReliability: 'LOW',
    scorePermission: 'CONDITIONAL',
    investmentConclusionStatus: 'CERTIFIED_WITH_LIMITATIONS',
    oqs: 81,
  });
  assert.equal(result.available, true);
  assert.ok(result.primaryExpectedReturnPct !== null && Math.abs(result.primaryExpectedReturnPct - 10) < 1e-9);
  assert.equal(result.valuationScore, 55);
  assert.equal(result.investmentScore, 70);
});
