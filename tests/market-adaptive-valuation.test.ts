import assert from 'node:assert/strict';
import test from 'node:test';
import { computeMarketAdaptiveValuation } from '../lib/orotitan-equity/market-adaptive';

const baseInput = {
  currentPrice: 393.33,
  referencePrice: 393.33,
  horizonYears: 5,
  primaryExpectedReturnPct: -5.105543129034096,
  normalizationExpectedReturnPct: -5.105543129034096,
  requiredReturnPct: 10,
  mosStatus: 'NONE' as const,
  valuationReliability: 'LOW' as const,
  scorePermission: 'CONDITIONAL' as const,
  investmentConclusionStatus: 'CERTIFIED_WITH_LIMITATIONS' as const,
  oqs: 81,
};

test('market adaptive valuation reproduces the certified ISRG score at reference price', () => {
  const result = computeMarketAdaptiveValuation(baseInput);
  assert.ok(result);
  assert.equal(result.ovs, 0);
  assert.equal(result.investmentScore, 15);
  assert.ok(Math.abs(result.primaryExpectedReturnPct - baseInput.primaryExpectedReturnPct) < 1e-9);
});

test('market adaptive valuation improves when price falls without changing OQS', () => {
  const result = computeMarketAdaptiveValuation({ ...baseInput, currentPrice: 187.930759359571 });
  assert.ok(result);
  assert.ok(result.primaryExpectedReturnPct >= 9.999999);
  assert.equal(result.ovs, 55);
  assert.equal(result.investmentScore, 70);
});

test('market adaptive valuation rejects invalid price inputs', () => {
  assert.equal(computeMarketAdaptiveValuation({ ...baseInput, currentPrice: 0 }), null);
});
