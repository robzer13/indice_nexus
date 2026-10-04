import assert from 'node:assert/strict';
import test from 'node:test';
import { computeLiveValuation, repriceExpectedReturn } from '@/lib/domain/live-valuation';

test('price-only repricing reproduces the reference expected return at the reference price', () => {
  const value = repriceExpectedReturn(-5.105543129034096, 393.33, 393.33, 5);
  assert.ok(Math.abs(value - (-5.105543129034096)) < 1e-10);
});

test('Intuitive Surgical price ladder reprices to the 10% hurdle', () => {
  const value = repriceExpectedReturn(-5.105543129034096, 393.33, 187.930759359571, 5);
  assert.ok(Math.abs(value - 10) < 1e-9);
});

test('live valuation keeps frozen non-price caps while repricing I2 return components', () => {
  const result = computeLiveValuation({
    currentPrice: 187.930759359571,
    referencePrice: 393.33,
    horizonYears: 5,
    primaryExpectedReturn: -5.105543129034096,
    matureNormalizationReturn: -5.105543129034096,
    noMultipleExpansionReturn: 11.499201973561867,
    requiredReturnH: 10,
    marginOfSafety: 'NONE',
    valuationReliability: 'LOW',
    scorePermission: 'CONDITIONAL',
    investmentConclusionStatus: 'CERTIFIED_WITH_LIMITATIONS',
    oqs: 81,
    canonicalOvs: 0,
    canonicalInvestmentScore: 15,
  });

  assert.ok(result);
  assert.ok(Math.abs(result.primaryExpectedReturnPct - 10) < 1e-9);
  assert.equal(result.primaryExpectedReturnScore, 70);
  assert.equal(result.ovs, 55);
  assert.equal(result.investmentScore, 70);
});
