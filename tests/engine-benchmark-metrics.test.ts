import assert from 'node:assert/strict';
import test from 'node:test';
import {
  categoricalAgreementRate,
  computeClassificationMetrics,
  computeForecastMetrics,
  computeStabilityMetrics,
  spearmanRankCorrelation,
  verdictFlipRate,
} from '../lib/orotitan-equity/benchmark/metrics';

test('stability metrics detect deterministic and unstable score sets', () => {
  assert.deepEqual(computeStabilityMetrics([81, 81, 81]), {
    n: 3,
    mean: 81,
    min: 81,
    max: 81,
    range: 0,
    meanAbsoluteDeviation: 0,
    exactAgreementRate: 1,
  });
  const unstable = computeStabilityMetrics([80, 85, 80, 90]);
  assert.equal(unstable.n, 4);
  assert.equal(unstable.exactAgreementRate, 0.5);
  assert.equal(unstable.range, 10);
});

test('categorical agreement and flip rate are complementary', () => {
  const values = ['SUPPORTED', 'SUPPORTED', 'SUPPORTED', 'INSUFFICIENT'] as const;
  assert.equal(categoricalAgreementRate([...values]), 0.75);
  assert.equal(verdictFlipRate([...values]), 0.25);
});

test('classification precision recall f1 are exact', () => {
  const metrics = computeClassificationMetrics(90, 10, 5);
  assert.equal(metrics.precision, 0.9);
  assert.ok(metrics.recall !== null && Math.abs(metrics.recall - 90 / 95) < 1e-12);
  assert.ok(metrics.f1 !== null && metrics.f1 > 0.92);
});

test('spearman recognizes perfect ranking even with shifted values', () => {
  assert.equal(spearmanRankCorrelation([1, 2, 3, 4], [10, 20, 30, 40]), 1);
  assert.equal(spearmanRankCorrelation([1, 2, 3, 4], [40, 30, 20, 10]), -1);
});

test('forecast metrics quantify bias and error separately', () => {
  const result = computeForecastMetrics([10, 12, 14], [9, 11, 15]);
  assert.equal(result.n, 3);
  assert.equal(result.mae, 1);
  assert.ok(result.bias !== null && Math.abs(result.bias - 1 / 3) < 1e-12);
  assert.ok(result.rmse !== null && Math.abs(result.rmse - 1) < 1e-12);
  assert.equal(result.spearman, 1);
});
