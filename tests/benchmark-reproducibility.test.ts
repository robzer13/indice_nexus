import assert from 'node:assert/strict';
import test from 'node:test';
import {
  BENCHMARK_DIMENSIONS,
  evaluateCaseReproducibility,
  summarizeCampaign,
  type ReproducibilityExecution,
} from '../lib/orotitan-equity/benchmark/reproducibility';

function execution(caseId: string, repetitionIndex: number, oqs: number, verdict = 'NO'): ReproducibilityExecution {
  return {
    caseId,
    repetitionIndex,
    dimensionScores: Object.fromEntries(BENCHMARK_DIMENSIONS.map((dimension) => [dimension, 80])) as ReproducibilityExecution['dimensionScores'],
    oqs,
    certificationStatus: 'CERTIFIED_WITH_LIMITATIONS',
    terminalVerdict: verdict,
    i2ReconciliationPass: true,
  };
}

test('deterministic repeated executions produce perfect agreement', () => {
  const result = evaluateCaseReproducibility([
    execution('CASE-1', 1, 81),
    execution('CASE-1', 2, 81),
    execution('CASE-1', 3, 81),
  ]);
  assert.equal(result.oqs.meanAbsoluteDeviation, 0);
  assert.equal(result.oqs.exactAgreementRate, 1);
  assert.equal(result.terminalVerdictFlipRate, 0);
  assert.equal(result.certificationFlipRate, 0);
  assert.equal(result.i2PassRate, 1);
});

test('score and verdict instability are measurable separately', () => {
  const result = evaluateCaseReproducibility([
    execution('CASE-2', 1, 80, 'NO'),
    execution('CASE-2', 2, 85, 'NO'),
    execution('CASE-2', 3, 80, 'YES'),
  ]);
  assert.ok(result.oqs.meanAbsoluteDeviation !== null && result.oqs.meanAbsoluteDeviation > 0);
  assert.equal(result.oqs.exactAgreementRate, 2 / 3);
  assert.equal(result.terminalVerdictFlipRate, 1 / 3);
  assert.equal(result.i2PassRate, 1);
});

test('campaign summary weights case metrics by executions', () => {
  const a = evaluateCaseReproducibility([execution('A', 1, 81), execution('A', 2, 81)]);
  const b = evaluateCaseReproducibility([execution('B', 1, 70), execution('B', 2, 75), execution('B', 3, 70)]);
  const summary = summarizeCampaign([a, b]);
  assert.equal(summary.cases, 2);
  assert.equal(summary.executions, 5);
  assert.ok(summary.exactOqsAgreementRate !== null && summary.exactOqsAgreementRate < 1);
  assert.equal(summary.i2PassRate, 1);
});
