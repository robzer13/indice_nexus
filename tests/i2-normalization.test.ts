import assert from 'node:assert/strict';
import test from 'node:test';
import { deterministicOutputsEqual, normalizeI2Report } from '../lib/orotitan-equity/benchmark/i2-normalization';

test('normalizes current I2 report shape', () => {
  const report = normalizeI2Report({
    status: 'PASS',
    exact_reconciliation: true,
    authority: { name: 'I2_CANONICAL_COMPUTATION', version: '1.0', content_sha256: 'abc' },
    recomputation: { oqs: 81, ovs: 0, investment_score: 15 },
    admitted_outputs: { oqs: 81, ovs: 0, investment_score: 15 },
  });
  assert.equal(report.status, 'PASS');
  assert.equal(report.exactReconciliation, true);
  assert.equal(report.authorityName, 'I2_CANONICAL_COMPUTATION');
  assert.equal(deterministicOutputsEqual(report), true);
});

test('normalizes earlier V2 report shape', () => {
  const report = normalizeI2Report({
    reconciliation_status: 'PASS',
    authority: { name: 'I2_CANONICAL_COMPUTATION', version: '1.0', content_sha256: 'def' },
    recomputed: { oqs: '78.25', ovs: '1.1739997843906345' },
    authoritative_deep_dive_outputs: { oqs: 78.25, ovs: 1.1739997843906345 },
    comparisons: { all_deterministic_outputs_equal: true },
  });
  assert.equal(report.status, 'PASS');
  assert.equal(report.exactReconciliation, true);
  assert.deepEqual(report.recomputedOutputs, { oqs: 78.25, ovs: 1.1739997843906345 });
  assert.equal(deterministicOutputsEqual(report), true);
});

test('supports i2_reconciliation status and range outputs', () => {
  const report = normalizeI2Report({
    i2_reconciliation: 'PASS',
    recomputation: { ovs: { min: '10.5', max: '12.5' } },
    admitted_outputs: { ovs: { min: 10.5, max: 12.5 } },
  });
  assert.equal(report.status, 'PASS');
  assert.equal(deterministicOutputsEqual(report), true);
});

test('fails closed on unrecognized payloads', () => {
  const report = normalizeI2Report({ hello: 'world' });
  assert.equal(report.status, 'UNKNOWN');
  assert.equal(deterministicOutputsEqual(report), null);
});
