import assert from 'node:assert/strict';
import test from 'node:test';
import {
  deterministicOutputsEqual,
  deterministicOutputsWithinTolerance,
  normalizeI2Report,
} from '../lib/orotitan-equity/benchmark/i2-normalization';

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
  assert.equal(deterministicOutputsWithinTolerance(report), true);
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

test('supports Brookfield admitted shape and absolute-zero deltas', () => {
  const report = normalizeI2Report({
    i2_reconciliation_status: 'PASS',
    authority: { name: 'I2_CANONICAL_COMPUTATION', version: '1.0' },
    recomputed: { oqs: 77.5, ovs: 95, investment_score: 77.5 },
    admitted: { oqs: 77.5, ovs: 95, investment_score: 77.5 },
    absolute_deltas: { oqs: 0, ovs: 0, investment_score: 0 },
  });
  assert.equal(report.status, 'PASS');
  assert.equal(report.exactReconciliation, true);
  assert.equal(deterministicOutputsWithinTolerance(report), true);
});

test('supports string authority and candidate-match exactness', () => {
  const report = normalizeI2Report({
    reconciliation_status: 'PASS',
    authority: 'I2_CANONICAL_COMPUTATION@1.0',
    recomputed: { oqs: 85.75, ovs: 48.60840141205742 },
    candidate_match: { oqs: 'PASS', ovs: 'PASS' },
  });
  assert.equal(report.authorityName, 'I2_CANONICAL_COMPUTATION');
  assert.equal(report.authorityVersion, '1.0');
  assert.equal(report.exactReconciliation, true);
});

test('supports i2_authority and persisted outputs', () => {
  const report = normalizeI2Report({
    status: 'PASS',
    i2_authority: { name: 'I2_CANONICAL_COMPUTATION', version: '1.0', content_sha256: 'xyz' },
    recomputed: { oqs: 77.5, ovs: 70.29734252891285 },
    persisted: { oqs: 77.5, ovs: 70.29734252891285 },
    exact_match: true,
  });
  assert.equal(report.authorityName, 'I2_CANONICAL_COMPUTATION');
  assert.equal(report.exactReconciliation, true);
});

test('supports tolerance-based historical reconciliation', () => {
  const report = normalizeI2Report({
    result: 'PASS',
    tolerance: 1e-10,
    recomputed: { INVESTMENT_RAW: 60.33565299717668, OVS: 10.952176657255615 },
    certified: { INVESTMENT_RAW: 60.33565299717669, OVS: 10.952176657255615 },
  });
  assert.equal(report.status, 'PASS');
  assert.equal(report.exactReconciliation, null);
  assert.equal(deterministicOutputsEqual(report), false);
  assert.equal(deterministicOutputsWithinTolerance(report), true);
});

test('supports range outputs', () => {
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
  assert.equal(deterministicOutputsWithinTolerance(report), null);
});
