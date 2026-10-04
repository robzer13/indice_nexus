import assert from 'node:assert/strict';
import test from 'node:test';
import {
  findForbiddenScoringKeys,
  sanitizePreScoreInput,
  sha256Json,
} from '../lib/orotitan-equity/benchmark/pre-score-pack';

test('sanitizer strips score, aggregate, certification, valuation and terminal leakage', () => {
  const raw = {
    business: {
      moat: { evidence_state: 'SUPPORTED', score: 85 },
      cash: { standardized_fcf: 123, investment_score: 99 },
    },
    scores: { oqs: 81 },
    certification: { score_permission: 'ALLOWED' },
    valuation: { ovs: 20 },
    terminal: { orotitan_status: 'NO' },
  };
  const clean = sanitizePreScoreInput(raw);
  assert.deepEqual(clean, {
    business: {
      moat: { evidence_state: 'SUPPORTED' },
      cash: { standardized_fcf: 123 },
    },
  });
  assert.deepEqual(findForbiddenScoringKeys(clean), []);
});

test('sanitizer preserves analytical numeric facts and evidence ceilings', () => {
  const clean = sanitizePreScoreInput({
    moat: { evidence_ceiling: 90, duration: 'LONG' },
    return_quality: { roic_pct: 31.3, roiic_pct: 40.8 },
  });
  assert.deepEqual(clean, {
    moat: { evidence_ceiling: 90, duration: 'LONG' },
    return_quality: { roic_pct: 31.3, roiic_pct: 40.8 },
  });
});

test('stable JSON hash is invariant to object key order', () => {
  assert.equal(sha256Json({ a: 1, b: 2 }), sha256Json({ b: 2, a: 1 }));
});
