import assert from 'node:assert/strict';
import test from 'node:test';
import { computeOqs } from '../lib/orotitan-equity/v1/quality';

test('benchmark runner I2 principle: historical Intuitive dimensions reproduce OQS 81', () => {
  const result = computeOqs({
    MOAT: 85,
    RUNWAY: 85,
    RETURN_QUALITY: 80,
    CASH_ECONOMICS: 80,
    CAPITAL_ALLOCATION: 75,
    MANAGEMENT_GOVERNANCE: 80,
    RESILIENCE_RISK: 80,
  });
  assert.deepEqual(result, { oqsRaw: 81, weakLinkCap: 100, oqs: 81 });
});
