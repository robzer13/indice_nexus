import assert from 'node:assert/strict';
import test from 'node:test';
import { getFairValueUpsidePct } from '../lib/domain/valuation-upside';

test('fair value upside is currency-neutral when price and FV share the same unit', () => {
  assert.equal(Number(getFairValueUpsidePct(1457, 1600)?.toFixed(1)), 9.8);
  assert.equal(Number(getFairValueUpsidePct(2.486, 2.7)?.toFixed(1)), 8.6);
  assert.equal(Number(getFairValueUpsidePct(110.43, 102.5)?.toFixed(1)), -7.2);
});

test('fair value upside returns null for missing or invalid values', () => {
  assert.equal(getFairValueUpsidePct(null, 100), null);
  assert.equal(getFairValueUpsidePct(100, null), null);
  assert.equal(getFairValueUpsidePct(0, 100), null);
  assert.equal(getFairValueUpsidePct(100, 0), null);
});
