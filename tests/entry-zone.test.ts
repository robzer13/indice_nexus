import assert from 'node:assert/strict';
import test from 'node:test';
import { getEntryZone } from '../lib/domain/entry-zone';

test('entry zones classify O90 distance deterministically', () => {
  assert.equal(getEntryZone(4), 'AT_OR_BELOW_O90');
  assert.equal(getEntryZone(0), 'AT_OR_BELOW_O90');
  assert.equal(getEntryZone(-4.99), 'WITHIN_5');
  assert.equal(getEntryZone(-5), 'WITHIN_5');
  assert.equal(getEntryZone(-7), 'WITHIN_10');
  assert.equal(getEntryZone(-10), 'WITHIN_10');
  assert.equal(getEntryZone(-15), 'WITHIN_20');
  assert.equal(getEntryZone(-20), 'WITHIN_20');
  assert.equal(getEntryZone(-20.01), 'ABOVE_20');
  assert.equal(getEntryZone(null), 'UNCALIBRATED');
});
