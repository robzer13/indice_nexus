import assert from 'node:assert/strict';
import test from 'node:test';
import { getDistanceO90 } from '../lib/domain/distance';

function assertClose(actual: number | null, expected: number, epsilon = 1e-10): void {
  assert.notEqual(actual, null);
  assert.ok(Math.abs((actual as number) - expected) < epsilon);
}

test('distance O90: 90 vs 100 = -10%', () => {
  assertClose(getDistanceO90(100, 90), -10);
});

test('distance O90: 100 vs 100 = 0%', () => {
  assert.equal(getDistanceO90(100, 100), 0);
});

test('distance O90: 110 vs 100 = +10%', () => {
  assertClose(getDistanceO90(100, 110), 10);
});

test('distance O90: null remains null', () => {
  assert.equal(getDistanceO90(100, null), null);
});

test('distance O90 rejects invalid current prices', () => {
  assert.equal(getDistanceO90(0, 90), null);
  assert.equal(getDistanceO90(-1, 90), null);
});
