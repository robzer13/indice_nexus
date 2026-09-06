import assert from 'node:assert/strict';
import test from 'node:test';
import {
  MARKET_SYNC_COOLDOWN_MS,
  getMarketSyncCooldownRemainingMs,
} from '../lib/domain/market-sync';

test('market sync cooldown blocks another run for 65 seconds', () => {
  const finished = '2026-09-06T15:33:41.000Z';
  const finishedMs = new Date(finished).getTime();

  assert.equal(
    getMarketSyncCooldownRemainingMs(finished, finishedMs),
    MARKET_SYNC_COOLDOWN_MS,
  );
  assert.equal(
    getMarketSyncCooldownRemainingMs(finished, finishedMs + 60_000),
    5_000,
  );
  assert.equal(
    getMarketSyncCooldownRemainingMs(finished, finishedMs + 65_000),
    0,
  );
});

test('market sync cooldown ignores invalid timestamps', () => {
  assert.equal(getMarketSyncCooldownRemainingMs('not-a-date', 0), 0);
});
