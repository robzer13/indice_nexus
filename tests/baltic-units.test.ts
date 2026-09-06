import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

test('Baltic Classifieds Yahoo metadata uses major GBP without extra scaling', async () => {
  const migration = await readFile('migrations/v1_1_3_baltic_units.sql', 'utf8');
  assert.match(migration, /currency = 'GBP'/);
  assert.match(migration, /quote_unit = 'MAJOR'/);
  assert.match(migration, /market_data_multiplier = 1/);
  assert.match(migration, /market_data_multiplier = 0\.01/);
});
