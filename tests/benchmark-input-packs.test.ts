import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { findForbiddenScoringKeys } from '../lib/orotitan-equity/benchmark/pre-score-pack';

test('V2 reference benchmark has exactly 20 sanitized pre-score packs with no forbidden scoring keys', async () => {
  const root = path.join(process.cwd(), 'benchmark', 'input-packs', 'v2-reference-20');
  const files = (await readdir(root)).filter((name) => name.endsWith('.json')).sort();
  assert.equal(files.length, 20);
  assert.deepEqual(files, Array.from({ length: 20 }, (_, index) => `V2REF-${String(index + 1).padStart(3, '0')}.json`));

  for (const file of files) {
    const pack = JSON.parse(await readFile(path.join(root, file), 'utf8')) as Record<string, unknown>;
    assert.equal(pack.pack_type, 'OROTITAN_V2_PRE_SCORE_REPLAY_INPUT');
    assert.equal((pack.leakage_controls as Record<string, unknown>).forbidden_key_scan, 'PASS');
    assert.deepEqual(findForbiddenScoringKeys(pack.analytical_input), [], file);
  }
});
