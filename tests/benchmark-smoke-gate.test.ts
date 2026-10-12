import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import {
  validateBenchmarkSmokeGate,
  type BenchmarkSmokeRequest,
} from '../lib/orotitan-equity/benchmark/smoke-gate';

const allowed: BenchmarkSmokeRequest = {
  mode: 'SMOKE_ONLY',
  model: 'openai/gpt-5.4',
  reasoning: 'medium',
  caseId: 'V2REF-010',
  repetitionIndex: 1,
};

test('smoke gate only accepts the frozen one-case, one-repetition candidate', () => {
  assert.equal(validateBenchmarkSmokeGate(allowed), null);
});

test('smoke gate rejects absent, invalid, or campaign-wide modes', () => {
  for (const mode of [undefined, '', 'FULL', 'BENCHMARK', 'true', 'SMOKE']) {
    assert.equal(
      validateBenchmarkSmokeGate({ ...allowed, mode }),
      'BENCHMARK_SMOKE_ONLY_MODE_REQUIRED',
    );
  }
});

test('smoke gate cannot silently change candidate model or reasoning', () => {
  for (const model of [undefined, 'openai/gpt-5.5', 'gateway/gpt-5.4', 'openai/gpt-5.4-mini']) {
    assert.equal(
      validateBenchmarkSmokeGate({ ...allowed, model }),
      'BENCHMARK_SMOKE_PROFILE_MISMATCH',
    );
  }
  for (const reasoning of [undefined, 'high', 'none', 'low']) {
    assert.equal(
      validateBenchmarkSmokeGate({ ...allowed, reasoning }),
      'BENCHMARK_SMOKE_PROFILE_MISMATCH',
    );
  }
});

test('smoke gate rejects all other cases and repetitions', () => {
  for (const caseId of ['V2REF-001', 'V2REF-011', 'V2REF-020', 'v2ref-010']) {
    assert.equal(
      validateBenchmarkSmokeGate({ ...allowed, caseId }),
      'BENCHMARK_SMOKE_CASE_NOT_ALLOWED',
    );
  }
  for (const repetitionIndex of [0, 2, 10, 200]) {
    assert.equal(
      validateBenchmarkSmokeGate({ ...allowed, repetitionIndex }),
      'BENCHMARK_SMOKE_CASE_NOT_ALLOWED',
    );
  }
});

test('runner applies smoke gate before any database claims and paid model invocation', async () => {
  const route = await readFile('app/api/internal/benchmark/scoring-replay/route.ts', 'utf8');
  const env = await readFile('.env.example', 'utf8');
  const guard = route.indexOf('validateBenchmarkSmokeGate({');
  const claim = route.indexOf('await claimBenchmarkExecution(lookup)');
  const paid = route.indexOf('await runScoringReplay({');
  assert.ok(guard >= 0 && claim > guard && paid > claim, 'smoke guard precedes reservation and paid inference');
  assert.match(route, /OROTITAN_BENCHMARK_RUNNER_ENABLED[^\n]*'true'/);
  assert.match(env, /OROTITAN_BENCHMARK_RUNNER_ENABLED=false/);
  assert.match(env, /OROTITAN_BENCHMARK_RUN_MODE=DISABLED/);
});
