import assert from 'node:assert/strict';
import test from 'node:test';
import {
  benchmarkExecutionProfile,
  BENCHMARK_CAMPAIGN_ID,
  BENCHMARK_PHASE_ID,
} from '../lib/orotitan-equity/benchmark/execution-profile';

test('benchmark execution profile is deterministic and configuration-sensitive', () => {
  const a = benchmarkExecutionProfile({ model: 'openai/gpt-example', reasoning: 'medium' });
  const b = benchmarkExecutionProfile({ model: 'openai/gpt-example', reasoning: 'medium' });
  const c = benchmarkExecutionProfile({ model: 'openai/gpt-example', reasoning: 'high' });
  assert.equal(a.execution_profile_id, b.execution_profile_id);
  assert.notEqual(a.execution_profile_id, c.execution_profile_id);
  assert.match(a.execution_profile_id, /^[a-f0-9]{64}$/);
  assert.equal(a.campaign_id, BENCHMARK_CAMPAIGN_ID);
  assert.equal(a.phase_id, BENCHMARK_PHASE_ID);
});
