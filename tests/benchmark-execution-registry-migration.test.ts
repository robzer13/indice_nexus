import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const migrationPath = 'migrations/20261004_orotitan_engine_benchmark_execution_registry.sql';

test('benchmark registry is append-only and service-role RPC bounded', async () => {
  const sql = await readFile(migrationPath, 'utf8');
  assert.match(sql, /create table if not exists public\.orotitan_engine_benchmark_executions/i);
  assert.match(sql, /unique \(campaign_id, phase_id, case_id, repetition_index, execution_profile_id\)/i);
  assert.match(sql, /before update or delete on public\.orotitan_engine_benchmark_executions/i);
  assert.match(sql, /revoke all on table public\.orotitan_engine_benchmark_executions[\s\S]*service_role/i);
  assert.match(sql, /grant select on table public\.orotitan_engine_benchmark_executions to service_role/i);
  assert.match(sql, /security definer/i);
  assert.match(sql, /grant execute on function public\.record_orotitan_benchmark_execution\(jsonb\) to service_role/i);
  assert.doesNotMatch(sql, /grant insert on table public\.orotitan_engine_benchmark_executions to service_role/i);
});
