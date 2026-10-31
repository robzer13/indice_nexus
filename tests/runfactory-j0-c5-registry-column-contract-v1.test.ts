import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';

/**
 * Regression for the SHADOW incident on 2026-10-31.
 * PostgREST count queries must select ACTUAL Registry columns.
 * The historical orotitan_run_stages schema has NO stage_id column.
 */
const route=readFileSync(join(process.cwd(),
 'app/admin/research-factory/control-tower/c5-security-test/route.ts'),'utf8');

test('C5 uses real immutable Registry schema columns for read-only counts',()=>{
 assert.match(route,/client\.from\('orotitan_run_stages'\)\s*\.select\('run_id',\{count:'exact',head:true\}\)/);
 assert.doesNotMatch(route,/client\.from\('orotitan_run_stages'\)\s*\.select\('stage_id'/);
 assert.match(route,/client\.from\('orotitan_artifacts'\)\s*\.select\('artifact_id',\{count:'exact',head:true\}\)/);
 assert.match(route,/client\.from\('research_snapshots'\)\s*\.select\('snapshot_id',\{count:'exact',head:true\}\)/);
});

test('C5 distinguishes query failures from genuine run state mismatches',()=>{
 for(const name of [
  'J0_C5_RUN_READ_FAILED','J0_C5_STAGE_READ_FAILED',
  'J0_C5_ARTIFACT_READ_FAILED','J0_C5_SNAPSHOT_READ_FAILED',
  'J0_C5_RUN_STATE_MISMATCH',
 ])assert.ok(route.includes(name),name);
 assert.ok(!route.includes("throw Error('J0_C5_RUN_INVARIANT_FAILED')"));
});
