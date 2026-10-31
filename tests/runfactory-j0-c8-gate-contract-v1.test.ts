import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {join} from 'node:path';

const root=process.cwd();
const migration=readFileSync(join(root,
 'migrations/20261031_orotitan_j0_c8_human_acceptance_shadow.sql'),'utf8');
const acceptRoute=readFileSync(join(root,
 'app/admin/research-factory/control-tower/c8-gate/accept/route.ts'),'utf8');
const prepareRoute=readFileSync(join(root,
 'app/admin/research-factory/control-tower/c8-gate/prepare/route.ts'),'utf8');
const checkRoute=readFileSync(join(root,
 'app/admin/research-factory/control-tower/c8-gate/route.ts'),'utf8');

test('C8 SQL is SHADOW-only and seeds a DISABLED acceptance lock',()=>{
 assert.match(migration,/VALUES \('GLOBAL_SHADOW',false\)/);
 assert.match(migration,/DEFAULT false/);
 assert.match(migration,/J0_C8_ACCEPTANCE_LOCKED/);
 assert.match(migration,/GRANT SELECT ON public\.orotitan_j0_c8_decisions_v1 TO service_role/);
 assert.doesNotMatch(migration,/GRANT INSERT ON public\.orotitan_j0_c8_decisions_v1 TO service_role/i);
 assert.doesNotMatch(migration,/UPDATE\s+public\.orotitan_runs/i);
 assert.doesNotMatch(migration,/UPDATE\s+public\.orotitan_j0_milestones_v1/i);
});
test('C8 acceptance requires separate runtime enablement as well as DB GO',()=>{
 assert.match(acceptRoute,/OROTITAN_C8_ACCEPTANCE_GO_ENABLED!=='1'/);
 assert.match(acceptRoute,/orotitan_j0_c8_acceptance_control_v1/);
 assert.match(acceptRoute,/activation_authority_ref!==p\.authorityRef/);
 assert.match(acceptRoute,/requireAdminSession\(\)/);
 assert.match(acceptRoute,/ORIGIN_MISMATCH/);
 assert.match(acceptRoute,/persistedReadbackVerified:true/);
});
test('C8 prepare accepts exactly 7 pinned proof IDs and independent receipt',()=>{
 assert.match(prepareRoute,/v\.evidenceEventIds\.length!==7/);
 assert.match(prepareRoute,/orotitan_j0_c8_prepare_review_v1/);
 assert.match(prepareRoute,/J0_C8_REVIEW_READBACK_MISMATCH/);
 assert.match(prepareRoute,/accepted:false/);
 assert.match(migration,/J0_C8_PROOF_MISSING_OR_INVALID_C%/);
});
test('C8 gate check is nonmutating, keeps a separate accepted decision projection',()=>{
 assert.match(checkRoute,/orotitan_j0_c8_read_gate_v1/);
 assert.match(checkRoute,/orotitan_j0_c8_decisions_v1/);
 assert.match(checkRoute,/humanGoRequired:true/);
 assert.match(migration,/humanAcceptanceRequired',true/);
 assert.match(migration,/J0_C8_REVIEW_STALE/);
});
test('C8 records both reviews and decisions immutably with service-only RPC',()=>{
 assert.match(migration,/orotitan_j0_c8_reviews_append_only/);
 assert.match(migration,/orotitan_j0_c8_decisions_append_only/);
 assert.match(migration,/REVOKE ALL ON FUNCTION public\.orotitan_j0_c8_accept_review_v1/);
 assert.match(migration,/GRANT EXECUTE ON FUNCTION public\.orotitan_j0_c8_accept_review_v1/);
 assert.match(migration,/J0_C8_DECISION_IDEMPOTENCY_CONFLICT/);
});
