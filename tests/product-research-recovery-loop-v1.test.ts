import test from 'node:test';
import assert from 'node:assert/strict';
import {
  runRecoveryCycle, type RecoveryAttempt, type RecoveryInput, type RecoveryPort,
  type RecoveryStrategy,
} from '../lib/orotitan-equity/product/research-recovery-loop-v1';

const input: RecoveryInput = {
  runId:'f4ae40f7-eb9b-4472-b124-9fb152b739ce',
  runStatus:'ACTIVE',stageCode:'DEEP_DIVE',stageLifecycle:'IN_PROGRESS',
  runVersion:6,stageVersion:1,
  blockers:[
    {code:'ASML-G001',reason:'Economic share count unknown'},
    {code:'ASML-G006',reason:'Historical close not admitted'},
    {code:'DD-PRIMARY-SOURCES',reason:'Primary source bytes not admitted'},
    {code:'DD-ROIC-OE',reason:'Operating capital and owner earnings unverified'},
  ],
};
function fixture() {
  const stored = new Map<string,RecoveryAttempt[]>();
  const calls: RecoveryStrategy[] = [];
  const key = (x:RecoveryInput) => [x.runId,x.stageCode,x.runVersion,x.stageVersion].join(':');
  const port: RecoveryPort = {
    async listAttempts(x) { return stored.get(key(x)) ?? []; },
    async recordAttempt(x,a) {
      const arr = stored.get(key(x)) ?? [];
      if (!arr.some(v=>v.blockerCode===a.blockerCode && v.strategy===a.strategy)) arr.push(a);
      stored.set(key(x),arr);
    },
    async investigate(_x,_b,strategy) {
      calls.push(strategy);
      return strategy === 'REUSE_PRIMARY_ARTIFACTS'
        ? {outcome:'CANDIDATE_RECORDED' as const,detail:'Cached SHA candidate found (not admitted).',evidenceRefs:['sha256:abc']}
        : {outcome:'NO_PROGRESS' as const,detail:'No admitted evidence in this read-only pass.',evidenceRefs:[]};
    },
  };
  return {port,calls,stored};
}
test('ASML schedules source recovery first and persists each attempt without certifying anything',async()=>{
  const f=fixture();
  const first=await runRecoveryCycle(input,f.port);
  assert.equal(first.status,'PROGRESSED');
  assert.deepEqual(f.calls,['REUSE_PRIMARY_ARTIFACTS','RECONCILE_SHARE_COUNT','RECONCILE_HISTORICAL_PRICE','REBUILD_ROIC_OWNER_EARNINGS']);
  assert.equal(first.attempted.length,4);
  assert.equal(first.remaining.length,4);
  assert.equal(first.publicationAuthorized,false);
  assert.equal(first.registryChanged,false);
  assert.equal(f.stored.size,1);
});
test('fresh repeat uses persisted attempts and avoids duplicate investigations',async()=>{
  const f=fixture();
  await runRecoveryCycle(input,f.port);
  const second=await runRecoveryCycle({...input},f.port);
  assert.equal(second.attempted.length,0);
  assert.equal(second.status,'NO_NEW_AUTOMATED_STEPS');
  assert.equal(f.calls.length,4);
});
test('bounded cycles resume at next unattempted blocker',async()=>{
  const f=fixture();
  assert.equal((await runRecoveryCycle(input,f.port,2)).attempted.length,2);
  assert.equal((await runRecoveryCycle(input,f.port,2)).attempted.length,2);
  assert.equal((await runRecoveryCycle(input,f.port,2)).attempted.length,0);
  assert.equal(f.calls.length,4);
});
test('state version changes reopens attempts without erasing old journal',async()=>{
  const f=fixture();
  await runRecoveryCycle(input,f.port,1);
  const next={...input,stageVersion:2};
  await runRecoveryCycle(next,f.port,1);
  assert.equal(f.calls.length,2);
  assert.equal(f.stored.size,2);
});
test('run/contractual pause is a hard stop even when tasks exist',async()=>{
  const f=fixture();
  const result=await runRecoveryCycle({...input,runStatus:'BLOCKED'},f.port);
  assert.equal(result.status,'PAUSED');
  assert.equal(f.calls.length,0);
});
test('unknown blockers require review, not invented auto resolution',async()=>{
  const f=fixture();
  const result=await runRecoveryCycle({...input,blockers:[{code:'AUTHORITATIVE_CONTRACT_DRIFT',reason:'Bad hash'}]},f.port);
  assert.equal(result.attempted[0].outcome,'REQUIRES_AUTHORITY');
  assert.equal(f.calls.length,0);
});
test('temporary resolver failure is retryable and does not persist a false success',async()=>{
  let calls=0;
  const port: RecoveryPort={
    async listAttempts(){return [];},
    async recordAttempt(){throw Error('should not be reached');},
    async investigate(){calls++;throw Error('temporary network failure');},
  };
  const r=await runRecoveryCycle(input,port,1);
  assert.equal(r.status,'ERROR');
  assert.equal(calls,1);
});
test('reject invalid budget, duplicate blockers and malformed run revision',async()=>{
  const f=fixture();
  await assert.rejects(runRecoveryCycle(input,f.port,0),/INVALID_BUDGET/);
  await assert.rejects(runRecoveryCycle({...input,stageVersion:-1},f.port),/INVALID_INPUT/);
  await assert.rejects(runRecoveryCycle({...input,blockers:[input.blockers[0],input.blockers[0]]},f.port),/INVALID_INPUT/);
});
