import test from 'node:test';
import assert from 'node:assert/strict';
import {
 assertIsin,inspectRunFactoryIdentity,type RunFactoryInventory,
} from '../lib/orotitan-equity/product/runfactory-identity-preflight-v1';
import {getContractProfileSummary} from
 '../lib/orotitan-equity/product/runfactory-contract-profile-v1';

const issuerId='11111111-1111-1111-1111-111111111111';
const securityId='22222222-2222-2222-2222-222222222222';
const dossierId='33333333-3333-3333-3333-333333333333';
const snapshotId='44444444-4444-4444-4444-444444444444';
const runId='55555555-5555-5555-5555-555555555555';
const profile=getContractProfileSummary('OROTITAN_V3_1');
const input={isin:'FR0000120321',exchange:null,dataCutoff:'2026-10-30',
 profileId:'OROTITAN_V3_1' as const};
const fixture:RunFactoryInventory={
 securities:[{issuerId,securityId,isin:'FR0000120321',exchange:'XPAR',
  primaryListing:true}],
 issuers:[{issuerId,displayName:'L’Oréal'}],
 dossiers:[{issuerId,dossierId,active:true,currentSnapshotId:null}],
 snapshots:[],runs:[],
};

test('P0-B validates ISIN checksum (ASML and L’Oreal) and fails on typo',()=>{
 assert.equal(assertIsin('fr0000120321'),'FR0000120321');
 assert.equal(assertIsin('NL0010273215'),'NL0010273215');
 assert.throws(()=>assertIsin('FR0000120322'),/INVALID_ISIN_CHECK_DIGIT/);
 assert.throws(()=>assertIsin('FR0000120321;DROP TABLE'),/INVALID_ISIN/);
});

test('P0-B new initial candidate: pure, stable, NO RPC or GO',()=>{
 const a=inspectRunFactoryIdentity(input,fixture);
 const b=inspectRunFactoryIdentity(input,fixture);
 assert.deepEqual(a,b);
 assert.equal(a.status,'NEW_RUN_PLAN_CANDIDATE');
 assert.equal(a.proposedRunType,'INITIAL');
 assert.equal(a.contractSetSha256,profile.contractSetSha256);
 assert.equal(a.rpcExecuted,false);
 assert.equal(a.goAuthorized,false);
 assert.equal(a.runCreated,false);
 assert.equal(a.candidateCreationIdempotencyKey,null);
 assert.equal(a.planExecutableInShadow,false);
});

test('P0-B missing issuer and missing active dossier fail closed',()=>{
 const missing=inspectRunFactoryIdentity(input,{
  ...fixture,securities:[],issuers:[],dossiers:[],
 });
 assert.equal(missing.status,'BLOCKED_SECURITY_NOT_FOUND');
 const dossierMissing=inspectRunFactoryIdentity(input,{
  ...fixture,dossiers:[],
 });
 assert.equal(dossierMissing.status,'BLOCKED_DOSSIER_NOT_FOUND');
 const missingIssuer=inspectRunFactoryIdentity(input,{
  ...fixture,issuers:[],
 });
 assert.equal(missingIssuer.status,'BLOCKED_ISSUER_INCOMPLETE');
});

test('P0-B ambiguity does not pick arbitrary exchange or active dossier',()=>{
 const alt={...fixture.securities[0],securityId:runId,exchange:'XAMS'};
 assert.equal(inspectRunFactoryIdentity(input,{
  ...fixture,securities:[...fixture.securities,alt],
 }).status,'BLOCKED_SECURITY_AMBIGUOUS');
 assert.equal(inspectRunFactoryIdentity({...input,exchange:'XPAR'},{
  ...fixture,securities:[...fixture.securities,alt],
 }).status,'NEW_RUN_PLAN_CANDIDATE');
 assert.equal(inspectRunFactoryIdentity(input,{
  ...fixture,dossiers:[...fixture.dossiers,
   {...fixture.dossiers[0],dossierId:runId}],
 }).status,'BLOCKED_DOSSIER_AMBIGUOUS');
});

test('P0-B snapshot guarantees consistent issuer, security, dossier and contract',()=>{
 const next={...fixture,
  dossiers:[{...fixture.dossiers[0],currentSnapshotId:snapshotId}],
  snapshots:[{snapshotId,issuerId,dossierId,securityId,
   contractVersion:'3.1'}]};
 const d=inspectRunFactoryIdentity(input,next);
 assert.equal(d.status,'NEW_RUN_PLAN_CANDIDATE');
 assert.equal(d.proposedRunType,'REFRESH');
 assert.equal(d.baselineSnapshotId,snapshotId);
 assert.equal(inspectRunFactoryIdentity(input,{
  ...next,snapshots:[{...next.snapshots[0],securityId:runId}],
 }).status,'BLOCKED_SNAPSHOT_MISMATCH');
 assert.equal(inspectRunFactoryIdentity(input,{
  ...next,snapshots:[{...next.snapshots[0],contractVersion:null}],
 }).status,'BLOCKED_SNAPSHOT_MISMATCH');
});

const existing={
 runId,issuerId,securityId,dossierId,dataCutoff:'2026-10-30',
 contractSetSha256:profile.contractSetSha256,runStatus:'ACTIVE',
 currentStage:'RESEARCH',runType:'INITIAL',creationIdempotencyKey:'key',
};

test('P0-B existing run is REUSED candidate, never duplicated',()=>{
 const reused=inspectRunFactoryIdentity(input,{
  ...fixture,runs:[existing],
 });
 assert.equal(reused.status,'EXISTING_RUN_REUSE_CANDIDATE');
 assert.equal(reused.existingRunId,runId);
 assert.equal(reused.rpcExecuted,false);
});

test('P0-B immutable contract conflict and duplicate runs both BLOCK',()=>{
 assert.equal(inspectRunFactoryIdentity(input,{
  ...fixture,runs:[{...existing,contractSetSha256:'a'.repeat(64)}],
 }).status,'BLOCKED_EXISTING_METHODOLOGY_CONFLICT');
 assert.equal(inspectRunFactoryIdentity(input,{
  ...fixture,runs:[existing,{...existing,runId:snapshotId}],
 }).status,'BLOCKED_EXISTING_RUN_AMBIGUOUS');
 assert.equal(inspectRunFactoryIdentity(input,{
  ...fixture,runs:[{...existing,securityId:snapshotId}],
 }).status,'BLOCKED_EXISTING_RUN_IDENTITY_MISMATCH');
 assert.equal(inspectRunFactoryIdentity(input,{
  ...fixture,runs:[{...existing,runStatus:'CANCELLED'}],
 }).status,'NEW_RUN_PLAN_CANDIDATE');
});
test('P0-B never derives contract profile from ISIN or silently accepts cutoff',()=>{
 assert.throws(()=>inspectRunFactoryIdentity({...input,profileId:'LATEST' as
  typeof input.profileId},fixture),/EXPLICIT_CONTRACT_PROFILE_REQUIRED/);
 assert.throws(()=>inspectRunFactoryIdentity({
  ...input,dataCutoff:'2026-02-31',
 },fixture),/INVALID_DATA_CUTOFF/);
});
