import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import test from 'node:test';
import {
 getContractProfileSummary,listContractProfileSummaries,
 requireExplicitProfile,verifyPinnedContractBytes,
} from '../lib/orotitan-equity/product/runfactory-contract-profile-v1';
import type {GithubBlobFetcher} from
 '../lib/orotitan-equity/v1/contract-pin-resolver';

const fetchLocalPinned:GithubBlobFetcher=async req=>{
 assert.equal(req.repository,'robzer13/indice_nexus');
 assert.match(req.commitSha,/^[a-f0-9]{40}$/);
 assert.equal(req.path.includes('..'),false);
 return readFile(req.path);
};

test('P0-A: exactly three explicit, immutable profile choices (13/13/15)',()=>{
 const catalog=listContractProfileSummaries();
 assert.deepEqual(catalog.map(x=>x.profileId),
  ['OROTITAN_V1','OROTITAN_V2','OROTITAN_V3_1']);
 assert.deepEqual(catalog.map(x=>x.pinCount),[13,13,15]);
 assert.deepEqual(catalog.map(x=>x.contractSetSha256),[
  '34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63',
  '1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e',
  '3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2',
 ]);
 for(const profile of catalog){
  assert.equal(profile.structuralStatus,'STRUCTURE_AND_SET_HASH_VERIFIED');
  assert.equal(profile.originalBytesStatus,'NOT_VERIFIED');
  assert.equal(profile.registryAdmitted,false);
  assert.equal(profile.issuerResearchAuthorized,false);
  assert.equal(new Set(profile.pins.map(x=>x.logicalName)).size,profile.pinCount);
  assert.equal(profile.pins.every(x=>/^[a-f0-9]{64}$/.test(x.contentSha256)),true);
 }
});

test('P0-A: reject missing/unknown/implicit version selection (never pick newest)',()=>{
 for(const value of [undefined,null,'','LATEST','V3.1','L_ORÉAL',
  'OROTITAN_V3_1\n']){
  assert.throws(()=>requireExplicitProfile(value),/EXPLICIT_CONTRACT_PROFILE_REQUIRED/);
 }
 assert.equal(requireExplicitProfile('OROTITAN_V1'),'OROTITAN_V1');
});

test('P0-A: every authority resolves canonical bytes and pinned Git blob integrity',async()=>{
 for(const profileId of ['OROTITAN_V1','OROTITAN_V2','OROTITAN_V3_1'] as const){
  const profile=getContractProfileSummary(profileId);
  const checked=new Set<string>();
  for(const pin of profile.pins){
   const result=await verifyPinnedContractBytes(
    profileId,pin.logicalName,fetchLocalPinned);
   assert.equal(result.resolvedSha256,pin.contentSha256);
   assert.equal(result.expectedSha256,pin.contentSha256);
   assert.equal(result.status,'ORIGINAL_CONTRACT_BYTES_VERIFIED');
   assert.equal(result.registryAdmitted,false);
   assert.ok(result.resolvedBytes>0);
   checked.add(result.logicalName);
  }
  assert.equal(checked.size,profile.pinCount);
 }
});

test('P0-A: reject tampered or missing bytes, missing pin and mixed-profile key',async()=>{
 const malicious:GithubBlobFetcher=async req=>{
  const bytes=await fetchLocalPinned(req);
  const mutated=Uint8Array.from(bytes);
  mutated[0]^=1;
  return mutated;
 };
 await assert.rejects(()=>verifyPinnedContractBytes(
  'OROTITAN_V1','process',malicious),/git blob mismatch/);
 await assert.rejects(()=>verifyPinnedContractBytes(
  'OROTITAN_V2','process',async()=>{throw Error('unavailable');}),/unavailable/);
 await assert.rejects(()=>verifyPinnedContractBytes(
  'OROTITAN_V1','dcf_timing',fetchLocalPinned),
  /CONTRACT_PIN_NOT_IN_SELECTED_PROFILE/);
 await assert.rejects(()=>verifyPinnedContractBytes(
  'OROTITAN_V1','UNKNOWN',fetchLocalPinned),
  /CONTRACT_PIN_NOT_IN_SELECTED_PROFILE/);
});
