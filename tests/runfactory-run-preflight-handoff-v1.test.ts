import assert from 'node:assert/strict';
import test from 'node:test';
import {
 parseUntrustedRunPreflightQuery,buildUntrustedRunPreflightHandoff,
} from '../lib/orotitan-equity/product/runfactory-run-preflight-handoff-v1';

test('P0-D -> P0-B: only valid identity and cutoff are transmitted; profile never selected',()=>{
 const url=buildUntrustedRunPreflightHandoff({
  isin:'FR0000120321',exchange:'XPAR',dataCutoff:'2026-10-30',
 });
 const params=Object.fromEntries(new URL('http://local'+url).searchParams);
 assert.deepEqual(parseUntrustedRunPreflightQuery(params),{
  isin:'FR0000120321',exchange:'XPAR',dataCutoff:'2026-10-30',
  profileId:'',
 });
 assert.ok(!url.includes('profileId'));
 assert.ok(!url.includes('goAuthorized'));
});
test('P0-D success can transfer ISIN/market without inventing a future cutoff',()=>{
 const url=buildUntrustedRunPreflightHandoff({
  isin:'FR0000120321',exchange:'XPAR',
 });
 const params=Object.fromEntries(new URL('http://local'+url).searchParams);
 assert.deepEqual(parseUntrustedRunPreflightQuery(params),{
  isin:'FR0000120321',exchange:'XPAR',dataCutoff:'',
  profileId:'',
 });
});
test('Bad ISIN, listing and date fail closed independently',()=>{
 const seed=parseUntrustedRunPreflightQuery({
  isin:'FR0000120322',exchange:'XPAR;DROP',
  dataCutoff:'2026-02-31',profileId:'OROTITAN_V3_1',
 });
 assert.deepEqual(seed,{isin:'',exchange:'',dataCutoff:'',profileId:''});
});
test('Untrusted handoff never selects profile, even for current methodology',()=>{
 const params=parseUntrustedRunPreflightQuery({
  isin:'FR0000120321',exchange:'xpar',dataCutoff:'2026-10-30',
  profileId:'OROTITAN_V3_1',go:'TRUE',rpc:'create_orotitan_run',
 });
 assert.equal(params.profileId,'');
 assert.equal(params.exchange,'XPAR');
 assert.equal('rpc' in params,false);
});
