import test from 'node:test';
import assert from 'node:assert/strict';
import {
 planCanonicalIdentityOnboarding,type P0CInventory,
} from '../lib/orotitan-equity/product/runfactory-canonical-onboarding-candidate-v1';

const issuerId='11111111-1111-1111-1111-111111111111';
const securityId='22222222-2222-2222-2222-222222222222';
const dossierId='33333333-3333-3333-3333-333333333333';
const legacyId='44444444-4444-4444-4444-444444444444';
const blank:P0CInventory={
 companies:[],issuers:[],securities:[],dossiers:[],maps:[],
};
const loreal={isin:'FR0000120321',ticker:'OR',exchange:'XPAR'};
const legacy:P0CInventory={...blank,companies:[{
 id:legacyId,name:'L’Oréal',ticker:'OR',exchange:'XPAR',
 currency:'EUR',active:true,country:'FR',
}]};
const canonical:P0CInventory={
 companies:legacy.companies,
 issuers:[{issuerId,displayName:'L’Oréal',legalName:'L’Oréal S.A.'}],
 securities:[{securityId,issuerId,isin:loreal.isin,ticker:'OR',
  exchange:'XPAR',tradingCurrency:'EUR',primaryListing:true}],
 dossiers:[{dossierId,issuerId,active:true}],
 maps:[{legacyCompanyId:legacyId,securityId,issuerId,dossierId}],
};

test('P0-C: missing company generates no UUID, no writes and no verified claim',()=>{
 const x=planCanonicalIdentityOnboarding(loreal,blank);
 assert.equal(x.status,'SOURCE_AND_METADATA_REVIEW_REQUIRED');
 assert.equal(x.proposedDisplayName,null);
 assert.equal(x.existingIssuerId,null);
 assert.equal(x.evidenceStatus,'NO_SOURCE_REFERENCE');
 assert.equal(x.registryMutated,false);
 assert.equal(x.productionTouched,false);
 assert.equal(x.rpcExecuted,false);
 assert.equal(x.operationAuthorized,false);
 assert.equal(x.proposedWrites.length,1);
 assert.deepEqual(x,planCanonicalIdentityOnboarding(loreal,blank));
});
test('P0-C: legacy Screener only suggests metadata, never proves the legal issuer',()=>{
 const x=planCanonicalIdentityOnboarding(loreal,legacy);
 assert.equal(x.status,'SOURCE_AND_METADATA_REVIEW_REQUIRED');
 assert.equal(x.proposedDisplayName,'L’Oréal');
 assert.equal(x.tradingCurrencyCandidate,'EUR');
 assert.equal(x.legacyCompanyId,legacyId);
 assert.equal(x.metadataSource,'LEGACY_SCREENER_UNVERIFIED');
 assert.equal(x.needsIndependentLegalAndListingVerification,true);
 assert.equal(x.existingSecurityId,null);
});
test('P0-C: existing canonical identity reused, no identity insert allowed',()=>{
 const x=planCanonicalIdentityOnboarding(loreal,canonical);
 assert.equal(x.status,'EXISTING_SECURITY_REUSE_CANDIDATE');
 assert.equal(x.existingIssuerId,issuerId);
 assert.equal(x.existingSecurityId,securityId);
 assert.equal(x.existingDossierId,dossierId);
 assert.equal(x.proposedWrites.length,0);
});
test('P0-C: conflicting exchange/ticker ISIN blocks, never silently rewrites',()=>{
 const x=planCanonicalIdentityOnboarding(loreal,{
  ...canonical,securities:[{...canonical.securities[0],isin:'FR0000120322'}],
  maps:[],
 });
 assert.equal(x.status,'BLOCKED_SECURITY_COLLISION');
 assert.equal(x.proposedWrites.length,0);
});
test('P0-C: existing ISIN on another listing needs issuer review, not new issuer',()=>{
 const x=planCanonicalIdentityOnboarding({...loreal,exchange:'XAMS'},{
  ...blank,securities:canonical.securities,issuers:canonical.issuers,
 });
 assert.equal(x.status,'EXISTING_ISIN_OTHER_LISTING_REVIEW');
 assert.equal(x.registryMutated,false);
});
test('P0-C: detect orphaned or contradictory crosswalk and dossier',()=>{
 assert.equal(planCanonicalIdentityOnboarding(loreal,{
  ...legacy,maps:canonical.maps,
 }).status,'BLOCKED_LEGACY_MAPPING_CONFLICT');
 assert.equal(planCanonicalIdentityOnboarding(loreal,{
  ...canonical,maps:[{...canonical.maps[0],securityId:legacyId}],
 }).status,'BLOCKED_LEGACY_MAPPING_CONFLICT');
 assert.equal(planCanonicalIdentityOnboarding(loreal,{
  ...canonical,dossiers:[],
 }).status,'BLOCKED_CANONICAL_INCONSISTENT');
});
test('P0-C: matching issuer name is a REVIEW hint, not an issuer identity proof',()=>{
 const x=planCanonicalIdentityOnboarding(
  {...loreal,issuerDisplayName:'L’Oréal'},{
   ...blank,issuers:canonical.issuers,
  });
 assert.equal(x.status,'EXISTING_ISSUER_NAME_REVIEW');
 assert.equal(x.existingIssuerId,issuerId);
 assert.equal(x.needsIndependentLegalAndListingVerification,true);
 const ambiguous=planCanonicalIdentityOnboarding(
  {...loreal,issuerDisplayName:'L’Oréal'},{
   ...blank,issuers:[...canonical.issuers,{
    issuerId:legacyId,displayName:'L’Oréal',legalName:null,
   }],
  });
 assert.equal(ambiguous.status,'BLOCKED_NAME_AMBIGUOUS');
});
test('P0-C: duplicate legacy company blocks and source URL is only pointer',()=>{
 assert.equal(planCanonicalIdentityOnboarding(loreal,{
  ...legacy,companies:[...legacy.companies,
   {...legacy.companies[0],id:securityId}],
 }).status,'BLOCKED_LEGACY_AMBIGUOUS');
 const x=planCanonicalIdentityOnboarding({
  ...loreal,evidenceUrl:'https://example.com/issuer/isin',
 },legacy);
 assert.equal(x.evidenceStatus,'USER_SUPPLIED_POINTER_NOT_FETCHED');
 assert.equal(x.needsIndependentLegalAndListingVerification,true);
 assert.throws(()=>planCanonicalIdentityOnboarding({
  ...loreal,evidenceUrl:'http://localhost/file',
 },legacy),/INVALID_EVIDENCE_URL/);
 assert.throws(()=>planCanonicalIdentityOnboarding({
  ...loreal,evidenceUrl:'https://127.0.0.1/a',
 },legacy),/INVALID_EVIDENCE_URL/);
});
test('P0-C: malformed ISIN and unsafe ticker/exchange fail closed',()=>{
 assert.throws(()=>planCanonicalIdentityOnboarding(
  {...loreal,isin:'FR0000120322'},blank),/INVALID_ISIN_CHECK_DIGIT/);
 assert.throws(()=>planCanonicalIdentityOnboarding(
  {...loreal,ticker:'OR%_foo'},blank),/IDENTITY_LISTING_INPUT_INVALID/);
 assert.throws(()=>planCanonicalIdentityOnboarding(
  {...loreal,exchange:'other;drop'},blank),/IDENTITY_LISTING_INPUT_INVALID/);
});
