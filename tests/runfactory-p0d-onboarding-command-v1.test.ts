import assert from 'node:assert/strict';
import test from 'node:test';
import {buildP0DOnboardingCommand,type P0DInput} from
 '../lib/orotitan-equity/product/runfactory-p0d-onboarding-command-v1';

const baseline:P0DInput={
 isin:'FR0000120321',ticker:'OR',exchange:'XPAR',
 displayName:'L’Oréal',legalName:'L’Oréal S.A.',
 issuerCountry:'FR',reportingCurrency:'EUR',tradingCurrency:'EUR',
 primaryListing:true,quoteUnit:'MAJOR',priceDecimals:2,
 legacyCompanyId:null,existingIssuerId:null,
 issuerSourceUrl:'https://www.loreal-finance.com/fr/laction',
 listingSourceUrl:'https://live.euronext.com/fr/product/equities/fr0000120321-xpar/overview',
 reviewedAt:'2026-10-31T12:00:00.000Z',
 confirmedIdentity:true,confirmation:'ONBOARD FR0000120321',
};
const now=new Date('2026-10-31T12:05:00.000Z');

test('P0-D: explicit manual operator authority and reproducible idempotency key',()=>{
 const a=buildP0DOnboardingCommand(baseline,now);
 const b=buildP0DOnboardingCommand(baseline,now);
 assert.deepEqual(a,b);
 assert.match(a.p_idempotency_key,/^orotitan:p0d:shadow:[a-f0-9]{64}$/);
 assert.equal(a.p_evidence.method,'OPERATOR_PRIMARY_SOURCE_REVIEW');
 assert.equal(a.p_evidence.reviewed_by,'ADMIN_SESSION_MANUAL_ATTESTATION_V1');
 assert.equal(a.p_operator_attestation,'APPROVE_SHADOW_CANONICAL_IDENTITY_V1');
 assert.equal('runCreated' in a,false);
 assert.equal('service_role' in a,false);
});
test('P0-D: no implicit GO from valid ISIN alone',()=>{
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  confirmedIdentity:false},now),/P0D_REVIEW_REQUIRED/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  confirmation:'ONBOARD OR'},now),/P0D_CONFIRMATION_INVALID/);
});
test('P0-D: rejects external/reused source host and SSRF-like URLs',()=>{
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  issuerSourceUrl:'http://www.loreal-finance.com/fr/laction'},now),
  /P0D_SOURCE_INVALID/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  issuerSourceUrl:'https://127.0.0.1/a'},now),/P0D_SOURCE_INVALID/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  listingSourceUrl:'https://www.loreal-finance.com/fr/another'},now),
  /P0D_SOURCES_NOT_INDEPENDENT/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  issuerSourceUrl:'https://user:pass@example.org/private'},now),
  /P0D_SOURCE_INVALID/);
});
test('P0-D: same reviewer and sources but new legal name changes request key',()=>{
 const initial=buildP0DOnboardingCommand(baseline,now);
 const changed=buildP0DOnboardingCommand({...baseline,
  legalName:'L’Oréal Holding SA'},now);
 assert.notEqual(initial.p_idempotency_key,changed.p_idempotency_key);
});
test('P0-D: reject invalid currencies, ISIN, price precision and ID overrides',()=>{
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  isin:'FR0000120322'},now),/INVALID_ISIN_CHECK_DIGIT/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  reportingCurrency:'EURO'},now),/P0D_TEXT_INVALID|P0D_METADATA_INVALID/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  priceDecimals:7},now),/P0D_METADATA_INVALID/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  existingIssuerId:'invented-uuid'},now),/P0D_ID_INVALID/);
});
test('P0-D: the source review must be recent and nonfuture',()=>{
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  reviewedAt:'2026-10-01T12:00:00.000Z'},now),
  /P0D_REVIEW_TIMESTAMP_INVALID/);
 assert.throws(()=>buildP0DOnboardingCommand({...baseline,
  reviewedAt:'2026-11-01T12:00:00.000Z'},now),
  /P0D_REVIEW_TIMESTAMP_INVALID/);
});
