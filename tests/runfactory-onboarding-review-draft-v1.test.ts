import assert from 'node:assert/strict';
import test from 'node:test';
import {
 parseUntrustedOnboardingDraft,buildUntrustedReviewDraftQuery,
} from '../lib/orotitan-equity/product/runfactory-onboarding-review-draft-v1';

const draft={
 isin:'FR0000120321',
 ticker:'OR',exchange:'XPAR',
 displayName:'L’Oréal',legalName:'L’Oréal S.A.',
 issuerCountry:'FR',reportingCurrency:'EUR',tradingCurrency:'EUR',
 primaryListing:true,quoteUnit:'MAJOR' as const,priceDecimals:2,
 issuerSourceUrl:'https://www.loreal-finance.com/fr/laction',
 listingSourceUrl:'https://live.euronext.com/fr/product/equities/fr0000120321-xpar/overview',
};
const query=Object.fromEntries(new URLSearchParams(buildUntrustedReviewDraftQuery(draft)));

test('review draft transports issuer metadata without code/Registry writes',()=>{
 const params=new URLSearchParams(buildUntrustedReviewDraftQuery(draft));
 const result=parseUntrustedOnboardingDraft(Object.fromEntries(params));
 assert.deepEqual(result,draft);
 assert.equal('confirmedIdentity' in result!,false);
 assert.equal('reviewedAt' in result!,false);
 assert.equal('confirmation' in result!,false);
 assert.equal('rpcExecuted' in result!,false);
});
test('ISIN and exchange are validated; cannot inject dangerous URLs',()=>{
 assert.equal(parseUntrustedOnboardingDraft({...query,isin:'FR0000120322'}),null);
 assert.equal(parseUntrustedOnboardingDraft({...query,exchange:'XPAR;DELETE'}),null);
 const result=parseUntrustedOnboardingDraft({
  ...query,issuerSourceUrl:'javascript:alert(1)',
  listingSourceUrl:'https://127.0.0.1/private',
 });
 assert.equal(result?.issuerSourceUrl,'');
 assert.equal(result?.listingSourceUrl,'');
});
test('unknown currency is never auto-filled from ISIN or name',()=>{
 const q=parseUntrustedOnboardingDraft({
  isin:draft.isin,ticker:draft.ticker,exchange:draft.exchange,
 });
 assert.equal(q?.reportingCurrency,'');
 assert.equal(q?.tradingCurrency,'');
 assert.equal(q?.issuerCountry,'');
 assert.equal(q?.displayName,'');
});
test('invalid or absent review draft is ignored without authorizing writes',()=>{
 assert.equal(parseUntrustedOnboardingDraft({}),null);
 assert.equal(parseUntrustedOnboardingDraft({isin:['FR0000120321'],
  ticker:'OR',exchange:'XPAR'}),null);
 const p=parseUntrustedOnboardingDraft({
  ...query,primaryListing:'false',quoteUnit:'MINOR',priceDecimals:'6',
 });
 assert.equal(p?.primaryListing,false);
 assert.equal(p?.quoteUnit,'MINOR');
 assert.equal(p?.priceDecimals,6);
});
