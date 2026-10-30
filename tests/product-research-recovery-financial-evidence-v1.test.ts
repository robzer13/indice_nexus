import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ASML_RECOVERY_CUTOFF,reviewEconomicShareLedger,reviewHistoricalXamsClose,
  type ShareLedger,type HistoricalClosingQuote,type DatedPrimaryEvidence,
} from '../lib/orotitan-equity/product/research-recovery-financial-evidence-v1';

const issuer:DatedPrimaryEvidence={
  sourceId:'ASML_OFFICIAL_SHARE_LEDGER_SYNTHETIC',publicationDate:'2026-09-21',
  sha256:'a'.repeat(64),sourceUrl:'https://investor.asml.com/share-information',
};
const exchange:DatedPrimaryEvidence={
  sourceId:'EURONEXT_XAMS_HISTORICAL_EOD_SYNTHETIC',publicationDate:'2026-09-21',
  sha256:'b'.repeat(64),
  sourceUrl:'https://live.euronext.com/en/product/equities/NL0010273215-XAMS',
};
const ledger:ShareLedger={
  valuationDate:ASML_RECOVERY_CUTOFF,openingDate:'2026-06-28',
  issuedAtOpening:1000,treasuryAtOpening:100,
  openingEvidence:issuer,
  movementsCoverageVerified:true,movementsCoverageEvidence:issuer,
  movements:[
    {eventId:'BUYBACK_SETTLED',effectiveDate:'2026-08-15',shares:50,
      kind:'TREASURY_ACQUISITION_SETTLED',evidence:issuer},
    {eventId:'ISSUANCE',effectiveDate:'2026-07-20',shares:20,
      kind:'ISSUANCE',evidence:issuer},
    {eventId:'CANCEL_TREASURY',effectiveDate:'2026-09-01',shares:30,
      kind:'CANCELLATION_TREASURY',evidence:issuer},
  ],
};
const quote:HistoricalClosingQuote={
  valuationDate:ASML_RECOVERY_CUTOFF,tradingDate:'2026-09-21',
  securityId:'synthetic-canonical-security',isin:'NL0010273215',
  exchange:'XAMS',currency:'EUR',quoteUnit:'EUR_PER_SHARE',
  close:100,priceBasis:'UNADJUSTED_CLOSE',corporateActionsChecked:true,
  evidence:exchange,
};
test('G001 accounts settled treasury movements and cancellations exactly once',()=>{
  const r=reviewEconomicShareLedger(ledger);
  assert.equal(r.state,'CANDIDATE_FOR_INDEPENDENT_REVIEW');
  assert.equal(r.certified,false);
  if (r.state==='CANDIDATE_FOR_INDEPENDENT_REVIEW') {
    assert.deepEqual(r.value,{issued:990,treasury:120,economicShares:870,asOf:'2026-09-21'});
  }
});
test('G001 missing inputs, incomplete coverage, invalid timestamp and duplicate IDs block',()=>{
  assert.equal(reviewEconomicShareLedger(undefined).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,movementsCoverageVerified:false}).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,openingDate:'2026-09-22'}).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,openingEvidence:{...issuer,sourceUrl:'https://evil.com'}}).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,movements:[ledger.movements[0],ledger.movements[0]]}).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,movements:[{...ledger.movements[0],effectiveDate:'2026-10-01'}]}).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,movements:[{...ledger.movements[0],shares:5000}]}).state,'BLOCKED');
  assert.equal(reviewEconomicShareLedger({...ledger,openingEvidence:{...issuer,publicationDate:'2026-10-14'}}).state,'BLOCKED');
});
test('G006 only exact ASML XAMS primary EUR closing quote becomes unadmitted candidate',()=>{
  const r=reviewHistoricalXamsClose(quote);
  assert.equal(r.state,'CANDIDATE_FOR_INDEPENDENT_REVIEW');
  assert.equal(r.certified,false);
  if(r.state==='CANDIDATE_FOR_INDEPENDENT_REVIEW') {
    assert.deepEqual(r.value,{close:100,currency:'EUR',asOf:'2026-09-21',securityId:quote.securityId});
  }
});
test('G006 rejects other exchanges, dates, adjusted prices and wrong issuer evidence',()=>{
  assert.equal(reviewHistoricalXamsClose(undefined).state,'BLOCKED');
  for(const changes of [
    {exchange:'NASDAQ'},{exchange:'BGEM'},{currency:'USD'},
    {isin:'USN070592100'},{tradingDate:'2026-09-22'},
    {valuationDate:'2026-09-22'},{quoteUnit:'EUR_PER_100_SHARES'},
    {close:-1},{close:NaN},{priceBasis:'SPLIT_ADJUSTED_CLOSE'},
    {corporateActionsChecked:false},{evidence:issuer},
    {evidence:{...exchange,sourceUrl:'https://live.euronext.com.evil.com'}}]) {
    assert.equal(reviewHistoricalXamsClose({...quote,...changes}).state,'BLOCKED');
  }
});
