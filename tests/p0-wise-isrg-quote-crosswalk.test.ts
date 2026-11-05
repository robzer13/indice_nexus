import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { resolveCanonicalQuoteCompanyId, type LegacyCompanyQuoteLink } from '../lib/domain/canonical-quote-identity';
import { getFreshness } from '../lib/domain/freshness';
import { assertCanonicalMarketReference, isStoredQuoteCurrencyCompatible, requireUsListingMatch } from '../lib/market/price-quality';

const wise: LegacyCompanyQuoteLink = {
  legacy_company_id:'185329c2-c577-4461-9954-6ecfd75c7483',
  issuer_id:'185329c2-c577-4461-9954-6ecfd75c7483',
  security_id:'cc46c84b-a073-448a-8109-fea143605ac0',
  dossier_id:'b17d1bd3-6a7b-4b89-bbe3-b7022a36b651',
};
const isrg: LegacyCompanyQuoteLink = {
  legacy_company_id:'9748c01b-ec84-48dd-b5a5-3dafd0750d89',
  issuer_id:'9748c01b-ec84-48dd-b5a5-3dafd0750d89',
  security_id:'c459062e-baa6-43ec-b75e-86916b805e90',
  dossier_id:'2796e551-4241-4ae8-b6e1-135f9b653767',
};

test('P0: Wise and ISRG remain unlinked unless both exact audited quadruplets are added',()=>{
  for(const x of [wise,isrg]){
    assert.equal(resolveCanonicalQuoteCompanyId(x,[]),null);
    assert.equal(resolveCanonicalQuoteCompanyId(x,[wise,isrg]),x.legacy_company_id);
    assert.equal(resolveCanonicalQuoteCompanyId({...x,security_id:'different security'},[wise,isrg]),null);
    assert.equal(resolveCanonicalQuoteCompanyId({...x,dossier_id:'different dossier'},[wise,isrg]),null);
  }
  assert.equal(resolveCanonicalQuoteCompanyId(wise,[wise,{...wise,legacy_company_id:isrg.legacy_company_id}]),null);
});
test('P0: audited US Nasdaq symbol aliases are venue-consistent; no foreign/ticker substitution',()=>{
  for (const [ticker,reference] of [['WSE','WSE:NASDAQ'],['ISRG','ISRG']] as const){
    assert.doesNotThrow(()=>assertCanonicalMarketReference({
      reference,ticker,exchange:'Nasdaq',currency:'USD',quoteUnit:'MAJOR',multiplier:1,previousPrice:null,
    }));
    assert.doesNotThrow(()=>requireUsListingMatch('Nasdaq','NMS'));
    assert.equal(isStoredQuoteCurrencyCompatible({currency:'USD'},'USD','MAJOR'),true);
    assert.equal(isStoredQuoteCurrencyCompatible({currency:'GBP'},'USD','MAJOR'),false);
  }
  assert.throws(()=>assertCanonicalMarketReference({
    reference:'WISE:LSE',ticker:'WSE',exchange:'Nasdaq',currency:'USD',quoteUnit:'MAJOR',multiplier:1,previousPrice:null,
  }),/QUOTE_REFERENCE_SECURITY_MISMATCH/);
});
test('P0: Oct 21 verified close would not be stale on Oct 22',()=>{
  const now=new Date('2026-10-22T14:00:00Z');
  assert.equal(getFreshness('2026-10-21T20:00:00Z',now).stale,false);
  assert.equal(getFreshness('2026-09-18T00:00:00Z',now).stale,true);
});
test('P0: manual-only SQL proposal is pinned, guarded and does not rewrite historical evidence',()=>{
  const sql=readFileSync('docs/orotitan-equity/proposals/P0_WISE_ISRG_CROSSWALK_STAGED_GO_REQUIRED.sql','utf8');
  assert.match(sql,/STAGED PROPOSAL ONLY/);
  assert.match(sql,/Requires a separate, explicit production data-change GO/);
  assert.match(sql,/P0_QUOTE_FOUR_PART_IDENTITY_NOT_PROVEN/);
  assert.match(sql,/P0_QUOTE_CROSSWALK_ALREADY_ASSIGNED_REVIEW/);
  assert.match(sql,/P0_QUOTE_RECENT_FEED_IDENTITY_NOT_PROVEN/);
  assert.match(sql,/P0_QUOTE_CROSSWALK_POSTCONDITION_FAILED/);
  assert.match(sql,/GET DIAGNOSTICS rows_updated = ROW_COUNT/);
  assert.match(sql,/rows_updated <> 1/);
  assert.match(sql,/insert into public\.legacy_company_identity_map/i);
  assert.match(sql,/set market_data_symbol='WSE:NASDAQ'/i);
  assert.match(sql,/snap\.security_id is distinct from p\.security_id/i);
  for(const x of [wise,isrg]){
    for(const id of Object.values(x))assert.ok(sql.includes(id));
  }
  for(const relation of ['market_prices','companies','issuers','research_snapshots','research_dossiers','orotitan_runs','orotitan_run_stages']){
    assert.doesNotMatch(sql,new RegExp('(?:insert\\s+into|update|delete\\s+from|truncate|alter\\s+table)\\s+public\\.'+relation+'\\b','i'));
  }
  assert.ok(sql.indexOf('P0_QUOTE_FOUR_PART_IDENTITY_NOT_PROVEN') <
    sql.indexOf('INSERT INTO public.legacy_company_identity_map'));
});
