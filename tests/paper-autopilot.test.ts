import assert from 'node:assert/strict';
import test from 'node:test';
import { planPaperAutoFill, trustedExchangeQuote, admissibleAutomaticRepricing } from '../lib/domain/paper-autopilot';
import { cashDividendEur, computeTimeWeightedReturn, maximumDrawdown, relativePerformance, translateToEur } from '../lib/domain/paper-performance';

const policy = { enabled: true, initialCapitalEur: 100000, availableCashEur: 100000, maxInitialCapitalWeight: .05, simulatedFeeEur: 1 };
const quoted = { source:'YAHOO_FINANCE', as_of:'2026-10-09T16:30:00Z', created_at:'2026-10-09T22:15:00Z', raw:{regularMarketTime:'2026-10-09T16:30:00Z'} };
test('automatic actions require opt-in and preserve the 5% initial capital budget', () => {
  assert.deepEqual(planPaperAutoFill('BUY_CANDIDATE', 120, 0, policy), { permitted:true, quantity:41 });
  assert.equal(planPaperAutoFill('BUY_CANDIDATE', 120, 0, { ...policy, enabled:false }).permitted,false);
  assert.equal(planPaperAutoFill('BUY_CANDIDATE', 120, 3, policy).permitted,false);
  assert.equal(planPaperAutoFill('BUY_CANDIDATE', 6000, 0, policy).permitted,false);
  assert.equal(planPaperAutoFill('BUY_CANDIDATE', 120, 0, { ...policy, maxInitialCapitalWeight:.5 }).permitted,false);
});
test('sell exits use all held whole shares and avoid unsupported fractional sells', () => {
  assert.deepEqual(planPaperAutoFill('SELL_CANDIDATE',12,10,policy),{ permitted:true,quantity:10 });
  assert.equal(planPaperAutoFill('SELL_CANDIDATE',12,0,policy).permitted,false);
  assert.equal(planPaperAutoFill('SELL_CANDIDATE',12,.5,policy).permitted,false);
  assert.equal(planPaperAutoFill('HOLD',12,10,policy).permitted,false);
});
test('only a provenance-matched real Yahoo exchange timestamp enters autonomous execution', () => {
  assert.equal(trustedExchangeQuote(quoted,'2026-10-11T09:00:00Z'),true);
  assert.equal(trustedExchangeQuote({...quoted,source:'TWELVE_DATA'},'2026-10-11T09:00:00Z'),false);
  assert.equal(trustedExchangeQuote({...quoted,raw:{}},'2026-10-11T09:00:00Z'),false);
  assert.equal(trustedExchangeQuote({...quoted,raw:{regularMarketTime:'2026-10-10T16:30:00Z'}},'2026-10-11T09:00:00Z'),false);
  assert.equal(trustedExchangeQuote(quoted,'2026-10-20T09:00:00Z'),false);
});
test('time-weighted return and maximum drawdown stay auditable', () => {
  const nav = [{date:'2026-01-01',navEur:100,externalNetContributionEur:0},{date:'2026-01-02',navEur:111,externalNetContributionEur:10}];
  assert.equal(computeTimeWeightedReturn(nav),.010000000000000009);
  assert.equal(computeTimeWeightedReturn([...nav,nav[1]]),null);
  assert.equal(maximumDrawdown([100,130,104,140]),.2);
  assert.equal(relativePerformance(.12,.09),.03);
});
test('FX and cash dividends fail closed on absent or future provenance', () => {
  const r={currency:'USD',eurPerUnit:.9,observedAt:'2026-10-09T16:30:00Z',source:'ECB'};
  assert.equal(translateToEur(100,r,'2026-10-09T17:00:00Z'),90);
  assert.equal(translateToEur(100,r,'2026-10-09T15:00:00Z'),null);
  assert.equal(translateToEur(100,{...r,source:''},'2026-10-09T17:00:00Z'),null);
  assert.equal(cashDividendEur(100,2,.15,.9),153);
  assert.equal(cashDividendEur(100,2,1.1,.9),null);
});

test('autopilot rejects expired signals and major gaps rather than chasing a quote',()=>{
  const start='2026-10-05T22:45:00Z',end='2026-10-06T22:45:00Z';
  assert.equal(admissibleAutomaticRepricing('BUY_CANDIDATE',100,104,start,end),true);
  assert.equal(admissibleAutomaticRepricing('BUY_CANDIDATE',100,120,start,end),false);
  assert.equal(admissibleAutomaticRepricing('BUY_CANDIDATE',100,85,start,end),false);
  assert.equal(admissibleAutomaticRepricing('SELL_CANDIDATE',12,11.5,start,end),true);
  assert.equal(admissibleAutomaticRepricing('SELL_CANDIDATE',12,8,start,end),false);
  assert.equal(admissibleAutomaticRepricing('BUY_CANDIDATE',100,100,start,'2026-10-20T22:45:00Z'),false);
});
