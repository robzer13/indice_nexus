/**
 * Independent, NON-AUTHORITATIVE validation gates for ASML-G001 and G006.
 *
 * These consume verified primary-source records in a future workflow. They do
 * not retrieve prices, infer shares, admit documents or change Registry state.
 *
 * Economic shares = issued shares - treasury shares, with buybacks counted on
 * the TREASURY ledger once (not both as cancellations and treasury purchases).
 */
export const ASML_RECOVERY_CUTOFF='2026-09-21';
const SHA=/^[a-f0-9]{64}$/;
const DAY=/^\d{4}-\d{2}-\d{2}$/;
function validDay(date:string):boolean {
  if(!DAY.test(date))return false;
  const d=new Date(date+'T00:00:00Z');
  return !Number.isNaN(d.valueOf())&&d.toISOString().slice(0,10)===date;
}
export type DatedPrimaryEvidence=Readonly<{
  sourceId:string; publicationDate:string; sha256:string; sourceUrl:string;
}>;
export type ShareMovement=Readonly<{
  eventId:string; effectiveDate:string; shares:number;
  kind:'ISSUANCE'|'CANCELLATION_TREASURY'|'CANCELLATION_NON_TREASURY'|
       'TREASURY_ACQUISITION_SETTLED'|'TREASURY_DISPOSAL_SETTLED';
  evidence:DatedPrimaryEvidence;
}>;
export type ShareLedger=Readonly<{
  valuationDate:string; openingDate:string;
  issuedAtOpening:number; treasuryAtOpening:number;
  openingEvidence:DatedPrimaryEvidence;
  movements:readonly ShareMovement[];
}>;
export type HistoricalClosingQuote=Readonly<{
  valuationDate:string; tradingDate:string; securityId:string; exchange:string;
  currency:string; quoteUnit:string; close:number; priceBasis:string;
  corporateActionsChecked:boolean; evidence:DatedPrimaryEvidence;
}>;
type Blocked=Readonly<{state:'BLOCKED';reason:string;certified:false}>;
type Candidate<T>=Readonly<{state:'CANDIDATE_FOR_INDEPENDENT_REVIEW';value:T;certified:false}>;
function safeShares(x:number):boolean {
  return Number.isSafeInteger(x) && x>=0;
}
function evidenceOk(x:DatedPrimaryEvidence|undefined,cutoff:string):boolean {
  return !!x && typeof x.sourceId==='string' && x.sourceId.length>5 &&
    typeof x.sourceUrl==='string' && x.sourceUrl.startsWith('https://') &&
    SHA.test(x.sha256) && validDay(x.publicationDate) &&
    x.publicationDate<=cutoff;
}
const fail=(reason:string):Blocked=>({state:'BLOCKED',reason,certified:false});
export function reviewEconomicShareLedger(
  ledger:ShareLedger|undefined,
):Blocked|Candidate<{issued:number;treasury:number;economicShares:number;asOf:string}> {
  if(!ledger)return fail('G001: source-dated opening issued/treasury shares and corporate-action ledger missing');
  const cutoff=ASML_RECOVERY_CUTOFF;
  if(ledger.valuationDate!==cutoff || !validDay(ledger.openingDate) ||
     ledger.openingDate>cutoff || !evidenceOk(ledger.openingEvidence,cutoff) ||
     !safeShares(ledger.issuedAtOpening) || !safeShares(ledger.treasuryAtOpening) ||
     ledger.treasuryAtOpening>ledger.issuedAtOpening ||
     !Array.isArray(ledger.movements) || ledger.movements.length>1000) {
    return fail('G001: opening source, issued/treasury balance or date invalid');
  }
  let issued=ledger.issuedAtOpening;
  let treasury=ledger.treasuryAtOpening;
  const seen=new Set<string>();
  const movements=[...ledger.movements].sort((a,b)=>a.effectiveDate.localeCompare(b.effectiveDate));
  for(const m of movements) {
    if(!m.eventId || seen.has(m.eventId) || !validDay(m.effectiveDate) ||
       m.effectiveDate<=ledger.openingDate || m.effectiveDate>cutoff ||
       !safeShares(m.shares) || m.shares===0 || !evidenceOk(m.evidence,cutoff) ||
       m.evidence.publicationDate<m.effectiveDate) {
      return fail('G001: untraceable, post-cutoff or duplicated share movement');
    }
    seen.add(m.eventId);
    switch(m.kind) {
      case 'ISSUANCE': issued+=m.shares; break;
      case 'CANCELLATION_TREASURY': issued-=m.shares;treasury-=m.shares;break;
      case 'CANCELLATION_NON_TREASURY': issued-=m.shares;break;
      case 'TREASURY_ACQUISITION_SETTLED': treasury+=m.shares;break;
      case 'TREASURY_DISPOSAL_SETTLED': treasury-=m.shares;break;
      default: return fail('G001: unknown share movement type');
    }
    if(!safeShares(issued)||!safeShares(treasury)||treasury>issued) {
      return fail('G001: share/treasury ledger does not reconcile');
    }
  }
  if(issued===0)return fail('G001: no issued shares');
  return {state:'CANDIDATE_FOR_INDEPENDENT_REVIEW',
    value:{issued,treasury,economicShares:issued-treasury,asOf:cutoff},certified:false};
}
export function reviewHistoricalXamsClose(
  quote:HistoricalClosingQuote|undefined,
):Blocked|Candidate<{close:number;currency:'EUR';asOf:string;securityId:string}> {
  if(!quote)return fail('G006: independently sourced XAMS 2026-09-21 closing price missing');
  const cutoff=ASML_RECOVERY_CUTOFF;
  if(quote.valuationDate!==cutoff || quote.tradingDate!==cutoff ||
     quote.exchange!=='XAMS' || quote.currency!=='EUR' ||
     quote.quoteUnit!=='EUR_PER_SHARE' || quote.priceBasis!=='UNADJUSTED_CLOSE' ||
     !quote.corporateActionsChecked || !quote.securityId ||
     !Number.isFinite(quote.close) || quote.close<=0 ||
     !evidenceOk(quote.evidence,cutoff)) {
    return fail('G006: date, identity, exchange, EUR unit, adjustment or source provenance invalid');
  }
  return {state:'CANDIDATE_FOR_INDEPENDENT_REVIEW',
    value:{close:quote.close,currency:'EUR',asOf:cutoff,
      securityId:quote.securityId},certified:false};
}
