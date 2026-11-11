import assert from "node:assert/strict";
import test from "node:test";
import {
  evaluateValuationExploratoryPreview,
  type PreviewRequest,
} from "../lib/orotitan-equity/v3/valuation-exploratory-preview";

const cutoff = "2026-11-11";
function valid(): PreviewRequest {
  return {
    mode:"EXPLORATORY_PREVIEW", analysisCutoff:cutoff, valuationDate:cutoff,
    upstreamFundamentalsCutoff:"2026-10-30", upstreamFreshness:"HISTORICAL_DISCLOSED",
    upstreamRef:{runId:"db00ce77-04eb-492b-a76b-164ecbe9ccfe",manifestId:"75000000-0000-4000-8000-00000000000b",
      lockSha256:"6eda7f780f87c309276cd183384e1864e3f8e31e4402a622902b7330d6b53961"},
    sourceRefs:[{sourceId:"DEMO_PUBLIC", publicationDate:"2026-11-09",economicEffectiveDate:"2026-11-06",locator:"https://example.org/demo",access:"PUBLIC"}],
    cashFlowBasis:"FCFF", discountRateBasis:"WACC",historicalOutputCalibration:false,
    forecastCases:[{id:"DEMO_BASE",assumptionNote:"Synthetic cash flow scenario; no Schneider estimate",sourceIds:["DEMO_PUBLIC"],discountRate:0.1,discountRateNote:"Synthetic WACC",
      firstFiscalPeriodStartDate:"2026-01-01",fiscalCalendarResolved:true,fiscalYearMismatchResolved:true,
      flows:[{periodStartDate:cutoff,paymentDate:"2027-11-11",amount:110,construction:"POST_ORIGIN_ONLY"}],
      terminal:{date:"2027-11-11",growthRate:0.02,nextPeriodCashFlow:88,stableEconomicsResolved:true,assumptionNote:"Synthetic stable economics"}}]
  };
}
const evaluate=(mutate:(x:PreviewRequest)=>void=()=>{})=>{const v=valid();mutate(v);return evaluateValuationExploratoryPreview(v)};
const denied=(code:string,mutate:(x:PreviewRequest)=>void)=>{
  const out=evaluate(mutate);assert.equal(out.admissibleAsPreview,false,code);
  assert.ok(out.refusalReasons.includes(code),JSON.stringify(out.refusalReasons));
  assert.equal(out.valuationLockPermitted,false); assert.equal(out.scoringPermitted,false);
};

test("T01: synthetic FCFF computes enterprise DCF without any share count or equity bridge",()=>{
  const x=evaluate();assert.equal(x.admissibleAsPreview,true);
  assert.deepEqual(x.outputTypes,["ENTERPRISE_DCF_SCENARIO"]);
  assert.equal(x.cases.length,1);
  // PV = 110 / 1.1 + (88 / (0.1 - 0.02)) / 1.1 = 100 + 1000
  assert.ok(Math.abs(x.cases[0].enterpriseValueScenario! - 1100)<1e-9);
  assert.equal(x.cases[0].equityValueScenario,null);
  assert.ok(x.warnings.includes("HISTORICAL_FUNDAMENTALS_NOT_REVALIDATED_FOR_NEW_CUTOFF"));
});
test("T02: optional hypothetical bridge and share scenarios are never canonical",()=>{
  const x=evaluate(v=>{v.bridgeScenario={netEquityBridgeAdjustment:-100,assumptionNote:"Synthetic adjustment",economicDate:cutoff};
    v.shareScenarios=[{id:"LOW",count:500,assumptionNote:"Synthetic shares",economicDate:cutoff},
       {id:"HIGH",count:1000,assumptionNote:"Synthetic shares",economicDate:cutoff}];});
  assert.deepEqual(x.outputTypes,["ENTERPRISE_DCF_SCENARIO","EQUITY_VALUE_SCENARIO","PER_SHARE_SCENARIO"]);
  assert.ok(Math.abs(x.cases[0].equityValueScenario! - 1000)<1e-9);
  assert.ok(Math.abs(x.cases[0].perShareScenarios[0].valuePerShare - 2)<1e-12);
  assert.ok(Math.abs(x.cases[0].perShareScenarios[1].valuePerShare - 1)<1e-12);
});
test("T03: price comparison is not MOS, expected return or a score",()=>{
  const x=evaluate(v=>{v.bridgeScenario={netEquityBridgeAdjustment:0,assumptionNote:"Synthetic",economicDate:cutoff};
    v.shareScenarios=[{id:"S",count:1000,assumptionNote:"Synthetic",economicDate:cutoff}];
    v.priceReference={date:cutoff,price:1,sourceId:"DEMO_PUBLIC",freshnessResolved:true};});
  assert.ok(x.outputTypes.includes("PRICE_COMPARISON_SCENARIO"));
  assert.ok(Math.abs(x.cases[0].priceComparisons[0].hypotheticalPremiumPct - 10)<1e-10);
  for(const k of ["MOS","OQS","OVS","InvestmentScore","VALUTION_LOCK","readyForCertification"]) {
    if(k!=="readyForCertification")assert.equal(k in x,false,k);
    else assert.equal(x.readyForCertification,false);
  }
});
test("T04: all responses permanently refuse canonical locks, scoring, writes and orders",()=>{
  for(const x of [evaluate(),evaluate(v=>{v.mode="CANONICAL" as PreviewRequest["mode"]}),evaluate(v=>{v.analysisCutoff="2026-11-12";})]){
    assert.equal(x.authority,"NON_AUTHORITATIVE");assert.equal(x.contractStatus,"G2_PROPOSED_UNAPPROVED");
    assert.equal(x.canonicalValuationAdmission,"NOT_EVALUATED_IN_PREVIEW");
    for(const k of ["valuationLockPermitted","readyForCertification","scoringPermitted","publicationPermitted","orderPermitted","registryMutationPermitted"] as const)assert.equal(x[k],false);
  }
});
test("T05: canonical mode cannot traverse preview",()=>denied("UNSUPPORTED_MODE",v=>{v.mode="CANONICAL" as PreviewRequest["mode"]}));
test("T06: post-cutoff public evidence is rejected",()=>denied("SOURCE_PUBLISHED_AFTER_CUTOFF",v=>{v.sourceRefs=[{...v.sourceRefs[0],publicationDate:"2026-11-12"}]}));
test("T07: future economic fact is rejected",()=>denied("SOURCE_EFFECTIVE_AFTER_CUTOFF",v=>{v.sourceRefs=[{...v.sourceRefs[0],economicEffectiveDate:"2026-11-12"}]}));
test("T08: private sources forbidden",()=>denied("SOURCE_PROVENANCE_INCOMPLETE",v=>{v.sourceRefs=[{...v.sourceRefs[0],access:"PRIVATE" as "PUBLIC"}]}));
test("T09: absence of public evidence refused",()=>denied("NO_PUBLIC_EVIDENCE",v=>{v.sourceRefs=[]}));
test("T10: duplicate source identity refused",()=>denied("DUPLICATE_SOURCE_ID",v=>{v.sourceRefs=[v.sourceRefs[0],v.sourceRefs[0]]}));
test("T11: cutoff must equal valuation date",()=>denied("VALUATION_DATE_DATA_CUTOFF_MISMATCH",v=>{v.valuationDate="2026-11-10"}));
test("T12: invalid calendar date refused",()=>denied("INVALID_DATE",v=>{v.analysisCutoff="2026-11-31"}));
test("T13: upstream future cutoff forbidden",()=>denied("UPSTREAM_FUNDAMENTALS_AFTER_CUTOFF",v=>{v.upstreamFundamentalsCutoff="2026-12-01"}));
test("T14: historically stale upstream cannot be described as fresh",()=>denied("HISTORICAL_FUNDAMENTALS_MUST_BE_DISCLOSED",v=>{v.upstreamFreshness="FRESH_VERIFIED"}));
test("T15: malformed upstream immutable reference refused",()=>denied("UPSTREAM_REF_INVALID",v=>{v.upstreamRef.lockSha256="bad"}));
test("T16: FCFF requires WACC",()=>denied("FCFF_REQUIRES_WACC",v=>{v.discountRateBasis="COST_OF_EQUITY"}));
test("T17: FCFE requires equity-cost discount rate",()=>denied("EQUITY_FLOWS_REQUIRE_COST_OF_EQUITY",v=>{v.cashFlowBasis="FCFE"}));
test("T18: unsupported Owner Earnings always rejected",()=>denied("OWNER_EARNINGS_UNSUPPORTED",v=>{v.cashFlowBasis="OWNER_EARNINGS";v.discountRateBasis="COST_OF_EQUITY"}));
test("T19: supported FCFE produces equity scenario without ordinary bridge",()=>{
 const x=evaluate(v=>{v.cashFlowBasis="FCFE";v.discountRateBasis="COST_OF_EQUITY"});
 assert.deepEqual(x.outputTypes,["EQUITY_VALUE_SCENARIO"]);assert.equal(x.cases[0].enterpriseValueScenario,null);
});
test("T20: forbidden equity DCF bridge denied",()=>denied("EQUITY_DCF_EV_BRIDGE_FORBIDDEN",v=>{v.cashFlowBasis="FCFE";v.discountRateBasis="COST_OF_EQUITY";v.bridgeScenario={netEquityBridgeAdjustment:0,assumptionNote:"Demo",economicDate:cutoff}}));
test("T21: historical calibration forbidden",()=>denied("HISTORICAL_CALIBRATION_FORBIDDEN",v=>{v.historicalOutputCalibration=true as false}));
test("T22: empty forecast fails",()=>denied("FORECAST_MISSING",v=>{v.forecastCases=[]}));
test("T23: zero or invalid rate fails",()=>denied("INVALID_DISCOUNT_RATE",v=>{v.forecastCases[0].discountRate=-1}));
test("T24: terminal rate less than discount rate",()=>denied("TERMINAL_G_NOT_LESS_THAN_R",v=>{v.forecastCases[0].terminal.growthRate=0.11}));
test("T25: nonfinite inputs fail",()=>denied("NONFINITE_CASH_FLOW",v=>{v.forecastCases[0].flows[0].amount=Infinity}));
test("T26: forecast period cannot include pre-origin cash by auto-proration",()=>denied("FORECAST_DATE_CHAIN_OR_STUB_INVALID",v=>{v.forecastCases[0].flows[0].periodStartDate="2026-01-01"}));
test("T27: payment must be after cutoff",()=>denied("FORECAST_DATE_CHAIN_OR_STUB_INVALID",v=>{v.forecastCases[0].flows[0].paymentDate=cutoff;v.forecastCases[0].terminal.date=cutoff}));
test("T28: terminal must occur at last forecast payment",()=>denied("TERMINAL_DATE_MISMATCH",v=>{v.forecastCases[0].terminal.date="2028-11-11"}));
test("T29: terminal assumptions must be supportable",()=>denied("TERMINAL_ASSUMPTIONS_UNSUPPORTED",v=>{v.forecastCases[0].terminal.stableEconomicsResolved=false}));
test("T30: fiscal calendar must be resolved",()=>denied("FISCAL_TIMING_UNRESOLVED",v=>{v.forecastCases[0].fiscalCalendarResolved=false}));
test("T31: per-share cannot use invalid share assumptions",()=>denied("SHARE_SCENARIO_INVALID",v=>{v.shareScenarios=[{id:"X",count:0,assumptionNote:"Demo",economicDate:cutoff}]}));
test("T32: after-cutoff price is ignored, not injected",()=>{
 const x=evaluate(v=>{v.bridgeScenario={netEquityBridgeAdjustment:0,assumptionNote:"Demo",economicDate:cutoff};
 v.shareScenarios=[{id:"X",count:100,assumptionNote:"Demo",economicDate:cutoff}];
 v.priceReference={date:"2026-11-12",price:1,sourceId:"DEMO_PUBLIC",freshnessResolved:true};});
 assert.equal(x.admissibleAsPreview,true);assert.ok(!x.outputTypes.includes("PRICE_COMPARISON_SCENARIO"));
});
test("T33: multiple synthetic DCF cases produce separate sensitivity outputs",()=>{
 const x=evaluate(v=>{v.forecastCases=[v.forecastCases[0],{...v.forecastCases[0],id:"DEMO_RATE_HIGH",discountRate:0.12,discountRateNote:"Synthetic high WACC"}];});
 assert.equal(x.cases.length,2);assert.ok(x.cases[0].enterpriseValueScenario!>x.cases[1].enterpriseValueScenario!);
});
test("T34: same source evidence with 2026-10-30 historical cutoff remains isolated (no post-cutoff leakage)",()=>{
 denied("SOURCE_PUBLISHED_AFTER_CUTOFF",v=>{v.analysisCutoff="2026-10-30";v.valuationDate="2026-10-30";v.sourceRefs=[{...v.sourceRefs[0],publicationDate:"2026-11-02"}];});
});
test("T35: missing bridge only removes equity/per-share, never fabricates them",()=>{
 const x=evaluate(v=>{v.shareScenarios=[{id:"S",count:100,assumptionNote:"Demo",economicDate:cutoff}]});
 assert.deepEqual(x.outputTypes,["ENTERPRISE_DCF_SCENARIO"]);
 assert.ok(x.warnings.includes("PER_SHARE_OMITTED_EQUITY_BRIDGE_MISSING"));
});

test("T36: forecast evidence lineage must refer to a public source",()=>denied("FORECAST_SOURCE_LINEAGE_MISSING",v=>{v.forecastCases[0].sourceIds=["UNSUPPORTED"]; }));
test("T37: the calculation record retains source, assumptions, date method and input cases",()=>{
 const x=evaluate();assert.equal(x.methodVersion,"G2_EXPLORATORY_DRAFT_V0_1");
 assert.deepEqual(x.cases[0].sourceIds,["DEMO_PUBLIC"]);
 assert.equal(x.cases[0].method,"ACT/365F_END_PERIOD_NO_AUTO_PRORATION");
 assert.equal(x.cases[0].discountRate,0.1);
 assert.equal(x.evidenceAsOfAndPublicationDates[0].publicationDate,"2026-11-09");
 assert.equal(x.evidenceAsOfAndPublicationDates[0].economicEffectiveDate,"2026-11-06");
});
