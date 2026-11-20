import {readFileSync} from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";
import {projectOperationalValuation,OperationalValuationInputError,upsideToFairValue} from "../lib/orotitan-equity/v3/operational-valuation-view";
const s=()=>({
  core_fair_value_eur:100,combined_fair_value_eur:94,core_irr:.05,combined_irr:.04,
  core_wacc:.08,ptc_acquisition_npv_b:-.9,core_terminal_pv_fraction_of_ev:.75,
  price_ladder_combined:{"0.1":80,"0.125":72,"0.15":65},
});
const fixture=()=>({
  identity:{type:"NON_AUTHORITATIVE_OPERATIONAL_ANALYST_MODEL",
    run_id:"00000000-0000-4000-8000-000000000001",
    data_cutoff:"2026-10-30",valuation_date:"2026-10-30",
    calculation_date:"2026-11-19",valuation_lock:"NOT_CREATED",
    registry_write:false,valuation_admission_under_v31:"NO",
    certified:true,score_permission:"ALLOWED"},
  observations:{market_close_eur:110},
  analyst_assumptions:{economic_external_shares_proxy_m:500,illustrative_oct_net_debt_b:6},
  scenarios:{BEAR:s(),BASE:s(),BULL:s()},
  sensitivity_base_wacc_terminal:[{wacc:.08,terminal_growth:.025,combined_fair_value_eur:94}],
});
const rejects=(input:unknown,code:string)=>assert.throws(
  ()=>projectOperationalValuation(input),
  (err:unknown)=>err instanceof OperationalValuationInputError&&err.code===code);
test("renders generic three-scenario model, never grants certification",()=>{
  const x=projectOperationalValuation(fixture());
  assert.equal(x.scenarios.BASE.combinedFairValue,94);
  assert.equal(x.certified,false);assert.equal(x.valuationLockCreated,false);
  assert.equal(x.waccSensitivity.length,1);
});
test("rejects spurious authority flags",()=>{
  const x=fixture();x.identity.valuation_lock="CREATED";rejects(x,"AUTHORITY_CONFLICT");
});
test("rejects mismatched valuation cutoff",()=>{
  const x=fixture();x.identity.valuation_date="2026-10-29";rejects(x,"CUTOFF_MISMATCH");
});
test("rejects impossible calendar dates",()=>{
  const x=fixture();x.identity.valuation_date="2026-02-30";rejects(x,"VALUATION_DATE_INVALID");
});
test("rejects invalid run identities",()=>{
  const x=fixture();x.identity.run_id="not-a-uuid";rejects(x,"RUN_ID_INVALID");
});
test("rejects missing and invalid scenarios",()=>{
  const x=fixture();delete(x.scenarios as Partial<typeof x.scenarios>).BULL;
  rejects(x,"SCENARIO_MISSING_BULL");
  const y=fixture();y.scenarios.BASE.combined_fair_value_eur=Infinity;
  rejects(y,"COMBINED_VALUE_INVALID_BASE");
});
test("rejects invalid share proxy and negative reference price",()=>{
  const x=fixture();x.analyst_assumptions.economic_external_shares_proxy_m=0;
  rejects(x,"SHARES_INVALID");
  const y=fixture();y.observations.market_close_eur=-1;
  rejects(y,"REFERENCE_PRICE_INVALID");
});
test("rejects sensitivity overflow",()=>{
  const x=fixture();x.sensitivity_base_wacc_terminal[0].combined_fair_value_eur=NaN;
  rejects(x,"SENSITIVITY_VALUE_INVALID");
});
test("allows hypothetical purchase-price upside but not invented IRR",()=>{
  assert.ok(Math.abs((upsideToFairValue(94,80)??0)-.175)<1e-12);
  assert.equal(upsideToFairValue(94,0),null);
  assert.equal(upsideToFairValue(Infinity,1),null);
});

test("client file chooser guards against stale asynchronous read completion",()=>{
  const source=readFileSync(new URL("../app/admin/orotitan/valuation/operational-valuation-client.tsx",import.meta.url),"utf8");
  assert.match(source,/const generation=\+\+selectionGeneration\.current/);
  assert.match(source,/const fileContents=await file\.text\(\);\s*if\(generation!==selectionGeneration\.current\)return;/);
  assert.match(source,/\}catch\(err\)\{\s*if\(generation!==selectionGeneration\.current\)return;/);
});
