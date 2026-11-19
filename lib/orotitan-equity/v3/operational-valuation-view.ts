/**
 * Read-only projection of analyst scenario exports.
 * Browser-only calculations; no Registry access, certification or persistence.
 */
export type ScenarioName="BEAR"|"BASE"|"BULL";
export const SCENARIOS:readonly ScenarioName[]=["BEAR","BASE","BULL"];
export type ScenarioView={
  coreFairValue:number; combinedFairValue:number; coreIrr:number; combinedIrr:number;
  wacc:number; ptcNpvB:number; terminalWeight:number;
  ladder:{ten:number;twelveFive:number;fifteen:number};
};
export type OperationalView={
  type:"ANALYST_SCENARIO_ONLY";certified:false;valuationLockCreated:false;
  runId:string;cutoff:string;referencePrice:number;calculationDate:string;
  externalSharesProxyM:number;netDebtProxyB:number;
  scenarios:Record<ScenarioName,ScenarioView>;
  waccSensitivity:Array<{wacc:number;terminalGrowth:number;combinedFairValue:number}>;
};
export class OperationalValuationInputError extends Error{
  readonly code:string;
  constructor(code:string){super(code);this.code=code;this.name="OperationalValuationInputError";}
}
function fail(code:string):never{throw new OperationalValuationInputError(code);}
function record(v:unknown,c:string):Record<string,unknown>{
  if(!v||typeof v!=="object"||Array.isArray(v))return fail(c);
  return v as Record<string,unknown>;
}
function num(v:unknown,c:string,min=-Infinity,max=Infinity):number{
  if(typeof v!=="number"||!Number.isFinite(v)||v<min||v>max)return fail(c);
  return v;
}
function day(v:unknown,c:string):string{
  if(typeof v!=="string"||!/^\d{4}-\d{2}-\d{2}$/.test(v))return fail(c);
  const d=new Date(v+"T00:00:00Z");
  if(!Number.isFinite(d.getTime())||d.toISOString().slice(0,10)!==v)return fail(c);
  return v;
}
/** Never trusts a caller-supplied flag to grant authority. */
export function projectOperationalValuation(payload:unknown):OperationalView{
  const root=record(payload,"JSON_OBJECT_REQUIRED");
  const id=record(root.identity,"IDENTITY_MISSING");
  if(id.type!=="NON_AUTHORITATIVE_OPERATIONAL_ANALYST_MODEL")return fail("NOT_OPERATIONAL_ANALYST_MODEL");
  if(id.valuation_lock!=="NOT_CREATED"||id.registry_write!==false||
     id.valuation_admission_under_v31!=="NO")return fail("AUTHORITY_CONFLICT");
  const cutoff=day(id.data_cutoff,"CUTOFF_INVALID");
  if(day(id.valuation_date,"VALUATION_DATE_INVALID")!==cutoff)return fail("CUTOFF_MISMATCH");
  const calculationDate=day(id.calculation_date,"CALCULATION_DATE_INVALID");
  if(typeof id.run_id!=="string"||
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id.run_id))return fail("RUN_ID_INVALID");
  const obs=record(root.observations,"OBSERVATIONS_MISSING");
  const assumptions=record(root.analyst_assumptions,"ASSUMPTIONS_MISSING");
  const source=record(root.scenarios,"SCENARIOS_MISSING");
  const scenarios={} as Record<ScenarioName,ScenarioView>;
  for(const name of SCENARIOS){
    const s=record(source[name],"SCENARIO_MISSING_"+name);
    const ladder=record(s.price_ladder_combined,"LADDER_MISSING_"+name);
    scenarios[name]={
      coreFairValue:num(s.core_fair_value_eur,"CORE_VALUE_INVALID_"+name,0),
      combinedFairValue:num(s.combined_fair_value_eur,"COMBINED_VALUE_INVALID_"+name,0),
      coreIrr:num(s.core_irr,"CORE_IRR_INVALID_"+name,-1,100),
      combinedIrr:num(s.combined_irr,"IRR_INVALID_"+name,-1,100),
      wacc:num(s.core_wacc,"WACC_INVALID_"+name,0,1),
      ptcNpvB:num(s.ptc_acquisition_npv_b,"PTC_NPV_INVALID_"+name),
      terminalWeight:num(s.core_terminal_pv_fraction_of_ev,"TERMINAL_INVALID_"+name,0,1),
      ladder:{
        ten:num(ladder["0.1"],"LADDER_10_INVALID_"+name,0),
        twelveFive:num(ladder["0.125"],"LADDER_12_5_INVALID_"+name,0),
        fifteen:num(ladder["0.15"],"LADDER_15_INVALID_"+name,0),
      },
    };
  }
  const waccSensitivity:OperationalView["waccSensitivity"]=[];
  if(root.sensitivity_base_wacc_terminal!==undefined){
    if(!Array.isArray(root.sensitivity_base_wacc_terminal)||
       root.sensitivity_base_wacc_terminal.length>200)return fail("SENSITIVITY_INVALID");
    for(const row of root.sensitivity_base_wacc_terminal){
      const r=record(row,"SENSITIVITY_ROW_INVALID");
      waccSensitivity.push({
        wacc:num(r.wacc,"SENSITIVITY_WACC_INVALID",0,1),
        terminalGrowth:num(r.terminal_growth,"SENSITIVITY_G_INVALID",-1,1),
        combinedFairValue:num(r.combined_fair_value_eur,"SENSITIVITY_VALUE_INVALID",0),
      });
    }
  }
  return {
    type:"ANALYST_SCENARIO_ONLY",certified:false,valuationLockCreated:false,
    runId:id.run_id,cutoff,calculationDate,
    referencePrice:num(obs.market_close_eur,"REFERENCE_PRICE_INVALID",.001,100000),
    externalSharesProxyM:num(assumptions.economic_external_shares_proxy_m,"SHARES_INVALID",.001,1e10),
    netDebtProxyB:num(assumptions.illustrative_oct_net_debt_b,"DEBT_INVALID",-1e7,1e7),
    scenarios,waccSensitivity,
  };
}
export function upsideToFairValue(fair:number,price:number):number|null{
  if(![fair,price].every(Number.isFinite)||fair<0||price<=0)return null;
  return fair/price-1;
}
