import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { validateV2ResearchSnapshotWithExactCertifiedLiterals } from "../lib/orotitan-equity/v2/research-snapshot-schema-exact-literals";
import { validateV2ResearchSnapshotForPersistence } from "../lib/orotitan-equity/v2/research-snapshot-schema";
import { validateFicoStyleExactLiteralsFromSealedSources } from "../lib/orotitan-equity/v2/sealed-literal-source-gate";

const runId="00000000-0000-4000-8000-000000000001";
const dossierId="00000000-0000-4000-8000-000000000002";
const issuerId="00000000-0000-4000-8000-000000000003";
const securityId="00000000-0000-4000-8000-000000000004";
const cutoff="2026-09-15";
const gateMap=[
  ["CERTIFICATION_GATE","RESEARCH_NOT_FULLY_CERTIFIED"],["MOAT_ELITE","MOAT_NOT_ELITE"],
  ["RUNWAY_ELITE","RUNWAY_NOT_ELITE"],["RETURN_QUALITY_ELITE","RETURN_QUALITY_NOT_ELITE"],
  ["CASH_ECONOMICS_ELITE","CASH_ECONOMICS_NOT_ELITE"],["CAPITAL_ALLOCATION_ELITE","CAPITAL_ALLOCATION_NOT_ELITE"],
  ["MANAGEMENT_GOVERNANCE_ELITE","MANAGEMENT_GOVERNANCE_NOT_ELITE"],["RESILIENCE_ELITE","RESILIENCE_NOT_ELITE"],
  ["VALUATION_ELITE","VALUATION_NOT_ELITE"],["MATERIAL_WEAK_LINK_GATE","MATERIAL_WEAK_LINK_PRESENT"],
] as const;

function analyze(): Record<string, unknown> {
  return {
    snapshot_id:"00000000-0000-4000-8000-000000000007",
    report_id:"synthetic-analysis-report",issuer_id:issuerId,security_id:securityId,
    execution_mode:"ANALYZE",
    data_lock:{
      data_cutoff:cutoff,reference_price:100,reference_price_currency:"USD",
      reference_price_date:cutoff,calculation_date:cutoff,last_full_research_date:cutoff,score_date:cutoff,
    },
    versions:{
      report_version:"synthetic-report@1",method_version:"synthetic-method@1",
      calculation_version:"synthetic-calc@1",evidence_ledger_version:"synthetic-evidence@1",
      method_versions:{
        discovery_version:"d1",moat_version:"m1",runway_version:"r1",roic_version:"roic1",
        fcf_version:"fcf1",valuation_version:"v1",certification_version:"cert1",
      },
    },
    l2_research_fundamentals:{
      fundamental_states:{
        moat_evidence_state:"SUPPORTED",moat_trend:"STABLE",moat_durability:"MODERATE",
        runway_evidence_state:"SUPPORTED",runway_magnitude:"MATERIAL",runway_horizon:"MODERATE",
        roic_trend:"UNCLEAR",forensic_reliability:"WATCH",
        sector_method_results:[{method:"Synthetic ROIC computation",status:"APPLIED",
          rationale:"Synthetic evidence supports only a conservative annual return estimate.",
          evidence_ids:["E-001"]}],
      },
      analytical_metrics:{
        standard_roic:20,all_in_roic:20,roic_ex_goodwill:20,rd_adjusted_roic:"NOT_INTERPRETABLE",
        roiic:"NOT_INTERPRETABLE",economic_spread:"CLEARLY_POSITIVE",
        return_interpretability:"MEDIUM",return_attributability:"MEDIUM",
        standardized_fcf:1000000,owner_earnings:"UNKNOWN",
        fcf_per_share:"NOT_AVAILABLE",share_count_cagr:"NOT_AVAILABLE",
      },
      certification:{
        business_research_status:"CERTIFIED_WITH_LIMITATIONS",
        investment_conclusion_status:"CERTIFIED_WITH_LIMITATIONS",score_permission:"ALLOWED",
        hard_blockers:[],material_limitations:[],critical_unknowns:[],unresolved_conflicts:[],required_fixes:[],
      },
      business_quality:{
        moat_score:50,runway_score:50,return_quality_score:50,cash_economics_score:50,
        capital_allocation_score:50,management_governance_score:50,resilience_risk_score:50,
        oqs_raw:50,weak_link_cap:75,oqs:50,quality_class:"MIXED / ADEQUATE",
        dimension_evidence_ids:{
          MOAT:["E-001"],RUNWAY:["E-001"],RETURN_QUALITY:["E-001"],CASH_ECONOMICS:["E-001"],
          CAPITAL_ALLOCATION:["E-001"],MANAGEMENT_GOVERNANCE:["E-001"],RESILIENCE_RISK:["E-001"],
        },
      },
    },
    l3_investment_valuation:{
      valuation:{
        primary_expected_return:10,return_horizon:5,
        no_multiple_expansion_return:10,mature_normalization_return:10,
        margin_of_safety:"NONE",valuation_reliability:"LOW",
        market_expectation_gap:"ABOVE_FUNDAMENTAL_CASE",ovs:55,
        price_ladder:{
          valuation_date:cutoff,data_cutoff:cutoff,reference_price:100,currency:"USD",
          required_return_h:10,strong_return_threshold:12.5,exceptional_return_threshold:15,
          price_for_required_return_h:90,price_for_strong_return:80,price_for_exceptional_return:70,
          investable_price_zone:{max_price:90,currency:"USD",basis:"Synthetic 10 percent entry threshold"},
          strong_opportunity_zone:{max_price:80,currency:"USD",basis:"Synthetic 12.5 percent entry threshold"},
          potential_orotitan_price_zone:"NOT_AVAILABLE",potential_orotitan_max_price:"NOT_AVAILABLE",
          potential_orotitan_components:{
            p_er:"NOT_AVAILABLE",p_no_expansion:"NOT_AVAILABLE",p_mos:"NOT_AVAILABLE",p_implied:"NOT_AVAILABLE",
          },
          assumptions:["Synthetic assumptions only."],assumptions_version:"assumptions@1",
          method_version:"method@1",calculation_version:"calc@1",
        },
      },
      investment:{investment_raw:51.5,investment_score:50,investment_class:"POOR"},
    },
    l4_operational_state:{
      dossier_readiness:"READY",opportunity_path:"PREPARED",next_action:"WAIT_FOR_PRICE",
      price_ladder_status:"CURRENT",thesis_invalidation_triggers:["Synthetic price assumption invalidated."],
      activation_check_required:true,
      refresh_class:"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED",
      orotitan:{
        orotitan_status:"NO",
        orotitan_gate_results:gateMap.map(([gate])=>({gate,state:"FAIL",rationale:"Synthetic negative gate.",evidence_ids:[]})),
        orotitan_gaps:gateMap.map(([,gap_code])=>({gap_code,rationale:"Synthetic negative gate.",material_evidence_ids:[]})),
      },
    },
    traceability:{report_id:"synthetic-analysis-report",evidence_ids:["E-001"],calculation_ids:["CALC-001"]},
    v2_product:{
      classification:{
        issuer_country_code:"US",primary_listing_country_code:"US",sector:"INFORMATION_TECHNOLOGY",
        industry_group:"SOFTWARE",business_model_primary:"RECURRING_SUBSCRIPTION",
        business_model_secondary:null,economic_exposure_regions:["GLOBAL"],taxonomy_version:"OROTITAN_TAXONOMY_V2.0",
      },
      business_summary:{business_description_short:"Synthetic recurring subscription software for institutions."},
      investment_thesis:{
        quality_case:"Quality evidence is limited but traceable. ".repeat(12),
        valuation_case:"Valuation is sensitive to pricing and durable cash flows. ".repeat(8),
        key_risk:"Synthetic competitive substitution risk.",
      },
      portfolio_filters:{pea_eligibility:"UNKNOWN"},
    },
  };
}

function seal(obj: Record<string, unknown>, id:string){
  const utf8Body=JSON.stringify(obj);
  return {utf8Body,registeredSha256:createHash("sha256").update(utf8Body).digest("hex"),
    registeredSizeBytes:Buffer.byteLength(utf8Body),registeredArtifactId:id,registeredVersion:1};
}

test("full synthetic ANALYZE respects the live composed V2.0.12 validation path",()=>{
  const input=analyze();
  const original=JSON.stringify(input);
  const legacy=validateV2ResearchSnapshotForPersistence(input,dossierId);
  assert.equal(legacy.ok,false);
  const admitted=validateV2ResearchSnapshotWithExactCertifiedLiterals(input,dossierId);
  assert.equal(admitted.ok,true,JSON.stringify(admitted));
  assert.equal(JSON.stringify(input),original);
});

test("real semantic boundary is only reachable with both SHA-sealed source literals",()=>{
  const input=analyze();
  const lockId="00000000-0000-4000-8000-000000000005";
  const readyId="00000000-0000-4000-8000-000000000006";
  const lock=seal({
    run_id:runId,issuer_id:issuerId,security_id:securityId,dossier_id:dossierId,
    data_cutoff:cutoff,stage:"DEEP_DIVE",artifact_type:"FUNDAMENTALS_LOCK",version:1,
    artifact:{artifact_id:lockId,version:1},
    canonical_fundamental_verdicts:{RETURN_INTERPRETABILITY:"MEDIUM"},
  },lockId);
  const ready=seal({
    run_id:runId,data_cutoff:cutoff,stage:"DEEP_DIVE",
    artifact_type:"READINESS_NEXT_ACTION_ARTIFACT",version:1,
    refresh_class:"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED",
  },readyId);
  const admit=validateFicoStyleExactLiteralsFromSealedSources({
    runId,issuerId,securityId,dossierId,dataCutoff:cutoff,
    frozenFundamentalsLock:lock,frozenReadiness:ready,candidate:input,
  });
  assert.deepEqual(admit,{ok:true,stage:"EXACT_SOURCE_LITERAL_ADMITTED",semanticValidator:"PASS"});
});
