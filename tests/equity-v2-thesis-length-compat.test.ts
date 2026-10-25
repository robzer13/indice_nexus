import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { validateV2ResearchSnapshotForPersistence } from "../lib/orotitan-equity/v2/research-snapshot-schema";
import {
  validateV2ResearchSnapshotForPersistenceWithLongThesis,
} from "../lib/orotitan-equity/v2/research-snapshot-schema-thesis-compat";
import {
  validateV2ProductWithLongCertifiedThesis,
  compactThesisForDisplay,
} from "../lib/orotitan-equity/v2/thesis-length-compatibility";
import {
  persistValidatedV2ResearchSnapshotWithLongThesis,
} from "../lib/orotitan-equity/v2/persistence-core-thesis-compat";
import { validateV2ProductSemantics } from "../lib/orotitan-equity/v2/product";

const dossierId = "00000000-0000-4000-8000-000000000001";

function v2Discover(): Record<string, unknown> {
  return {
    snapshot_id: "00000000-0000-4000-8000-000000000004",
    report_id: "v2-report-1",
    issuer_id: issuerId,
    security_id: securityId,
    execution_mode: "DISCOVER",
    data_lock: {
      data_cutoff: "2026-09-15",
      reference_price: 10,
      reference_price_currency: "USD",
      reference_price_date: "2026-09-15",
      calculation_date: "2026-09-15",
      last_full_research_date: "NOT_AVAILABLE",
      score_date: "NOT_AVAILABLE",
    },
    versions: {
      report_version: "r2",
      method_version: "m1",
      calculation_version: "c1",
      evidence_ledger_version: "e2",
      method_versions: {
        discovery_version: "d1", moat_version: "m1", runway_version: "r1", roic_version: "roic1",
        fcf_version: "fcf1", valuation_version: "v1", certification_version: "cert1",
      },
    },
    l2_research_fundamentals: {
      discovery_record: {
        discovery_status: "DEEP_DIVE_CANDIDATE",
        discovery_methods: ["S1"],
        archetype_hints: ["A1"],
        attention_state: "HIGH",
        evidence_grade: "E1 OBSERVABLE_ISSUER_EVIDENCE",
        data_sufficiency: "SUFFICIENT",
        hard_kill_status: { "HK-A": "NOT_FOUND", "HK-B": "NOT_FOUND", "HK-C": "NOT_FOUND", "HK-D": "NOT_FOUND", "HK-E": "NOT_FOUND" },
        valuation_sanity: "NOT_ASSESSABLE",
        research_priority: "HIGH",
        watch_type: "NOT_APPLICABLE",
        reactivation_trigger: "NOT_APPLICABLE",
        rejection_reason: "NOT_APPLICABLE",
      },
    },
    traceability: { report_id: "v2-report-1", evidence_ids: ["evidence-1"], calculation_ids: ["calculation-1"] },
    v2_product: {
      classification: {
        issuer_country_code: "US",
        primary_listing_country_code: "US",
        sector: "INFORMATION_TECHNOLOGY",
        industry_group: "CYBERSECURITY",
        business_model_primary: "RECURRING_SUBSCRIPTION",
        business_model_secondary: null,
        economic_exposure_regions: ["GLOBAL"],
        taxonomy_version: "OROTITAN_TAXONOMY_V2.0",
      },
      business_summary: { business_description_short: "Cloud security software sold to enterprises through recurring subscriptions." },
      investment_thesis: {
        quality_case: "Recurring subscription economics support durable cash generation.",
        valuation_case: "Expected return remains sensitive to the entry price.",
        key_risk: "Competitive platform bundling can pressure growth and pricing.",
      },
      portfolio_filters: { pea_eligibility: "NO", pea_eligibility_as_of: "2026-09-15", pea_eligibility_source_ref: "issuer/listing-jurisdiction" },
    },
  };
}



const thesis = (snapshot: Record<string, unknown>): Record<string, unknown> =>
  ((snapshot.v2_product as Record<string, unknown>).investment_thesis as Record<string, unknown>);

function longThesis(): Record<string, unknown> {
  const snapshot = v2Discover();
  thesis(snapshot).quality_case = "Q".repeat(385);
  thesis(snapshot).valuation_case = "V".repeat(284);
  thesis(snapshot).key_risk = "R".repeat(1100);
  return snapshot;
}

test("frozen V2 legacy path still rejects over-240 certified thesis", () => {
  const input = longThesis();
  const result = validateV2ResearchSnapshotForPersistence(input, dossierId);
  assert.equal(result.ok, false);
  assert.ok(validateV2ProductSemantics(input.v2_product).some(e =>
    e.includes("quality_case") && e.includes("240")));
});

test("opt-in V2.0.11 accepts lossless, exact 385/284/1100 character thesis", () => {
  const input = longThesis();
  const original = JSON.stringify(input);
  const result = validateV2ResearchSnapshotForPersistenceWithLongThesis(input, dossierId);
  assert.equal(result.ok, true, JSON.stringify(result));
  assert.equal(JSON.stringify(input), original);
  if (result.ok) {
    assert.strictEqual(result.snapshot, input);
    assert.equal(thesis(result.snapshot).quality_case, "Q".repeat(385));
    assert.equal(thesis(result.snapshot).valuation_case, "V".repeat(284));
    assert.equal(thesis(result.snapshot).key_risk, "R".repeat(1100));
  }
});

test("short canonical V2 snapshot remains accepted in old and opt-in validators", () => {
  const v = v2Discover();
  assert.equal(validateV2ResearchSnapshotForPersistence(v, dossierId).ok,true);
  assert.equal(validateV2ResearchSnapshotForPersistenceWithLongThesis(v, dossierId).ok,true);
});

test("new overlay only differs by three thesis maxLength restrictions and metadata", () => {
  const f = (p: string): Record<string, unknown> =>
    JSON.parse(readFileSync(new URL(p, import.meta.url),"utf8"));
  const frozen = f("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2.json");
  const compatible = f("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2_THESIS_COMPAT_V2.0.11.json");
  for (const k of ["$id","title","description","x-orotitan-compatibility-authority","x-orotitan-difference"]) {
    delete frozen[k];delete compatible[k];
  }
  const p = (((frozen.properties as any).v2_product.properties.investment_thesis.properties)) as Record<string,{minLength?:number,maxLength?:number}>;
  for (const key of ["quality_case","valuation_case","key_risk"]) {
    assert.equal(p[key].maxLength,240);
    assert.equal(p[key].minLength,1);
    delete p[key].maxLength;
  }
  assert.deepEqual(compatible,frozen);
});

test("non-blank thesis and exactly three fields remain compulsory", () => {
  for (const badValue of ["", "   ", null, 7, {}, []]) {
    const v=longThesis();
    thesis(v).quality_case=badValue;
    const r=validateV2ResearchSnapshotForPersistenceWithLongThesis(v,dossierId);
    assert.equal(r.ok,false,`invalid quality_case: ${JSON.stringify(badValue)}`);
  }
  const v=longThesis(); thesis(v).extra_field="not permitted";
  assert.equal(validateV2ResearchSnapshotForPersistenceWithLongThesis(v,dossierId).ok,false);
});

test("other product controls are still enforced unchanged", () => {
  const badProduct = longThesis();
  const prod = badProduct.v2_product as Record<string,unknown>;
  (prod.classification as Record<string,unknown>).issuer_country_code="USA";
  assert.ok(validateV2ProductWithLongCertifiedThesis(prod).some(e=>e.includes("alpha-2")));
  (prod.classification as Record<string,unknown>).issuer_country_code="US";
  (prod.business_summary as Record<string,unknown>).business_description_short="x".repeat(451);
  assert.equal(validateV2ResearchSnapshotForPersistenceWithLongThesis(badProduct,dossierId).ok,false);
});

test("V1 core and deterministic checks still fail closed",()=>{
  const v=longThesis();delete v.data_lock;
  const r=validateV2ResearchSnapshotForPersistenceWithLongThesis(v,dossierId);
  assert.equal(r.ok,false);
  if(!r.ok)assert.equal(r.stage,"schema");
});

test("compact UI preview is never the canonical stored text",()=>{
  const full="Quality economics — "+"é".repeat(900);
  const out=compactThesisForDisplay(full,650);
  assert.equal(out.fullText,full);
  assert.equal(out.truncated,true);
  assert.ok(out.preview.endsWith("…"));
  assert.ok(out.preview.length<full.length);
  assert.equal(compactThesisForDisplay(full,10000).preview,full);
  assert.throws(()=>compactThesisForDisplay(full,0));
});

test("opt-in writer sends exact admitted canonical payload only on validation PASS",async()=>{
  const input=longThesis();
  let calls=0;
  const result=await persistValidatedV2ResearchSnapshotWithLongThesis({
    dossierId,expectedCurrentSnapshotId:null,canonicalPayload:input,
  },async args=>{
    calls++;
    assert.strictEqual(args.p_canonical_payload,input);
    assert.equal(args.p_dossier_id,dossierId);
    assert.equal(args.p_expected_current_snapshot_id,null);
    return {data:{status:"INSERTED",dossier_id:dossierId,snapshot_id:input.snapshot_id,current_snapshot_id:input.snapshot_id},error:null};
  });
  assert.equal(calls,1);
  assert.equal(result.status,"INSERTED");
  const invalid=longThesis();
  (invalid.v2_product as Record<string,unknown>).classification="untrusted";
  await assert.rejects(persistValidatedV2ResearchSnapshotWithLongThesis({
    dossierId,expectedCurrentSnapshotId:null,canonicalPayload:invalid,
  },async()=>{
    calls++;
    throw new Error("RPC MUST NOT BE CALLED ON INVALID INPUT");
  }));
  assert.equal(calls,1);
});
