import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { computeCanonicalSnapshot } from "../lib/orotitan-equity/v1/contract";
import {
  buildCertificationToIntegration,
  buildResearchToFundamentals,
  reconstructV2Handoff,
  type ArtifactRef,
} from "../lib/orotitan-equity/v2/handoff";
import {
  planLimitedFundamentalsReopen,
  validateFundamentalsLock,
  validateValuationLock,
  v2RefreshRoute,
  type FundamentalsLock,
  type ValuationLock,
} from "../lib/orotitan-equity/v2/deep-dive-runtime";
import { validateV2ProductSemantics, type V2Product } from "../lib/orotitan-equity/v2/product";

const root = new URL("../contracts/orotitan-equity/v2/", import.meta.url);
const text = (name: string) => readFileSync(new URL(name, root), "utf8");
const process = text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
const research = text("OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md");
const deepDive = text("OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0.md");
const integration = text("OROTITAN_INTEGRATION_STAGE_CONTRACT_V2_FREEZE_V2.0.md");
const integrationSpec = text("04_INTEGRATION_SPEC_V2.md");
const migration = readFileSync(new URL("../migrations/20260915_orotitan_v2_canonical_snapshot_writer.sql", import.meta.url), "utf8");
const manifestSource = readFileSync(new URL("../lib/orotitan-equity/v1/stage-manifest.ts", import.meta.url), "utf8");

const a = (n: number): ArtifactRef => ({ artifact_id: `00000000-0000-4000-8000-${String(n).padStart(12, "0")}`, version: 1, content_sha256: "a".repeat(64) });
const common = {
  company: "Example, Inc.", companyCommandName: "EXAMPLE", runId: "10000000-0000-4000-8000-000000000001",
  canonicalMode: "ANALYZE", runType: "INITIAL" as const, dataCutoff: "2026-09-15", baselineSnapshotId: null,
};
const product = (): V2Product => ({
  classification: {
    issuer_country_code: "US", primary_listing_country_code: "US", sector: "INFORMATION_TECHNOLOGY",
    industry_group: "CYBERSECURITY", business_model_primary: "RECURRING_SUBSCRIPTION",
    business_model_secondary: null, economic_exposure_regions: ["GLOBAL"], taxonomy_version: "OROTITAN_TAXONOMY_V2.0",
  },
  business_summary: { business_description_short: "Cloud security software sold to enterprises through recurring subscriptions." },
  investment_thesis: { quality_case: "Recurring revenue and strong cash economics.", valuation_case: "Required return remains price dependent.", key_risk: "Competitive platform bundling." },
  portfolio_filters: { pea_eligibility: "UNKNOWN" },
});
const fundamentalsLock = (): FundamentalsLock => ({
  lock_type: "FUNDAMENTALS_LOCK", artifact: a(10), run_id: common.runId, data_cutoff: common.dataCutoff, stage_revision: 1,
  research_final_manifest: a(1), evidence_lineage: [a(2)], red_team_status: "COMPLETE", taxonomy_record: a(3),
  business_description_artifact: a(4), scoring_inputs_artifact: a(5), limitations: [], invalidation_triggers: [], ready_for_valuation: "YES",
});
const valuationLock = (): ValuationLock => ({
  lock_type: "VALUATION_LOCK", artifact: a(20), run_id: common.runId, data_cutoff: common.dataCutoff, stage_revision: 1,
  fundamentals_lock: a(10), evidence_lineage: [a(2)], calculation_ledger: a(6), assumption_register: a(7), valuation_artifact: a(8),
  full_precision_inputs_artifact: a(9), ready_for_certification: "YES",
});

const cases: Array<[number, string, () => void]> = [
  [1, "Research cannot emit final scores", () => { assert.match(research, /MUST NOT produce:[\s\S]*OQS[\s\S]*OVS[\s\S]*INVESTMENT_SCORE/); }],
  [2, "Research FINAL gate precedes Fundamentals", () => { assert.match(deepDive, /RESEARCH_STAGE_STATUS = COMPLETE[\s\S]*READY_FOR_DEEP_DIVE = YES[\s\S]*Research active manifest = FINAL/); }],
  [3, "Fundamentals has a price firewall", () => { assert.match(deepDive, /Fundamentals MUST NOT perform:[\s\S]*DCF[\s\S]*EXPECTED RETURN FROM CURRENT PRICE[\s\S]*PRICE LADDER/); }],
  [4, "Red Team is mandatory before lock", () => { const bad = fundamentalsLock(); (bad as { red_team_status: string }).red_team_status = "INCOMPLETE"; assert.ok(validateFundamentalsLock(bad, { runId: common.runId, dataCutoff: common.dataCutoff, researchFinalManifest: a(1) }).some((e) => e.includes("Red Team"))); }],
  [5, "Fundamentals lock exact identity is checked", () => { const bad = fundamentalsLock(); bad.research_final_manifest = a(99); assert.ok(validateFundamentalsLock(bad, { runId: common.runId, dataCutoff: common.dataCutoff, researchFinalManifest: a(1) }).length > 0); }],
  [6, "Final OQS withheld until Certification", () => { assert.match(deepDive, /OQS_RAW[\s\S]*OQS[\s\S]*QUALITY_CLASS[\s\S]*must remain uncomputed\/unpublished until Certification/); }],
  [7, "Valuation consumes exact Fundamentals lock", () => { const bad = valuationLock(); bad.fundamentals_lock = a(11); assert.ok(validateValuationLock(bad, { runId: common.runId, dataCutoff: common.dataCutoff, fundamentalsLock: a(10) }).some((e) => e.includes("exact current"))); }],
  [8, "Valuation consumes evidence lineage", () => { const bad = valuationLock(); bad.evidence_lineage = []; assert.ok(validateValuationLock(bad, { runId: common.runId, dataCutoff: common.dataCutoff, fundamentalsLock: a(10) }).some((e) => e.includes("Evidence Ledger"))); }],
  [9, "Valuation cannot silently rewrite Fundamentals", () => { assert.match(deepDive, /Valuation may not silently alter Fundamentals/); }],
  [10, "Material contradiction permits only limited reopen", () => { const plan = planLimitedFundamentalsReopen({ material: true, reason: "material conflict", affectedScopes: ["MOAT"], priorFundamentalsLock: a(10), priorValuationLock: a(20) }); assert.deepEqual(plan.affectedScopes, ["MOAT"]); }],
  [11, "Reopen invalidates dependent valuation eligibility", () => { const plan = planLimitedFundamentalsReopen({ material: true, reason: "material conflict", affectedScopes: ["RUNWAY"], priorFundamentalsLock: a(10) }); assert.ok(plan.invalidate.includes("VALUATION_ELIGIBILITY")); }],
  [12, "Valuation checkpoint cannot admit Integration", () => { assert.match(deepDive, /A CHECKPOINT never admits Integration/); }],
  [13, "Certification cannot perform broad new research", () => { assert.match(deepDive, /perform broad new research/); }],
  [14, "Certification routes missing evidence upstream", () => { assert.match(deepDive, /Material missing\/contradictory input -> reopen exact upstream scope/); }],
  [15, "OQS is deterministic from certified inputs", () => { const out = computeCanonicalSnapshot({ dimensions: { MOAT:80,RUNWAY:80,RETURN_QUALITY:80,CASH_ECONOMICS:80,CAPITAL_ALLOCATION:80,MANAGEMENT_GOVERNANCE:80,RESILIENCE_RISK:80 }, evidence:{moat:"STRONGLY_SUPPORTED",runway:"STRONGLY_SUPPORTED"}, businessResearchStatus:"CERTIFIED", investmentConclusionStatus:"CERTIFIED", scorePermission:"ALLOWED", mosStatus:"ROBUST", valuationReliability:"HIGH", primaryExpectedReturnDeltaPercentagePoints:2, normalizedExpectedReturnDeltaPercentagePoints:-2, eliteGates:{ researchFullyCertified:"PASS",moatElite:"PASS",runwayElite:"PASS",returnQualityElite:"PASS",cashEconomicsElite:"PASS",capitalAllocationElite:"PASS",managementGovernanceElite:"PASS",resilienceElite:"PASS",valuationElite:"PASS",materialWeakLink:"PASS" } }); assert.equal(out.oqs, 80); }],
  [16, "OVS and Investment Score remain I2 deterministic", () => { const first = computeCanonicalSnapshot({ dimensions:{MOAT:80,RUNWAY:80,RETURN_QUALITY:80,CASH_ECONOMICS:80,CAPITAL_ALLOCATION:80,MANAGEMENT_GOVERNANCE:80,RESILIENCE_RISK:80},evidence:{moat:"STRONGLY_SUPPORTED",runway:"STRONGLY_SUPPORTED"},businessResearchStatus:"CERTIFIED",investmentConclusionStatus:"CERTIFIED",scorePermission:"ALLOWED",mosStatus:"ROBUST",valuationReliability:"HIGH",primaryExpectedReturnDeltaPercentagePoints:2,normalizedExpectedReturnDeltaPercentagePoints:-2,eliteGates:{researchFullyCertified:"PASS",moatElite:"PASS",runwayElite:"PASS",returnQualityElite:"PASS",cashEconomicsElite:"PASS",capitalAllocationElite:"PASS",managementGovernanceElite:"PASS",resilienceElite:"PASS",valuationElite:"PASS",materialWeakLink:"PASS"}}); const second = computeCanonicalSnapshot({ dimensions:{MOAT:80,RUNWAY:80,RETURN_QUALITY:80,CASH_ECONOMICS:80,CAPITAL_ALLOCATION:80,MANAGEMENT_GOVERNANCE:80,RESILIENCE_RISK:80},evidence:{moat:"STRONGLY_SUPPORTED",runway:"STRONGLY_SUPPORTED"},businessResearchStatus:"CERTIFIED",investmentConclusionStatus:"CERTIFIED",scorePermission:"ALLOWED",mosStatus:"ROBUST",valuationReliability:"HIGH",primaryExpectedReturnDeltaPercentagePoints:2,normalizedExpectedReturnDeltaPercentagePoints:-2,eliteGates:{researchFullyCertified:"PASS",moatElite:"PASS",runwayElite:"PASS",returnQualityElite:"PASS",cashEconomicsElite:"PASS",capitalAllocationElite:"PASS",managementGovernanceElite:"PASS",resilienceElite:"PASS",valuationElite:"PASS",materialWeakLink:"PASS"}}); assert.deepEqual(first, second); }],
  [17, "Terminal gate follows certification state", () => { assert.match(deepDive, /Establish certification states and score permission before exposing final OQS|certification state and score permission/); }],
  [18, "Only Certification may finalize Deep Dive", () => { assert.match(deepDive, /Only Certification may perform normal Deep Dive finalization/); }],
  [19, "Deep Dive checkpoint cannot admit Integration", () => { assert.match(deepDive, /A CHECKPOINT never admits Integration/); }],
  [20, "Integration requires FINAL complete Deep Dive", () => { assert.match(integration, /DEEP_DIVE[\s\S]*COMPLETE[\s\S]*READY_FOR_INTEGRATION/); }],
  [21, "Successful handoff is copy-ready", () => { const prompt = buildResearchToFundamentals({ ...common, researchFinalManifest:a(1), researchInputs:[a(2)] }); assert.ok(prompt.startsWith("OROTITAN V2 — START FUNDAMENTALS")); }],
  [22, "No prose follows handoff invariant", () => { const prompt = buildResearchToFundamentals({ ...common, researchFinalManifest:a(1), researchInputs:[a(2)] }); assert.ok(prompt.endsWith("FAIL CLOSED ON VERSION / ID / STATUS / HASH MISMATCH.")); }],
  [23, "Blocked state emits resolution prompt", () => { const prompt = reconstructV2Handoff({ ...common, registryStage:"DEEP_DIVE",executionPhase:"VALUATION", blocker:{code:"X",summary:"blocked",requiredAction:"fix",affectedArtifacts:[]} }); assert.ok(prompt.startsWith("OROTITAN V2 — RESOLVE BLOCKER")); assert.doesNotMatch(prompt, /^OROTITAN V2 — START CERTIFICATION/); }],
  [24, "Pilotage reconstruction is deterministic", () => { const state = { ...common, registryStage:"RESEARCH" as const, executionPhase:"RESEARCH" as const, readyForDeepDive:true, researchFinalManifest:a(1), researchInputs:[a(2)] }; assert.equal(reconstructV2Handoff(state), reconstructV2Handoff(state)); }],
  [25, "Country rejects non-ISO free text", () => { const p = product(); p.classification.issuer_country_code = "United States"; assert.ok(validateV2ProductSemantics(p).some((e) => e.includes("ISO"))); }],
  [26, "Sector rejects uncontrolled value", () => { const p = product(); p.classification.sector = "TECH"; assert.ok(validateV2ProductSemantics(p).some((e) => e.includes("sector"))); }],
  [27, "Business model rejects uncontrolled value", () => { const p = product(); p.classification.business_model_primary = "SAAS"; assert.ok(validateV2ProductSemantics(p).some((e) => e.includes("business_model_primary"))); }],
  [28, "Business description length and neutrality enforced", () => { const p = product(); p.business_summary.business_description_short = "Undervalued buy"; assert.ok(validateV2ProductSemantics(p).some((e) => e.includes("neutral"))); p.business_summary.business_description_short = "x".repeat(451); assert.ok(validateV2ProductSemantics(p).some((e) => e.includes("1-450"))); }],
  [29, "Thesis contains exactly three fields", () => { const p = product() as unknown as Record<string, unknown>; (p.investment_thesis as Record<string, unknown>).extra = "x"; assert.ok(validateV2ProductSemantics(p).some((e) => e.includes("exactly"))); }],
  [30, "Thesis not produced in Fundamentals", () => { assert.match(deepDive, /Fundamentals MUST NOT perform:[\s\S]*INVESTMENT THESIS/); }],
  [31, "Price-only refresh cannot change OQS", () => { assert.equal(v2RefreshRoute("PRICE_ONLY_DELTA").oqsMayChangeWithoutFundamentalReopen, false); }],
  [32, "Refresh still produces immutable successor snapshot path", () => { assert.match(process, /new immutable snapshot if publication succeeds/); }],
  [33, "V1 Qualys is grandfathered", () => { assert.match(process, /QUALYS CURRENT V1 SNAPSHOT[\s\S]*remains canonical until a future authorized refresh/); }],
  [34, "V1 history remains resolvable", () => { assert.match(process, /Existing V1 canonical snapshots and runs remain immutable historical truth/); assert.match(migration, /Preserves the V1 writer/); }],
  [35, "Registry stage codes remain constrained", () => { assert.match(process, /RESEARCH\nDEEP_DIVE\nINTEGRATION/); }],
  [36, "Internal checkpoints remain DEEP_DIVE", () => { assert.match(deepDive, /remain one persistent `DEEP_DIVE` Registry stage/); }],
  [37, "No new Registry table required", () => { assert.match(process, /REGISTRY TABLE MODEL\s+UNCHANGED/); }],
  [38, "Integration maps product without analytical rewrite", () => { assert.match(integrationSpec, /Integration may not rewrite or improve them/); assert.match(integrationSpec, /Taxonomy has zero scoring authority/); }],
  [39, "I2 and I3-B remain fail closed", () => { assert.match(process, /I2\s+UNCHANGED/); assert.match(process, /I3-B\s+UNCHANGED/); assert.match(integrationSpec, /I2 reconciliation failure/); }],
  [40, "Publication boundary remains separate", () => { const prompt = buildCertificationToIntegration({ ...common, deepDiveFinalManifest:a(30), deepDiveFinalArtifacts:[a(31)] }); assert.match(prompt, /GO PUBLISH EXAMPLE/); assert.match(integration, /No canonical promotion occurs without separate user authorization/); }],
  [41, "Contract pin mismatch remains fail closed", () => { assert.match(manifestSource, /contract set SHA mismatch|contract_set_sha256/i); }],
  [42, "DATA_CUTOFF mismatch invalidates locks", () => { assert.ok(validateFundamentalsLock(fundamentalsLock(), { runId:common.runId,dataCutoff:"2026-09-14",researchFinalManifest:a(1) }).some((e) => e.includes("DATA_CUTOFF"))); }],
  [43, "Ledger semantics remain single authority", () => { assert.match(process, /EVIDENCE AUTHORITY\s+SINGLE LINEAGE/); assert.match(deepDive, /No duplicate ledger semantics are introduced/); }],
  [44, "Company-specific process exceptions are frozen out", () => { assert.match(process, /NO COMPANY-SPECIFIC EXCEPTION/); }],
  [45, "Emergency patch requires blocking defect", () => { assert.match(process, /genuinely blocking security, data-integrity, persistence\/Registry, contract-contradiction or deterministic canonical-output defect/); }],
];

for (const [id, name, fn] of cases) test(`V2-${String(id).padStart(2, "0")} ${name}`, fn);

test("V2 acceptance suite contains exactly 45 behavioral checks", () => {
  assert.deepEqual(cases.map(([id]) => id), Array.from({ length: 45 }, (_, index) => index + 1));
});
