import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = "contracts/orotitan-equity/v2";

async function text(name: string): Promise<string> {
  return readFile(`${root}/${name}`, "utf8");
}

test("V2 process freezes the five-discussion sequence without changing Registry stage codes", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  for (const phase of ["RESEARCH", "FUNDAMENTALS", "VALUATION", "CERTIFICATION_RECONCILIATION", "INTEGRATION"]) {
    assert.match(process, new RegExp(phase));
  }
  assert.match(process, /Registry stage codes remain unchanged/);
  assert.match(process, /RESEARCH\nDEEP_DIVE\nINTEGRATION/);
  assert.match(process, /Methodology change:\*\* NO/);
  assert.match(process, /I2 \/ I3-B change:\*\* NO/);
});

test("V2 keeps final OQS behind Certification and Red Team before Fundamentals Lock", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  const deepDive = await text("OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0.md");
  assert.match(process, /final `OQS_RAW`, `OQS` and `QUALITY_CLASS` MUST NOT be calculated or displayed before Certification/);
  assert.match(deepDive, /Fundamental Red Team must be completed before the phase can lock/);
  assert.match(deepDive, /Final OQS may be calculated\/displayed only after required certification state/);
});

test("Valuation consumes both exact Fundamentals Lock and the full evidence lineage", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  const deepDive = await text("OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0.md");
  assert.match(process, /Mandatory inputs include BOTH/);
  assert.match(deepDive, /current FUNDAMENTALS_LOCK version/);
  assert.match(deepDive, /full authoritative Evidence Ledger lineage/);
  assert.match(deepDive, /Valuation may not silently alter Fundamentals/);
});

test("Deep Dive phase handoffs use CHECKPOINT and only Certification can emit FINAL", async () => {
  const deepDive = await text("OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V2_FREEZE_V2.0.md");
  assert.match(deepDive, /A CHECKPOINT never admits Integration/);
  assert.match(deepDive, /Only Certification may perform normal Deep Dive finalization/);
  assert.match(deepDive, /READY_FOR_INTEGRATION = YES/);
});

test("handoff UX is deterministic and mobile-safe", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  const templates = await text("OROTITAN_HANDOFF_PROMPT_TEMPLATES_V2.0.md");
  assert.match(process, /prompt is final visible block/);
  assert.match(process, /no prose after prompt/);
  assert.match(templates, /Research -> Fundamentals/);
  assert.match(templates, /Fundamentals -> Valuation/);
  assert.match(templates, /Valuation -> Certification/);
  assert.match(templates, /Certification -> Integration/);
  assert.match(templates, /DO NOT USE THIS BOOTSTRAP AS ANALYTICAL EVIDENCE\./);
});

test("taxonomy is closed for sector and business model and has zero scoring authority", async () => {
  const raw = await text("OROTITAN_TAXONOMY_V2.0.json");
  const taxonomy = JSON.parse(raw) as {
    taxonomy_version: string;
    sectors: string[];
    industry_groups: string[];
    business_models: string[];
    rules: Record<string, unknown>;
  };
  assert.equal(taxonomy.taxonomy_version, "OROTITAN_TAXONOMY_V2.0");
  assert.equal(taxonomy.sectors.length, 11);
  assert.ok(taxonomy.industry_groups.includes("OTHER"));
  assert.ok(taxonomy.business_models.includes("HYBRID"));
  assert.equal(taxonomy.rules.industry_group_free_text_allowed, false);
  assert.equal(taxonomy.rules.business_model_free_text_allowed, false);
  assert.equal(taxonomy.rules.taxonomy_has_zero_scoring_authority, true);
});

test("canonical product extension requires classification, business summary and structured thesis", async () => {
  const raw = await text("OROTITAN_CANONICAL_PRODUCT_EXTENSION_V2.schema.json");
  const schema = JSON.parse(raw) as {
    required: string[];
    properties: Record<string, unknown>;
    [key: string]: unknown;
  };
  assert.deepEqual(schema.required, ["classification", "business_summary", "investment_thesis"]);
  assert.ok(schema.properties.classification);
  assert.ok(schema.properties.business_summary);
  assert.ok(schema.properties.investment_thesis);
});

test("V2 freezes Qualys grandfathering and targeted refresh", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  assert.match(process, /QUALYS CURRENT V1 SNAPSHOT/);
  assert.match(process, /remains canonical until a future authorized refresh/);
  assert.match(process, /PRICE_ONLY_DELTA/);
  assert.match(process, /ROUTINE_FUNDAMENTAL_DELTA/);
  assert.match(process, /FULL_REFRESH_REQUIRED/);
});

test("V2 freeze cannot be production-activated by design docs alone", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  const checklist = await text("OROTITAN_V2_IMPLEMENTATION_ADMISSION_CHECKLIST.md");
  assert.match(process, /PRODUCTION_ACTIVATION = BLOCKED UNTIL V2 IMPLEMENTATION ADMISSION PASSES/);
  assert.match(checklist, /GO ACTIVATE OROTITAN V2/);
  assert.match(checklist, /V2_PRODUCTION_STATUS = NOT_ACTIVE/);
});

test("V2 acceptance matrix contains exactly V2-01 through V2-45", async () => {
  const process = await text("OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0.md");
  const ids = [...process.matchAll(/V2-(\d{2}) /g)].map((m) => Number(m[1]));
  assert.deepEqual(ids, Array.from({ length: 45 }, (_, i) => i + 1));
});
