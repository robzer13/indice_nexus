import assert from "node:assert/strict";
import test from "node:test";
import { validateAnalyticalDataPackage } from "../lib/orotitan-equity/post-c7/analytical-data-contracts-v2";

const runId = "11111111-1111-4111-8111-111111111111";
const issuerId = "22222222-2222-4222-8222-222222222222";
const context = {
  runId,
  issuerId,
  securityId: null,
  dossierId: null,
  dataCutoff: "2026-10-01",
};

function dnaDimension(summary: string | null = null, evidenceIds: string[] = []) {
  return { summary, evidence_ids: evidenceIds };
}

type MutableTestPackage = Record<string, unknown> & {
  sources: Array<Record<string, unknown>>;
  evidence: Array<Record<string, unknown> & { numeric_data: Array<Record<string, unknown>> }>;
  gaps: Array<Record<string, unknown>>;
  analytical_blocks: Array<Record<string, unknown> & {
    status: string;
    conclusion: string | null;
    supporting_evidence_ids: string[];
    gap_ids: string[];
    upstream_block_refs: string[];
    sector_overlays: Array<Record<string, unknown>>;
  }>;
  material_changes: Array<Record<string, unknown>>;
  serial_acquirer_profile: Record<string, unknown> | null;
};

function validPackage(): MutableTestPackage {
  return {
    schema_version: "0.1",
    run_id: runId,
    issuer_id: issuerId,
    security_id: null,
    dossier_id: null,
    data_cutoff: "2026-10-01",
    sources: [{
      source_id: "S-001",
      title: "Annual report",
      publisher: "Example plc",
      source_type: "ISSUER_PRIMARY",
      source_date: "2026-03-01",
      as_of_date: "2025-12-31",
      locator: "https://example.com/ar",
      root_source_id: null,
      access_status: "AVAILABLE",
      content_sha256: null,
      limitations: [],
    }],
    evidence: [{
      evidence_id: "E-001",
      source_id: "S-001",
      claim: "Recurring revenue is material.",
      epistemic_type: "FACT",
      evidence_grade: "E1 OBSERVABLE_ISSUER_EVIDENCE",
      polarity: "SUPPORTING",
      block_relevance: ["BUSINESS_MODEL"],
      independence: "ISSUER",
      freshness: "CURRENT_AT_CUTOFF",
      limitations: [],
      numeric_data: [{
        metric: "Recurring revenue share",
        value_kind: "SCALAR",
        value: 70,
        unit: "PERCENT",
        currency: null,
        period_start: "2025-01-01",
        period_end: "2025-12-31",
        as_of_date: null,
        accounting_basis: "Reported",
        transformation: null,
        calculation_id: null,
      }],
    }],
    conflicts: [],
    gaps: [],
    assumptions: [],
    company_economic_dna: {
      revenue_engine: dnaDimension("Recurring subscription revenue.", ["E-001"]),
      customer_structure: dnaDimension(),
      supplier_structure: dnaDimension(),
      cost_structure: dnaDimension(),
      capital_intensity: dnaDimension(),
      reinvestment_model: dnaDimension(),
      pricing_power_mechanism: dnaDimension(),
      switching_cost_mechanism: dnaDimension(),
      network_effects: dnaDimension(),
      scale_effects: dnaDimension(),
      regulation_dependency: dnaDimension(),
      technology_dependency: dnaDimension(),
      cyclicality_exposure: dnaDimension(),
      acquisition_dependency: dnaDimension(),
      geographic_exposure: dnaDimension(),
    },
    analytical_blocks: [{
      block: "BUSINESS_MODEL",
      status: "COMPLETE",
      conclusion: "Recurring model established.",
      supporting_evidence_ids: ["E-001"],
      counterevidence_ids: [],
      conflict_ids: [],
      gap_ids: [],
      assumption_ids: [],
      causal_links: [{
        link_id: "CL-001",
        from_node: "Recurring contracts",
        to_node: "Revenue visibility",
        mechanism: "Contractual recurrence reduces annual reset risk.",
        status: "SUPPORTED",
        evidence_ids: ["E-001"],
        counterevidence_ids: [],
      }],
      sector_overlays: [{
        overlay: "SOFTWARE_OVERLAY",
        status: "APPLIED",
        rationale: "Software economics are material.",
        evidence_ids: ["E-001"],
      }],
      invalidation_triggers: ["Material decline in recurring share."],
      upstream_block_refs: [],
    }],
    material_changes: [],
    serial_acquirer_profile: null,
  };
}

test("Analytical Data Contracts V2 accepts a coherent package", () => {
  const result = validateAnalyticalDataPackage(validPackage(), context);
  assert.equal(result.ok, true);
});

test("post-cutoff source fails closed", () => {
  const input = validPackage();
  input.sources[0].source_date = "2026-10-02";
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /post-cutoff/);
});

test("unknown Evidence ID cannot enter a block", () => {
  const input = validPackage();
  input.analytical_blocks[0].supporting_evidence_ids = ["E-MISSING"];
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /unknown evidence_id E-MISSING/);
});

test("COMPLETE block cannot coexist with critical unresolved gap", () => {
  const input = validPackage();
  input.gaps.push({
    gap_id: "G-001",
    question: "Critical customer retention evidence missing.",
    affected_block: "BUSINESS_MODEL",
    materiality: "CRITICAL",
    status: "OPEN",
    searches_performed: ["Annual report searched"],
    evidence_ids: [],
    best_next_source: "Customer cohort disclosure",
    why_unresolved: null,
    impact: "Could invalidate revenue durability.",
  });
  input.analytical_blocks[0].gap_ids = ["G-001"];
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /cannot be COMPLETE with a critical unresolved gap/);
});

test("NOT_ASSESSABLE requires traceable exhausted or blocked gap", () => {
  const input = validPackage();
  input.analytical_blocks[0].status = "NOT_ASSESSABLE";
  input.analytical_blocks[0].conclusion = null;
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /NOT_ASSESSABLE requires/);
});

test("material revalidation cannot PASS with an incomplete checklist", () => {
  const input = validPackage();
  input.material_changes.push({
    change_id: "MC-001",
    block: "BUSINESS_MODEL",
    prior_artifact: null,
    reason: "New evidence changes prior conclusion.",
    trigger_evidence_ids: ["E-001"],
    affected_downstream_blocks: ["MOAT"],
    revalidation: {
      reopened_material_evidence: true,
      verified_primary_sources: true,
      searched_disconfirming_evidence: false,
      tested_best_alternative_explanation: true,
      reconciled_downstream_blocks: true,
      recorded_prior_state_change_reason: true,
      status: "PASS",
    },
  });
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /searched_disconfirming_evidence != true/);
});

test("serial acquirer profile requires acquisition economics when applicable", () => {
  const input = validPackage();
  input.serial_acquirer_profile = {
    applicability: "APPLICABLE",
    organic_vs_acquired_growth: null,
    purchase_price_discipline: null,
    integration_model: null,
    goodwill_intangibles_economics: null,
    earnout_contingent_consideration: null,
    dilution_funding: null,
    acquisition_roi: null,
    deployment_runway: null,
    evidence_ids: [],
  };
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.match(result.errors.join("\n"), /serial_acquirer_profile APPLICABLE requires organic_vs_acquired_growth/);
    assert.match(result.errors.join("\n"), /requires evidence_ids/);
  }
});

test("numeric evidence requires a temporal anchor", () => {
  const input = validPackage();
  input.evidence[0].numeric_data[0].period_end = null;
  input.evidence[0].numeric_data[0].as_of_date = null;
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /lacks period_end\/as_of_date/);
});

test("block cannot reference an absent upstream block", () => {
  const input = validPackage();
  input.analytical_blocks[0].upstream_block_refs = ["INDUSTRY_STRUCTURE"];
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /absent upstream block INDUSTRY_STRUCTURE/);
});

test("required missing sector overlay prevents block completion", () => {
  const input = validPackage();
  input.analytical_blocks[0].sector_overlays[0].status = "REQUIRED_MISSING";
  const result = validateAnalyticalDataPackage(input, context);
  assert.equal(result.ok, false);
  if (!result.ok) assert.match(result.errors.join("\n"), /cannot be COMPLETE with REQUIRED_MISSING sector overlay/);
});
