import assert from "node:assert/strict";
import test from "node:test";

import {
  validateAnalyticalDataBundleV2,
  validateMaterialNumericValueV2,
  type AnalyticalDataBundleV2,
} from "../lib/orotitan-equity/vnext/analytical-data-contracts";

const hash = "a".repeat(64);

function lock(stage: "RESEARCH" | "DEEP_DIVE") {
  return {
    run_id: "run-1",
    stage,
    stage_revision: 1,
    issuer_id: "issuer-1",
    security_id: "security-1",
    dossier_id: "dossier-1",
    run_type: "INITIAL",
    canonical_mode: "ANALYZE",
    data_cutoff: "2026-10-01",
    contract_set_sha256: hash,
  };
}

function validBundle(): AnalyticalDataBundleV2 {
  return {
    evidenceLedger: {
      contract_name: "OROTITAN_EVIDENCE_LEDGER_V2",
      schema_version: "2.0.0-draft",
      run_lock: lock("DEEP_DIVE"),
      generated_at: "2026-10-01T12:00:00Z",
      method_version: "v2-draft",
      artifact_role: "EVIDENCE_LEDGER",
      sources: [
        {
          source_id: "SRC-1",
          title: "Customer primary source",
          publisher: "Customer A",
          source_class: "CUSTOMER_PRIMARY",
          source_date: "2026-09-30",
          data_period: "FY2026",
          url_or_locator: "https://example.test/customer-a",
          root_source_id: "NOT_APPLICABLE",
          retrieved_at: "2026-10-01T11:00:00Z",
          access_status: "AVAILABLE",
          limitations: [],
        },
      ],
      evidence_items: [
        {
          evidence_id: "EVD-1",
          source_id: "SRC-1",
          claim_summary: "Customer behavior supports meaningful switching friction.",
          epistemic_type: "FACT",
          evidence_role: "CUSTOMER_EVIDENCE",
          polarity: "SUPPORTS",
          materiality: "MATERIAL",
          affected_blocks: ["MOAT"],
          source_locator: "section 2",
          source_date: "2026-09-30",
          data_period: "FY2026",
          admission_status: "ADMITTED",
          limitations: [],
        },
      ],
    },
    conflictLedger: {
      contract_name: "OROTITAN_CONFLICT_LEDGER_V2",
      schema_version: "2.0.0-draft",
      run_lock: lock("DEEP_DIVE"),
      generated_at: "2026-10-01T12:00:00Z",
      method_version: "v2-draft",
      artifact_role: "CONFLICT_LEDGER",
      conflicts: [],
    },
    researchGapRegister: {
      contract_name: "OROTITAN_RESEARCH_GAP_REGISTER_V2",
      schema_version: "2.0.0-draft",
      run_lock: lock("RESEARCH"),
      generated_at: "2026-10-01T10:00:00Z",
      method_version: "v2-draft",
      artifact_role: "RESEARCH_GAP_REGISTER",
      questions: [],
    },
    companyEconomicDna: {
      contract_name: "OROTITAN_COMPANY_ECONOMIC_DNA_V2",
      schema_version: "2.0.0-draft",
      run_lock: lock("RESEARCH"),
      generated_at: "2026-10-01T10:00:00Z",
      method_version: "v2-draft",
      artifact_role: "COMPANY_ECONOMIC_DNA",
      dna: {
        primary_sector: "Software",
        subsectors: ["Vertical software"],
        business_model_archetypes: ["SOFTWARE"],
        revenue_models: ["Subscription"],
        pricing_models: ["Per-seat"],
        demand_drivers: ["Digitization"],
        cost_structure: "High gross margin, R&D and sales intensive",
        capital_intensity: "Asset-light",
        working_capital_profile: "Negative to neutral working capital",
        fixed_cost_intensity: "Moderate",
        reinvestment_model: "R&D and go-to-market",
        organic_vs_acquired_growth: "Primarily organic",
        cyclicality_profile: { material: false, description: "Low direct cyclicality" },
        technology_exposure: { material: true, description: "Software architecture is economically material" },
        regulatory_exposure: { material: false, description: "Limited" },
        geographic_exposure: "Global",
        customer_concentration: "Low",
        supplier_concentration: "Low",
        distribution_model: "Direct",
        network_effect_exposure: { material: false, description: "NOT_APPLICABLE" },
        installed_base_exposure: { material: true, description: "Large recurring installed base" },
        intangible_intensity: "High",
        ma_dependence: { material: false, description: "Low" },
        commodity_exposure: { material: false, description: "NOT_APPLICABLE" },
        financial_leverage_model: "Low leverage",
        key_economic_bottlenecks: ["Product relevance"],
        key_value_drivers: ["Retention", "Pricing"],
        key_failure_modes: ["Product obsolescence"],
      },
      field_provenance: [
        {
          field_path: "/dna/capital_intensity",
          supporting_evidence_ids: ["EVD-1"],
          contradicting_evidence_ids: [],
          assumption_ids: [],
        },
      ],
      overlay_activations: [
        {
          overlay_id: "OVL-1",
          overlay_version: "0.1",
          overlay_type: "SOFTWARE",
          activation_reason: "Subscription software economics are material.",
          supporting_claim_ids: [],
          status: "ACTIVE",
        },
      ],
    },
    analyticalBlockOutputs: {
      contract_name: "OROTITAN_ANALYTICAL_BLOCK_OUTPUTS_V2",
      schema_version: "2.0.0-draft",
      run_lock: lock("DEEP_DIVE"),
      generated_at: "2026-10-01T12:00:00Z",
      method_version: "v2-draft",
      artifact_role: "ANALYTICAL_BLOCK_OUTPUTS",
      claims: [
        {
          claim_id: "CLM-1",
          block: "MOAT",
          claim_type: "CAUSAL",
          statement: "Observed customer switching friction contributes to retention durability.",
          materiality: "MATERIAL",
          supporting_evidence_ids: ["EVD-1"],
          contradicting_evidence_ids: [],
          calculation_ids: [],
          assumption_ids: [],
          conflict_ids: [],
          alternative_explanations: ["Contract duration rather than true switching friction."],
          invalidation_trigger_ids: ["INV-1"],
          status: "SUPPORTED",
        },
      ],
      causal_nodes: [
        {
          node_id: "CAU-N-1",
          node_type: "CLAIM",
          label: "Switching friction",
          claim_id: "CLM-1",
          evidence_ids: ["EVD-1"],
        },
        {
          node_id: "CAU-N-2",
          node_type: "OUTCOME",
          label: "Retention durability",
          claim_id: "NOT_APPLICABLE",
          evidence_ids: [],
        },
      ],
      causal_edges: [
        {
          edge_id: "CAU-E-1",
          from_node_id: "CAU-N-1",
          to_node_id: "CAU-N-2",
          mechanism: "Operational switching costs reduce churn.",
          supporting_evidence_ids: ["EVD-1"],
          counterevidence_ids: [],
          status: "SUPPORTED",
        },
      ],
      invalidation_triggers: [
        {
          trigger_id: "INV-1",
          statement: "Material increase in customer churn would weaken the switching-friction claim.",
          affected_claim_ids: ["CLM-1"],
          observable_signal: "Customer churn",
          threshold_or_condition: "Sustained material deterioration",
          evidence_ids: ["EVD-1"],
          status: "ACTIVE",
        },
      ],
      block_outputs: [
        {
          block_output_id: "BLOCK-1",
          block: "MOAT",
          module_type: "MOAT",
          module_schema_version: "0.1",
          status: "COMPLETE",
          summary: "Switching friction has traceable customer evidence.",
          material_claim_ids: ["CLM-1"],
          evidence_ids: ["EVD-1"],
          calculation_ids: [],
          assumption_ids: [],
          conflict_ids: [],
          open_question_ids: [],
          causal_node_ids: ["CAU-N-1", "CAU-N-2"],
          causal_edge_ids: ["CAU-E-1"],
          invalidation_trigger_ids: ["INV-1"],
          dependencies: ["TECHNOLOGY"],
          reopened_dependencies: [],
          module_payload: {},
        },
      ],
    },
  };
}

test("valid V2 analytical bundle passes structural and cross-artifact validation", () => {
  const result = validateAnalyticalDataBundleV2(validBundle());
  assert.equal(result.valid, true, result.errors.join("\n"));
});

test("post-cutoff admitted evidence fails closed", () => {
  const bundle = validBundle();
  const items = bundle.evidenceLedger.evidence_items as Array<Record<string, unknown>>;
  items[0].source_date = "2026-10-02";
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /exceeds DATA_CUTOFF/);
});

test("unresolved Evidence ID is rejected", () => {
  const bundle = validBundle();
  const claims = bundle.analyticalBlockOutputs.claims as Array<Record<string, unknown>>;
  claims[0].supporting_evidence_ids = ["EVD-DOES-NOT-EXIST"];
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /unresolved evidence_id/);
});

test("assumption cannot masquerade as supporting evidence", () => {
  const bundle = validBundle();
  const items = bundle.evidenceLedger.evidence_items as Array<Record<string, unknown>>;
  items[0].evidence_role = "ASSUMPTION";
  items[0].epistemic_type = "ASSUMPTION";
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /cannot masquerade as supporting evidence/);
});

test("material SUPPORTED claim requires evidence or calculation lineage", () => {
  const bundle = validBundle();
  const claims = bundle.analyticalBlockOutputs.claims as Array<Record<string, unknown>>;
  claims[0].supporting_evidence_ids = [];
  const outputs = bundle.analyticalBlockOutputs.block_outputs as Array<Record<string, unknown>>;
  outputs[0].evidence_ids = [];
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /requires evidence or calculation lineage/);
});

test("open material conflict prevents a SUPPORTED claim", () => {
  const bundle = validBundle();
  const conflicts = bundle.conflictLedger.conflicts as Array<Record<string, unknown>>;
  conflicts.push({
    conflict_id: "CFL-1",
    question_or_claim: "Switching friction mechanism",
    conflict_type: "INTERPRETIVE",
    side_a_evidence_ids: ["EVD-1"],
    side_b_evidence_ids: ["EVD-1"],
    affected_blocks: ["MOAT"],
    materiality: "MATERIAL",
    status: "OPEN",
    resolution: "UNKNOWN",
    resolution_evidence_ids: [],
    downstream_impact: "MOAT conclusion",
  });
  const claims = bundle.analyticalBlockOutputs.claims as Array<Record<string, unknown>>;
  claims[0].conflict_ids = ["CFL-1"];
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /cannot be SUPPORTED while material conflict CFL-1 is OPEN/);
});

test("COMPLETE block cannot retain an OPEN question", () => {
  const bundle = validBundle();
  const questions = bundle.researchGapRegister.questions as Array<Record<string, unknown>>;
  questions.push({
    question_id: "GAP-1",
    block: "MOAT",
    question: "Is retention durable through a downturn?",
    why_material: "Durability affects moat classification.",
    current_evidence_ids: ["EVD-1"],
    missing_evidence: "Downturn cohort evidence",
    searches_already_performed: ["Customer primary evidence"],
    best_next_source: "Historical customer cohorts",
    status: "OPEN",
    impact_if_unresolved: "Limits durability conclusion",
    last_attempt_fingerprint: "b".repeat(64),
  });
  const outputs = bundle.analyticalBlockOutputs.block_outputs as Array<Record<string, unknown>>;
  outputs[0].open_question_ids = ["GAP-1"];
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /COMPLETE block cannot retain OPEN question/);
});

test("cross-artifact run-lock drift is rejected", () => {
  const bundle = validBundle();
  const lockRecord = bundle.conflictLedger.run_lock as Record<string, unknown>;
  lockRecord.data_cutoff = "2026-09-30";
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /run lock mismatch/);
});

test("French UI labels cannot leak into canonical block enums", () => {
  const bundle = validBundle();
  const outputs = bundle.analyticalBlockOutputs.block_outputs as Array<Record<string, unknown>>;
  outputs[0].block = "Cyclicité";
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /schema error|must be equal to one of the allowed values/);
});

test("duplicate evidence IDs fail closed", () => {
  const bundle = validBundle();
  const items = bundle.evidenceLedger.evidence_items as Array<Record<string, unknown>>;
  items.push(structuredClone(items[0]));
  const result = validateAnalyticalDataBundleV2(bundle);
  assert.equal(result.valid, false);
  assert.match(result.errors.join("\n"), /duplicate evidence_id/);
});

test("material monetary flow requires currency, period and accounting basis", () => {
  const invalid = validateMaterialNumericValueV2({
    value_kind: "MONETARY_FLOW",
    value: 100,
    unit: "million",
    source_evidence_ids: ["EVD-1"],
  });
  assert.equal(invalid.valid, false);

  const valid = validateMaterialNumericValueV2({
    value_kind: "MONETARY_FLOW",
    value: 100,
    unit: "million",
    currency: "EUR",
    period_start: "2026-01-01",
    period_end: "2026-12-31",
    accounting_basis: "IFRS",
    source_evidence_ids: ["EVD-1"],
  });
  assert.equal(valid.valid, true, valid.errors.join("\n"));
});
