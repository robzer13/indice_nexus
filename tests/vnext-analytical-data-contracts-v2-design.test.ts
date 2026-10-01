import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { validateAnalyticalDataArtifact } from "../lib/orotitan-equity/vnext/analytical-data-contracts";

const common = {
  schema_version: "0.1.0",
  run_id: "run-001",
  issuer_id: "issuer-001",
  security_id: "security-001",
  dossier_id: "dossier-001",
  data_cutoff: "2026-10-01",
  artifact_version: 1,
  created_at: "2026-10-01T18:00:00Z",
};

const traceability = {
  supporting_evidence_ids: ["E-1"],
  contradicting_evidence_ids: [],
  calculation_ids: [],
  material_assumption_ids: [],
  conflict_ids: [],
};

function evidence(sourceDate = "2026-09-30") {
  return {
    evidence_id: "E-1",
    claim_id: "C-1",
    claim_or_metric: "Installed base",
    value_or_statement: "Installed base expanded year over year.",
    period_or_as_of_date: "FY2026",
    source_id: "S-1",
    root_source_id: null,
    independence_group: "ISSUER_PRIMARY",
    source_class: "REGULATORY_FILING",
    claim_fit: "DIRECT",
    source_date: sourceDate,
    data_cutoff: "2026-10-01",
    epistemic_type: "FACT",
    freshness_state: "CURRENT",
    limitations: [],
    conflict_status: "NONE",
    quantitative_provenance: null,
  };
}

test("analytical data schema is an implementation-only additive contract", () => {
  const schema = JSON.parse(
    readFileSync(
      "schemas/vnext/orotitan-analytical-data-contracts-v2.schema.v0.1.json",
      "utf8",
    ),
  );

  assert.equal(schema["x-orotitan"].authority, "IMPLEMENTATION_ONLY");
  assert.equal(schema["x-orotitan"].methodology_change, false);
  assert.equal(schema["x-orotitan"].scoring_authority, false);
  assert.equal(schema["x-orotitan"].valuation_policy_authority, false);
  assert.equal(schema["x-orotitan"].canonical_snapshot_authority, false);
  assert.equal(
    schema["x-orotitan"].ui_language,
    "FRENCH_FIRST_PRESENTATION_ENGLISH_CANONICAL_DATA",
  );
});

test("evidence ledger preserves cutoff and exact frozen evidence fields", () => {
  const artifact = {
    ...common,
    artifact_type: "EVIDENCE_LEDGER",
    body: {
      ledger_version: "1",
      records: [evidence()],
    },
  };

  const result = validateAnalyticalDataArtifact(artifact, {
    sourceIds: new Set(["S-1"]),
  });

  assert.equal(result.ok, true);
});

test("post-cutoff evidence fails semantic validation", () => {
  const artifact = {
    ...common,
    artifact_type: "EVIDENCE_LEDGER",
    body: {
      ledger_version: "1",
      records: [evidence("2026-10-02")],
    },
  };

  const result = validateAnalyticalDataArtifact(artifact, {
    sourceIds: new Set(["S-1"]),
  });

  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.equal(result.stage, "semantic");
    assert.ok(result.errors.some((error) => error.includes("source_date exceeds")));
  }
});

test("unknown Evidence ID is rejected when an authoritative reference index is supplied", () => {
  const artifact = {
    ...common,
    artifact_type: "ANALYTICAL_BLOCK_OUTPUT",
    body: {
      block_id: "MOAT",
      block_version: "1",
      execution_status: "PROVISIONALLY_STABLE",
      execution_confidence: "MEDIUM",
      canonical_verdict_fields: [],
      core_findings: [],
      supporting_evidence_ids: ["E-MISSING"],
      contradicting_evidence_ids: [],
      calculation_ids: [],
      material_assumption_ids: [],
      conflict_ids: [],
      unresolved_points: [],
      material_dependencies: ["BUSINESS_MODEL"],
      reopen_triggers: ["NEW_COMPETITOR_EVIDENCE"],
      rationale: "Provisional pending more external corroboration.",
      last_research_date: "2026-10-01",
    },
  };

  const result = validateAnalyticalDataArtifact(artifact, {
    evidenceIds: new Set(["E-1"]),
  });

  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.ok(result.errors.some((error) => error.includes("E-MISSING")));
  }
});

test("supporting and contradicting overlap cannot pass silently", () => {
  const artifact = {
    ...common,
    artifact_type: "ANALYTICAL_BLOCK_OUTPUT",
    body: {
      block_id: "RUNWAY",
      block_version: "1",
      execution_status: "IN_PROGRESS",
      execution_confidence: "LOW",
      canonical_verdict_fields: [],
      core_findings: [],
      supporting_evidence_ids: ["E-1"],
      contradicting_evidence_ids: ["E-1"],
      calculation_ids: [],
      material_assumption_ids: [],
      conflict_ids: [],
      unresolved_points: ["Same evidence is being interpreted both ways."],
      material_dependencies: ["BUSINESS_MODEL"],
      reopen_triggers: [],
      rationale: "Conflict not yet registered.",
      last_research_date: "2026-10-01",
    },
  };

  const result = validateAnalyticalDataArtifact(artifact);
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.ok(result.errors.some((error) => error.includes("explicit conflict semantics")));
  }
});

test("READY_FOR_DEEP_DIVE fails with an insufficient input block", () => {
  const artifact = {
    ...common,
    artifact_type: "DD_INPUT_SUFFICIENCY_RECORD",
    body: {
      record_version: "1",
      ready_for_deep_dive: true,
      critical_blockers: [],
      blocks: [
        {
          block_id: "MOAT_INPUTS",
          applicability: "APPLICABLE",
          dd_input_status: "INSUFFICIENT",
          mandatory_coverage: "Missing customer-side corroboration.",
          evidence_adequacy: "Insufficient independence.",
          material_blocking_gap_ids: ["G-1"],
          key_evidence_ids: ["E-1"],
          key_conflict_ids: [],
          searches_performed: ["Issuer filing review"],
          best_next_source: "Customer filing",
          rationale: "Cannot support an honest moat test yet.",
        },
      ],
    },
  };

  const result = validateAnalyticalDataArtifact(artifact);
  assert.equal(result.ok, false);
  if (!result.ok) {
    assert.ok(result.errors.some((error) => error.includes("INSUFFICIENT")));
  }
});

test("Company Economic DNA remains causal support and contains no score field", () => {
  const artifact = {
    ...common,
    artifact_type: "COMPANY_ECONOMIC_DNA",
    body: {
      dna_version: "1",
      economic_mechanisms: [
        {
          mechanism_id: "DNA-1",
          category: "PRICING",
          statement: "Pricing is primarily subscription based.",
          traceability,
          limitations: [],
        },
      ],
      material_dependencies: ["CUSTOMER_RETENTION"],
      open_question_ids: [],
    },
  };

  const result = validateAnalyticalDataArtifact(artifact);
  assert.equal(result.ok, true);

  const contaminated = structuredClone(artifact) as any;
  contaminated.body.score = 95;
  const contaminatedResult = validateAnalyticalDataArtifact(contaminated);
  assert.equal(contaminatedResult.ok, false);
  if (!contaminatedResult.ok) assert.equal(contaminatedResult.stage, "schema");
});

test("technology complexity cannot become a hidden moat score", () => {
  const artifact = {
    ...common,
    artifact_type: "TECHNOLOGY_ANALYSIS",
    body: {
      applicability: "APPLICABLE",
      not_applicable_reason: null,
      current_architecture: [],
      technical_bottlenecks: [],
      next_generation_path: [],
      substitution_paths: [],
      replication_difficulty: [],
      supplier_dependence: [],
      customer_dependence: [],
      standards_ecosystem: [],
      rd_economics: [],
      commoditization_risk: [],
      economic_consequence: [],
    },
  };

  assert.equal(validateAnalyticalDataArtifact(artifact).ok, true);

  const contaminated = structuredClone(artifact) as any;
  contaminated.body.moat_score = 100;
  const result = validateAnalyticalDataArtifact(contaminated);
  assert.equal(result.ok, false);
});

test("material conclusion change requires all six revalidation checks", () => {
  const checkNames = [
    "REOPEN_MATERIAL_EVIDENCE",
    "VERIFY_ROOT_PRIMARY_SOURCES",
    "SEARCH_DISCONFIRMING_EVIDENCE",
    "TEST_BEST_ALTERNATIVE_EXPLANATION",
    "RECONCILE_DOWNSTREAM_BLOCKS",
    "RECORD_PRIOR_STATE_CHANGE_REASON",
  ];

  const artifact = {
    ...common,
    artifact_type: "MATERIAL_CHANGE_REVALIDATION_RECORD",
    body: {
      change_id: "MC-1",
      affected_blocks: ["MOAT"],
      prior_state_ref: "BLOCK-MOAT-V1",
      proposed_state: "Updated moat conclusion",
      triggering_evidence_ids: ["E-1"],
      checks: checkNames.map((check) => ({
        check,
        status: "PASS",
        notes: "Completed.",
        evidence_ids: ["E-1"],
      })),
      downstream_reconciled_blocks: ["RUNWAY"],
      outcome: "REVALIDATED",
      rationale: "New primary evidence changes the durable conclusion.",
    },
  };

  assert.equal(
    validateAnalyticalDataArtifact(artifact, {
      evidenceIds: new Set(["E-1"]),
    }).ok,
    true,
  );

  const incomplete = structuredClone(artifact) as any;
  incomplete.body.checks = incomplete.body.checks.slice(0, 5);
  const result = validateAnalyticalDataArtifact(incomplete, {
    evidenceIds: new Set(["E-1"]),
  });
  assert.equal(result.ok, false);
});

test("design document preserves frozen ledgers and French-first presentation boundary", () => {
  const raw = readFileSync(
    "docs/orotitan-equity/OROTITAN_ANALYTICAL_DATA_CONTRACTS_V2_DESIGN_V0.1.md",
    "utf8",
  );

  assert.match(raw, /EXISTING FROZEN LEDGER SEMANTICS/);
  assert.match(raw, /PARALLEL NEW LEDGER SEMANTICS/);
  assert.match(raw, /Machine semantics remain English/);
  assert.match(raw, /OROTITAN USER-FACING LABELS\n= FRENCH-FIRST/);
  assert.match(raw, /TECHNICAL COMPLEXITY\n≠ MOAT/);
  assert.match(raw, /V3 economic-share-count methodology remains a separate global authority/);
  assert.match(raw, /STATUS = DESIGN_CANDIDATE/);
  assert.doesNotMatch(raw, /STATUS = FROZEN/);
});
