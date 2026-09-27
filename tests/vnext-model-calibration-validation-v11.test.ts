import assert from "node:assert/strict";
import test from "node:test";

import type {
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";
import type {
  Gate18PhaseBV10Output,
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  GATE18_V11_SAFE_NARRATIVE_BOUNDARY_EXCLUSIVE,
  GATE18_V11_VALIDATION_CONTRACT_ID,
  GATE18_V11_VALIDATION_CONTRACT_VERSION,
  evaluateGate18V11Validation,
  gate18V11ValidationContractSha256,
  inspectGate18V11Presentation,
  normalizeGate18V11Presentation,
} from "../runtime/vnext/model-calibration-validation-v11";

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "case-v11",
    display_name: "Synthetic",
    role: "SYNTHETIC",
    source_run_id: "case-v11",
    data_cutoff: "2026-09-19",
    source_integrity: {
      evidence_ledger_sha256: "a".repeat(64),
      conflict_ledger_sha256: "b".repeat(64),
    },
    evidence_items: [
      {
        evidence_id: "E-001",
        claim_id: "SUPPORT",
        claim: "Support",
        value: "Direct support.",
        period: "FY2025",
        source_refs: ["SRC-001"],
        source_class: "S1",
        claim_fit: "HIGH",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT",
        limitations: null,
        conflict_status: "C-001",
        module_tags: ["MOAT_INPUTS"],
      },
      {
        evidence_id: "E-002",
        claim_id: "COUNTER",
        claim: "Counter",
        value: "Direct counterevidence.",
        period: "FY2025",
        source_refs: ["SRC-002"],
        source_class: "S1",
        claim_fit: "HIGH",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT",
        limitations: null,
        conflict_status: "C-001",
        module_tags: ["MOAT_INPUTS"],
      },
    ],
    conflicts: [
      {
        conflict_id: "C-001",
        metric_claim: "Synthetic conflict",
        value_a: "Support",
        value_b: "Counter",
        conflict_type: "POLARITY",
        reason: "Synthetic.",
        resolution: "UNRESOLVED",
        resolution_note: "Carry forward.",
        materiality: "HIGH",
        evidence_refs: ["E-001", "E-002"],
        affected_outputs: ["MOAT_INPUTS"],
      },
    ],
  };
}

function validOutput(): Gate18PhaseBV10Output {
  return {
    case_id: "case-v11",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim: "Evidence indicates meaningful relationship persistence.",
        support_state: "MIXED",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        causal_link: "E-001 directly supports relationship persistence.",
        evidence_qualifications: [],
        counterevidence_ids: ["E-002"],
        counterevidence_link:
          "E-002 weakens the relationship-persistence proposition.",
      },
    ],
    material_conflicts: [
      {
        conflict_id: "C-001",
        implication:
          "Opposing evidence limits confidence in the persistence inference.",
        resolution_state: "UNRESOLVED_IN_PACKET",
      },
    ],
    weak_link_candidates: [
      {
        candidate: "Relationship persistence",
        evidence_ids: ["E-001", "E-002"],
        conflict_ids: ["C-001"],
        why_uncertain:
          "Support and counterevidence remain directionally opposed.",
      },
    ],
    unresolved_points: [
      {
        question: "How persistent is the observed relationship over time?",
        evidence_ids: ["E-001", "E-002"],
        conflict_ids: ["C-001"],
      },
    ],
  };
}

test("Gate 18 v1.1 validation contract is versioned and hashable", () => {
  assert.equal(
    GATE18_V11_VALIDATION_CONTRACT_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1",
  );
  assert.equal(GATE18_V11_VALIDATION_CONTRACT_VERSION, "1.1");
  assert.equal(
    GATE18_V11_SAFE_NARRATIVE_BOUNDARY_EXCLUSIVE,
    178,
  );
  assert.equal(gate18V11ValidationContractSha256().length, 64);
});

test("v1.1 clears punctuation-only raw noncompliance on a shadow copy", () => {
  const output = validOutput();
  output.priority_findings[0].claim =
    "Evidence indicates meaningful relationship persistence";
  const original = output.priority_findings[0].claim;

  const result = evaluateGate18V11Validation(
    packet(),
    output,
    "FULL",
  );

  assert.equal(result.rawSchemaPass, true);
  assert.equal(
    result.rawPresentationCompliance?.missingTerminalPunctuationCount,
    1,
  );
  assert.deepEqual(result.normalization.normalizedPaths, [
    "priority_findings[0].claim",
  ]);
  assert.equal(
    result.normalizedPresentationCompliance?.compliant,
    true,
  );
  assert.equal(result.substantiveValidation.status, "PASS");
  assert.equal(result.substantiveValidation.pass, true);
  assert.equal(output.priority_findings[0].claim, original);
  assert.equal(result.normalization.rawOutputMutated, false);
});

test("v1.1 preserves substantive failures after safe presentation normalization", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link =
    "This link remains present without counterevidence identifiers";
  output.material_conflicts[0].implication =
    "Opposing evidence limits confidence in the persistence inference";

  const result = evaluateGate18V11Validation(packet(), output);

  assert.equal(
    result.rawPresentationCompliance?.missingTerminalPunctuationCount,
    2,
  );
  assert.equal(
    result.normalizedPresentationCompliance?.compliant,
    true,
  );
  assert.equal(result.substantiveValidation.status, "FAIL");
  assert.equal(
    result.substantiveValidation.error,
    "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS",
  );
});

test("v1.1 never relabels a saturation blocker as a substantive failure", () => {
  const output = validOutput();
  output.priority_findings[0].claim =
    `${"A".repeat(177)}.`;

  const report = inspectGate18V11Presentation(output);
  assert.equal(report.saturationBoundaryCount, 1);

  const result = evaluateGate18V11Validation(packet(), output);
  assert.equal(
    result.substantiveValidation.status,
    "NOT_EVALUATED_PRESENTATION_BLOCKER",
  );
  assert.equal(result.substantiveValidation.pass, null);
  assert.equal(
    result.normalization.blockedPaths.includes(
      "priority_findings[0].claim",
    ),
    true,
  );
});

test("v1.1 refuses punctuation normalization that would hit the 178 boundary", () => {
  const output = validOutput();
  output.priority_findings[0].claim = "A".repeat(177);

  const normalized = normalizeGate18V11Presentation(output);

  assert.deepEqual(normalized.normalizedPaths, []);
  assert.deepEqual(normalized.blockedPaths, [
    "priority_findings[0].claim",
  ]);

  const result = evaluateGate18V11Validation(packet(), output);
  assert.equal(
    result.substantiveValidation.status,
    "NOT_EVALUATED_PRESENTATION_BLOCKER",
  );
});

test("v1.1 does not evaluate presentation or semantics when raw schema fails", () => {
  const output = validOutput() as unknown as Record<string, unknown>;
  output.case_id = "";

  const result = evaluateGate18V11Validation(packet(), output);

  assert.equal(result.rawSchemaPass, false);
  assert.equal(
    result.substantiveValidation.status,
    "NOT_EVALUATED_SCHEMA_FAILURE",
  );
  assert.equal(result.rawPresentationCompliance, null);
});

test("v1.1 leaves an already clean positive control unchanged", () => {
  const output = validOutput();

  const result = evaluateGate18V11Validation(packet(), output);

  assert.equal(result.rawPresentationCompliance?.compliant, true);
  assert.equal(result.normalization.normalizedPathCount, 0);
  assert.equal(result.normalization.blockedPathCount, 0);
  assert.equal(result.substantiveValidation.status, "PASS");
});
