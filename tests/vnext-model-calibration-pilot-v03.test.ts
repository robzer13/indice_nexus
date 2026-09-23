import assert from "node:assert/strict";
import test from "node:test";

import {
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V03_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V03_MODULE_ID,
  GATE18_PHASE_B_V03_SYSTEM_PROMPT,
  assertGate18PhaseBV03Semantics,
  gate18PhaseBV03OutputSchema,
  type Gate18PhaseBV03Output,
} from "../runtime/vnext/model-calibration-pilot-v03";
import type {
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "case-1",
    display_name: "Synthetic",
    role: "SYNTHETIC",
    source_run_id: "case-1",
    data_cutoff: "2026-09-19",
    source_integrity: {
      evidence_ledger_sha256: "a".repeat(64),
      conflict_ledger_sha256: "b".repeat(64),
    },
    evidence_items: [
      {
        evidence_id: "E-001",
        claim_id: "CL-001",
        claim: "Synthetic",
        value: "Synthetic evidence",
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
    ],
    conflicts: [
      {
        conflict_id: "C-001",
        metric_claim: "Synthetic conflict",
        value_a: "A",
        value_b: "B",
        conflict_type: "SYNTHETIC",
        reason: "Synthetic",
        resolution: "UNRESOLVED",
        resolution_note: "Carry forward",
        materiality: "HIGH",
        evidence_refs: ["E-001"],
        affected_outputs: ["MOAT_INPUTS"],
      },
    ],
  };
}

function validOutput(): Gate18PhaseBV03Output {
  return {
    case_id: "case-1",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim: "Synthetic evidence supports a bounded observation.",
        support_state: "SUPPORTED",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        causal_link: "The cited evidence directly supports the stated observation.",
        counterevidence_ids: [],
      },
    ],
    material_conflicts: [
      {
        conflict_id: "C-001",
        implication: "The conflict prevents a stronger inference.",
        resolution_state: "UNRESOLVED_IN_PACKET",
      },
    ],
    weak_link_candidates: [
      {
        candidate: "Synthetic weak link",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        why_uncertain: "The packet does not independently resolve the underlying uncertainty.",
      },
    ],
    unresolved_points: [
      {
        question: "What evidence would resolve the remaining uncertainty?",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
      },
    ],
  };
}

test("Gate 18 v0.3 keeps the provider cap while adding narrative headroom", () => {
  assert.equal(GATE18_PHASE_B_V03_MAX_OUTPUT_TOKENS, 1536);
  assert.equal(
    GATE18_PHASE_B_V03_MODULE_ID,
    "MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3",
  );
  assert.equal(
    GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_3",
  );
  assert.match(
    GATE18_PHASE_B_V03_SYSTEM_PROMPT,
    /complete and self-contained/,
  );
  assert.match(
    GATE18_PHASE_B_V03_SYSTEM_PROMPT,
    /Target at most 120 characters/,
  );

  const output = validOutput();
  output.priority_findings[0].claim =
    `${"x".repeat(170)}.`;

  assert.doesNotThrow(() =>
    gate18PhaseBV03OutputSchema.parse(output),
  );
});

test("Gate 18 v0.3 semantic validation accepts complete grounded narratives", () => {
  assert.doesNotThrow(() =>
    assertGate18PhaseBV03Semantics(
      packet(),
      validOutput(),
    ),
  );
});

test("Gate 18 v0.3 rejects incomplete narrative fields", () => {
  const output = validOutput();
  output.priority_findings[0].claim =
    "This claim ends mid-thought because";

  assert.throws(
    () =>
      assertGate18PhaseBV03Semantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V03_FINDING_CLAIM_INCOMPLETE/,
  );
});

test("Gate 18 v0.3 rejects narrative boundary saturation", () => {
  const output = validOutput();
  output.priority_findings[0].claim =
    `${"x".repeat(177)}.`;

  assert.throws(
    () =>
      assertGate18PhaseBV03Semantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V03_NARRATIVE_BOUNDARY_SATURATION/,
  );
});

test("Gate 18 v0.3 still rejects unknown evidence references", () => {
  const output = validOutput();
  output.priority_findings[0].evidence_ids = ["E-999"];

  assert.throws(
    () =>
      assertGate18PhaseBV03Semantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V03_UNKNOWN_EVIDENCE_REF/,
  );
});
