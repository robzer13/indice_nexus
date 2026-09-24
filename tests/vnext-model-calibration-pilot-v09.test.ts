import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type {
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V09_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V09_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V09_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V09_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V09_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V09_SYSTEM_PROMPT,
  assertGate18PhaseBV09Semantics,
  gate18PhaseBV09OutputSchema,
  type Gate18PhaseBV09Output,
} from "../runtime/vnext/model-calibration-pilot-v09";

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "case-v09",
    display_name: "Synthetic",
    role: "SYNTHETIC",
    source_run_id: "case-v09",
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
        limitations: "Selected disclosure.",
        conflict_status: "C-001",
        module_tags: ["MOAT_INPUTS"],
      },
      {
        evidence_id: "E-002",
        claim_id: "COUNTER",
        claim: "Counter",
        value: "Direct opposing mechanism.",
        period: "FY2025",
        source_refs: ["SRC-002"],
        source_class: "S1",
        claim_fit: "HIGH",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT",
        limitations: "Coverage is incomplete.",
        conflict_status: "C-001",
        module_tags: ["MOAT_INPUTS"],
      },
      {
        evidence_id: "E-003",
        claim_id: "OTHER",
        claim: "Other",
        value: "Other evidence.",
        period: "FY2025",
        source_refs: ["SRC-003"],
        source_class: "S1",
        claim_fit: "HIGH",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT",
        limitations: null,
        conflict_status: null,
        module_tags: ["MOAT_INPUTS"],
      },
    ],
    conflicts: [
      {
        conflict_id: "C-001",
        metric_claim: "Synthetic mechanism conflict",
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

function validOutput(): Gate18PhaseBV09Output {
  return {
    case_id: "case-v09",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim:
          "Integration and reported retention indicate relationship persistence.",
        support_state: "MIXED",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        causal_link:
          "E-001 directly supports the persistence proposition.",
        evidence_qualifications: [
          {
            evidence_id: "E-001",
            qualification:
              "E-001 supports the claim but comes from a selected disclosure.",
          },
        ],
        counterevidence_ids: ["E-002"],
        counterevidence_link:
          "E-002 weakens persistence by documenting an opposing mechanism.",
      },
    ],
    material_conflicts: [
      {
        conflict_id: "C-001",
        implication:
          "The opposing mechanism prevents a stronger persistence inference.",
        resolution_state: "UNRESOLVED_IN_PACKET",
      },
    ],
    weak_link_candidates: [
      {
        candidate: "Persistence from integration",
        evidence_ids: ["E-001", "E-002"],
        conflict_ids: ["C-001"],
        why_uncertain:
          "The packet contains directly opposing mechanism evidence.",
      },
    ],
    unresolved_points: [
      {
        question: "Which mechanism dominates economically?",
        evidence_ids: ["E-001", "E-002"],
        conflict_ids: ["C-001"],
      },
    ],
  };
}

test("Gate 18 v0.9 changes semantic prompt while preserving the v0.6 JSON shape", () => {
  assert.equal(GATE18_PHASE_B_PROTOCOL_VERSION, "0.9");
  assert.equal(GATE18_PHASE_B_V09_MAX_OUTPUT_TOKENS, 4096);
  assert.equal(
    GATE18_PHASE_B_V09_PROMPT_TEMPLATE_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_V0_7",
  );
  assert.equal(
    GATE18_PHASE_B_V09_PROMPT_TEMPLATE_VERSION,
    "0.7",
  );
  assert.equal(
    GATE18_PHASE_B_V09_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6",
  );
  assert.equal(
    GATE18_PHASE_B_V09_GENERATION_SCHEMA_VERSION,
    "0.6",
  );
  assert.match(
    GATE18_PHASE_B_V09_SYSTEM_PROMPT,
    /claim must state one atomic directional proposition only/,
  );
  assert.match(
    GATE18_PHASE_B_V09_SYSTEM_PROMPT,
    /keep it outside claim wording/,
  );
  assert.match(
    GATE18_PHASE_B_V09_SYSTEM_PROMPT,
    /weak_link_candidates\.candidate is a short label/,
  );
});

test("Gate 18 v0.9 accepts an atomic claim with separate counterevidence", () => {
  const output = gate18PhaseBV09OutputSchema.parse(
    validOutput(),
  );
  assert.doesNotThrow(() =>
    assertGate18PhaseBV09Semantics(packet(), output),
  );
});

test("Gate 18 v0.9 rejects the v0.8 coexistence-style compound claim", () => {
  const output = validOutput();
  output.priority_findings[0].claim =
    "Retention and integration indicators coexist with customers' ability to shift payment volume.";

  assert.throws(
    () => assertGate18PhaseBV09Semantics(packet(), output),
    /VNEXT_GATE18_V09_NON_ATOMIC_CONTRASTIVE_CLAIM/,
  );
});

test("Gate 18 v0.9 rejects the v0.8 despite-style compound claim", () => {
  const output = validOutput();
  output.priority_findings[0].claim =
    "Adyen demonstrates competitive displacement despite a multi-provider market structure.";

  assert.throws(
    () => assertGate18PhaseBV09Semantics(packet(), output),
    /VNEXT_GATE18_V09_NON_ATOMIC_CONTRASTIVE_CLAIM/,
  );
});

for (const [word, claim] of [
  ["but", "Integration indicates persistence but customers can switch."],
  ["although", "Integration indicates persistence although customers can switch."],
  ["while", "Integration indicates persistence while customers can switch."],
  ["whereas", "Integration indicates persistence whereas customers can switch."],
  ["yet", "Integration indicates persistence yet customers can switch."],
  ["however", "Integration indicates persistence however customers can switch."],
] as const) {
  test(`Gate 18 v0.9 rejects contrastive atomicity breaker: ${word}`, () => {
    const output = validOutput();
    output.priority_findings[0].claim = claim;

    assert.throws(
      () => assertGate18PhaseBV09Semantics(packet(), output),
      /VNEXT_GATE18_V09_NON_ATOMIC_CONTRASTIVE_CLAIM/,
    );
  });
}

test("Gate 18 v0.9 permits a weak-link label without terminal punctuation", () => {
  const output = validOutput();
  output.weak_link_candidates[0].candidate =
    "Switching-friction inference from integration";

  assert.doesNotThrow(() =>
    assertGate18PhaseBV09Semantics(packet(), output),
  );
});

test("Gate 18 v0.9 still requires weak-link explanation to be a complete sentence", () => {
  const output = validOutput();
  output.weak_link_candidates[0].why_uncertain =
    "The packet contains directly opposing evidence";

  assert.throws(
    () => assertGate18PhaseBV09Semantics(packet(), output),
    /VNEXT_GATE18_V09_WEAK_LINK_EXPLANATION_INCOMPLETE/,
  );
});

test("Gate 18 v0.9 preserves orthogonal qualification on supporting evidence", () => {
  const output = validOutput();
  output.priority_findings[0].support_state = "SUPPORTED";
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].conflict_ids = [];
  output.priority_findings[0].evidence_qualifications = [
    {
      evidence_id: "E-001",
      qualification:
        "E-001 supports the proposition but does not quantify economic weight.",
    },
  ];

  assert.doesNotThrow(() =>
    assertGate18PhaseBV09Semantics(packet(), output),
  );
});

test("Gate 18 v0.9 still rejects support and counterevidence overlap", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [
    "E-001",
  ];

  assert.throws(
    () => assertGate18PhaseBV09Semantics(packet(), output),
    /VNEXT_GATE18_V09_DIRECTION_ROLE_OVERLAP/,
  );
});

test("Gate 18 v0.9 invariants remain while the runner advances to v1.0", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(source, /model-calibration-pilot-v10/);
  assert.match(source, /gate18PhaseBV10OutputSchema/);
  assert.match(source, /assertGate18PhaseBV10Semantics/);
  assert.match(
    source,
    /GATE18_PHASE_B_V10_MAX_OUTPUT_TOKENS/,
  );
  assert.doesNotMatch(
    source,
    /model-calibration-pilot-v09/,
  );
});
