import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type {
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V07_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V07_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V07_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V07_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V07_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V07_SYSTEM_PROMPT,
  assertGate18PhaseBV07Semantics,
  gate18PhaseBV07OutputSchema,
  type Gate18PhaseBV07Output,
} from "../runtime/vnext/model-calibration-pilot-v07";

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "case-v07",
    display_name: "Synthetic",
    role: "SYNTHETIC",
    source_run_id: "case-v07",
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
        claim_id: "QUALIFICATION",
        claim: "Selected support",
        value: "Issuer-hosted selected case supports benefit.",
        period: "FY2025",
        source_refs: ["SRC-002"],
        source_class: "S2",
        claim_fit: "MEDIUM",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT",
        limitations: "Selected case; not representative.",
        conflict_status: null,
        module_tags: ["MOAT_INPUTS"],
      },
      {
        evidence_id: "E-003",
        claim_id: "COUNTER",
        claim: "Counter",
        value: "Direct opposing mechanism.",
        period: "FY2025",
        source_refs: ["SRC-003"],
        source_class: "S1",
        claim_fit: "HIGH",
        epistemic_type: "REPORTED",
        freshness_state: "CURRENT",
        limitations: null,
        conflict_status: "C-001",
        module_tags: ["MOAT_INPUTS"],
      },
      {
        evidence_id: "E-004",
        claim_id: "OTHER",
        claim: "Other",
        value: "Other evidence.",
        period: "FY2025",
        source_refs: ["SRC-004"],
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
        evidence_refs: ["E-001", "E-003"],
        affected_outputs: ["MOAT_INPUTS"],
      },
    ],
  };
}

function validOutput(): Gate18PhaseBV07Output {
  return {
    case_id: "case-v07",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim: "The packet supports a bounded benefit claim.",
        support_state: "MIXED",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        causal_link: "E-001 directly supports the bounded benefit claim.",
        qualification_evidence_ids: ["E-002"],
        qualification_link:
          "E-002 supports the benefit but limits generalization because the case is selected and non-representative.",
        counterevidence_ids: ["E-003"],
        counterevidence_link:
          "E-003 weakens the claim by documenting an opposing mechanism.",
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
        candidate: "The mechanism remains uncertain.",
        evidence_ids: ["E-001", "E-003"],
        conflict_ids: ["C-001"],
        why_uncertain:
          "The packet contains directly opposing mechanism evidence.",
      },
    ],
    unresolved_points: [
      {
        question: "Which mechanism dominates economically?",
        evidence_ids: ["E-001", "E-003"],
        conflict_ids: ["C-001"],
      },
    ],
  };
}

test("Gate 18 v0.7 adds a distinct qualification channel without widening scope", () => {
  assert.equal(GATE18_PHASE_B_PROTOCOL_VERSION, "0.7");
  assert.equal(GATE18_PHASE_B_V07_MAX_OUTPUT_TOKENS, 4096);
  assert.equal(
    GATE18_PHASE_B_V07_PROMPT_TEMPLATE_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_V0_5",
  );
  assert.equal(
    GATE18_PHASE_B_V07_PROMPT_TEMPLATE_VERSION,
    "0.5",
  );
  assert.equal(
    GATE18_PHASE_B_V07_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_5",
  );
  assert.equal(
    GATE18_PHASE_B_V07_GENERATION_SCHEMA_VERSION,
    "0.5",
  );
  assert.match(
    GATE18_PHASE_B_V07_SYSTEM_PROMPT,
    /Qualification is not opposition/,
  );
  assert.match(
    GATE18_PHASE_B_V07_SYSTEM_PROMPT,
    /Issuer-hosted, selected, unaudited, non-representative/,
  );
  assert.match(
    GATE18_PHASE_B_V07_SYSTEM_PROMPT,
    /must not be placed in counterevidence_ids/,
  );
});

test("Gate 18 v0.7 accepts separated support, qualification, and counterevidence", () => {
  const output = gate18PhaseBV07OutputSchema.parse(
    validOutput(),
  );
  assert.doesNotThrow(() =>
    assertGate18PhaseBV07Semantics(packet(), output),
  );
});

test("Gate 18 v0.7 rejects malformed canonical evidence IDs", () => {
  const output = validOutput() as unknown as Record<string, unknown>;
  const findings = output.priority_findings as Array<Record<string, unknown>>;
  findings[0].qualification_evidence_ids = ["E-002','E-003"];

  const parsed = gate18PhaseBV07OutputSchema.safeParse(output);
  assert.equal(parsed.success, false);
});

test("Gate 18 v0.7 rejects support and qualification overlap", () => {
  const output = validOutput();
  output.priority_findings[0].qualification_evidence_ids = [
    "E-001",
  ];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V07_SUPPORT_QUALIFICATION_OVERLAP/,
  );
});

test("Gate 18 v0.7 rejects qualification and counterevidence overlap", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [
    "E-002",
  ];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V07_EVIDENCE_ROLE_OVERLAP/,
  );
});

test("Gate 18 v0.7 requires qualification explanation when qualification evidence exists", () => {
  const output = validOutput();
  output.priority_findings[0].qualification_link = null;

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V07_QUALIFICATION_LINK_REQUIRED/,
  );
});

test("Gate 18 v0.7 rejects orphan qualification explanations", () => {
  const output = validOutput();
  output.priority_findings[0].qualification_evidence_ids = [];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V07_QUALIFICATION_LINK_WITHOUT_IDS/,
  );
});

test("Gate 18 v0.7 does not treat qualification alone as MIXED opposition", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].conflict_ids = [];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V07_MIXED_REQUIRES_CHALLENGE/,
  );
});

test("Gate 18 v0.7 permits supported findings with qualification but no opposition", () => {
  const output = validOutput();
  output.priority_findings[0].support_state = "SUPPORTED";
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].conflict_ids = [];

  assert.doesNotThrow(() =>
    assertGate18PhaseBV07Semantics(packet(), output),
  );
});

test("Gate 18 v0.7 requires conflicts to touch any finding role evidence", () => {
  const output = validOutput();
  output.priority_findings[0].evidence_ids = ["E-004"];
  output.priority_findings[0].qualification_evidence_ids = ["E-002"];
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V07_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS/,
  );
});

test("Gate 18 v0.7 invariants remain while the runner advances to v0.9", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(source, /model-calibration-pilot-v09/);
  assert.match(source, /gate18PhaseBV09OutputSchema/);
  assert.match(source, /assertGate18PhaseBV09Semantics/);
  assert.match(
    source,
    /GATE18_PHASE_B_V09_MAX_OUTPUT_TOKENS/,
  );
  assert.doesNotMatch(
    source,
    /model-calibration-pilot-v06/,
  );
});
