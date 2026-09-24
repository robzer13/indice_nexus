import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type {
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V06_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V06_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V06_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V06_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V06_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V06_SYSTEM_PROMPT,
  assertGate18PhaseBV06Semantics,
  gate18PhaseBV06OutputSchema,
  type Gate18PhaseBV06Output,
} from "../runtime/vnext/model-calibration-pilot-v06";

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "case-v06",
    display_name: "Synthetic",
    role: "SYNTHETIC",
    source_run_id: "case-v06",
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
        metric_claim: "Synthetic polarity conflict",
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

function validOutput(): Gate18PhaseBV06Output {
  return {
    case_id: "case-v06",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim: "The packet supports a bounded claim.",
        support_state: "MIXED",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        causal_link: "E-001 directly supports the bounded claim.",
        counterevidence_ids: ["E-002"],
        counterevidence_link:
          "E-002 weakens the claim by documenting the opposing mechanism.",
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

test("Gate 18 v0.6 hardens IDs and evidence-role instructions without changing scope", () => {
  assert.equal(GATE18_PHASE_B_PROTOCOL_VERSION, "0.6");
  assert.equal(GATE18_PHASE_B_V06_MAX_OUTPUT_TOKENS, 4096);
  assert.equal(
    GATE18_PHASE_B_V06_PROMPT_TEMPLATE_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_V0_4",
  );
  assert.equal(
    GATE18_PHASE_B_V06_PROMPT_TEMPLATE_VERSION,
    "0.4",
  );
  assert.equal(
    GATE18_PHASE_B_V06_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_4",
  );
  assert.equal(
    GATE18_PHASE_B_V06_GENERATION_SCHEMA_VERSION,
    "0.4",
  );
  assert.match(
    GATE18_PHASE_B_V06_SYSTEM_PROMPT,
    /Never concatenate, quote-combine, abbreviate, or invent IDs/,
  );
  assert.match(
    GATE18_PHASE_B_V06_SYSTEM_PROMPT,
    /Judge evidence polarity relative to the exact claim wording/,
  );
  assert.match(
    GATE18_PHASE_B_V06_SYSTEM_PROMPT,
    /limitation on supportive evidence is not automatically counterevidence/,
  );
});

test("Gate 18 v0.6 accepts a grounded role-separated output", () => {
  const output = gate18PhaseBV07OutputSchema.parse(
    validOutput(),
  );
  assert.doesNotThrow(() =>
    assertGate18PhaseBV07Semantics(packet(), output),
  );
});

test("Gate 18 v0.6 schema rejects concatenated evidence IDs seen in calibration", () => {
  const output = validOutput() as unknown as Record<string, unknown>;
  const findings = output.priority_findings as Array<Record<string, unknown>>;
  findings[0].evidence_ids = ["E-041','E-51"];

  const parsed = gate18PhaseBV07OutputSchema.safeParse(output);
  assert.equal(parsed.success, false);
});

test("Gate 18 v0.6 rejects support and counterevidence overlap", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [
    "E-001",
  ];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V06_SUPPORT_COUNTEREVIDENCE_OVERLAP/,
  );
});

test("Gate 18 v0.6 requires a counterevidence explanation when counterevidence exists", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_link = null;

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V06_COUNTEREVIDENCE_LINK_REQUIRED/,
  );
});

test("Gate 18 v0.6 rejects counterevidence explanations without counterevidence IDs", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V06_COUNTEREVIDENCE_LINK_WITHOUT_IDS/,
  );
});

test("Gate 18 v0.6 requires MIXED findings to expose a challenge", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].conflict_ids = [];

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V06_MIXED_REQUIRES_CHALLENGE/,
  );
});

test("Gate 18 v0.6 requires cited conflicts to touch finding evidence", () => {
  const output = validOutput();
  output.priority_findings[0].evidence_ids = ["E-003"];
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;

  assert.throws(
    () => assertGate18PhaseBV07Semantics(packet(), output),
    /VNEXT_GATE18_V06_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS/,
  );
});

test("Gate 18 v0.6 invariants remain while the runner advances to v0.7", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(source, /model-calibration-pilot-v07/);
  assert.match(source, /gate18PhaseBV07OutputSchema/);
  assert.match(source, /assertGate18PhaseBV07Semantics/);
  assert.match(
    source,
    /GATE18_PHASE_B_V07_MAX_OUTPUT_TOKENS/,
  );
  assert.doesNotMatch(
    source,
    /model-calibration-pilot-v06/,
  );
});
