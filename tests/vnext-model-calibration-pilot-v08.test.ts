import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type {
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V08_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V08_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V08_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V08_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V08_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V08_SYSTEM_PROMPT,
  assertGate18PhaseBV08Semantics,
  gate18PhaseBV08OutputSchema,
  type Gate18PhaseBV08Output,
} from "../runtime/vnext/model-calibration-pilot-v08";

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "case-v08",
    display_name: "Synthetic",
    role: "SYNTHETIC",
    source_run_id: "case-v08",
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

function validOutput(): Gate18PhaseBV08Output {
  return {
    case_id: "case-v08",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim: "The packet supports a bounded claim.",
        support_state: "MIXED",
        evidence_ids: ["E-001"],
        conflict_ids: ["C-001"],
        causal_link: "E-001 directly supports the bounded claim.",
        evidence_qualifications: [
          {
            evidence_id: "E-001",
            qualification:
              "E-001 supports the claim but comes from a selected disclosure.",
          },
          {
            evidence_id: "E-002",
            qualification:
              "E-002 weakens the claim but its coverage is incomplete.",
          },
        ],
        counterevidence_ids: ["E-002"],
        counterevidence_link:
          "E-002 weakens the claim by documenting an opposing mechanism.",
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

test("Gate 18 v0.8 makes qualification orthogonal to evidence direction", () => {
  assert.equal(GATE18_PHASE_B_PROTOCOL_VERSION, "0.8");
  assert.equal(GATE18_PHASE_B_V08_MAX_OUTPUT_TOKENS, 4096);
  assert.equal(
    GATE18_PHASE_B_V08_PROMPT_TEMPLATE_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_V0_6",
  );
  assert.equal(
    GATE18_PHASE_B_V08_PROMPT_TEMPLATE_VERSION,
    "0.6",
  );
  assert.equal(
    GATE18_PHASE_B_V08_GENERATION_SCHEMA_ID,
    "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6",
  );
  assert.equal(
    GATE18_PHASE_B_V08_GENERATION_SCHEMA_VERSION,
    "0.6",
  );
  assert.match(
    GATE18_PHASE_B_V08_SYSTEM_PROMPT,
    /Evidence direction and evidence qualification are separate dimensions/,
  );
  assert.match(
    GATE18_PHASE_B_V08_SYSTEM_PROMPT,
    /A qualified item keeps its directional role/,
  );
  assert.match(
    GATE18_PHASE_B_V08_SYSTEM_PROMPT,
    /record the limitation in evidence_qualifications rather than changing direction/,
  );
});

test("Gate 18 v0.8 accepts qualification metadata on supporting and counter evidence", () => {
  const output = gate18PhaseBV08OutputSchema.parse(
    validOutput(),
  );
  assert.doesNotThrow(() =>
    assertGate18PhaseBV08Semantics(packet(), output),
  );
});

test("Gate 18 v0.8 permits E-040-style support plus qualification without duplicating direction", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].support_state = "UNRESOLVED";
  output.priority_findings[0].evidence_qualifications = [
    {
      evidence_id: "E-001",
      qualification:
        "E-001 supports the observation but does not quantify its economic weight.",
    },
  ];

  assert.doesNotThrow(() =>
    assertGate18PhaseBV08Semantics(packet(), output),
  );
});

test("Gate 18 v0.8 rejects support and counterevidence direction overlap", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [
    "E-001",
  ];

  assert.throws(
    () => assertGate18PhaseBV08Semantics(packet(), output),
    /VNEXT_GATE18_V08_DIRECTION_ROLE_OVERLAP/,
  );
});

test("Gate 18 v0.8 rejects orphan qualification metadata", () => {
  const output = validOutput();
  output.priority_findings[0].evidence_qualifications = [
    {
      evidence_id: "E-003",
      qualification:
        "E-003 is limited but is not directionally cited in the finding.",
    },
  ];

  assert.throws(
    () => assertGate18PhaseBV08Semantics(packet(), output),
    /VNEXT_GATE18_V08_ORPHAN_EVIDENCE_QUALIFICATION/,
  );
});

test("Gate 18 v0.8 rejects duplicate qualification metadata for one evidence ID", () => {
  const output = validOutput();
  output.priority_findings[0].evidence_qualifications = [
    {
      evidence_id: "E-001",
      qualification:
        "E-001 has a first qualification.",
    },
    {
      evidence_id: "E-001",
      qualification:
        "E-001 has a second qualification.",
    },
  ];

  assert.throws(
    () => assertGate18PhaseBV08Semantics(packet(), output),
    /VNEXT_GATE18_V08_DUPLICATE_QUALIFICATION_REF/,
  );
});

test("Gate 18 v0.8 rejects malformed qualification evidence IDs", () => {
  const output = validOutput() as unknown as Record<string, unknown>;
  const findings = output.priority_findings as Array<Record<string, unknown>>;
  findings[0].evidence_qualifications = [
    {
      evidence_id: "E-001','E-002",
      qualification: "Malformed reference.",
    },
  ];

  const parsed = gate18PhaseBV08OutputSchema.safeParse(output);
  assert.equal(parsed.success, false);
});

test("Gate 18 v0.8 qualification alone does not justify MIXED", () => {
  const output = validOutput();
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].conflict_ids = [];
  output.priority_findings[0].evidence_qualifications = [
    {
      evidence_id: "E-001",
      qualification:
        "E-001 supports the claim but comes from a selected disclosure.",
    },
  ];

  assert.throws(
    () => assertGate18PhaseBV08Semantics(packet(), output),
    /VNEXT_GATE18_V08_MIXED_REQUIRES_CHALLENGE/,
  );
});

test("Gate 18 v0.8 allows supported evidence with qualification but no opposition", () => {
  const output = validOutput();
  output.priority_findings[0].support_state = "SUPPORTED";
  output.priority_findings[0].counterevidence_ids = [];
  output.priority_findings[0].counterevidence_link = null;
  output.priority_findings[0].conflict_ids = [];
  output.priority_findings[0].evidence_qualifications = [
    {
      evidence_id: "E-001",
      qualification:
        "E-001 supports the claim but comes from a selected disclosure.",
    },
  ];

  assert.doesNotThrow(() =>
    assertGate18PhaseBV08Semantics(packet(), output),
  );
});

test("Gate 18 v0.8 invariants remain while the runner advances to v0.9", () => {
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
    /model-calibration-pilot-v07/,
  );
});
