import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import type {
  Gate18V02EvidenceItem,
  Gate18V02EvidencePacket,
} from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_VERSION,
  gate18PhaseBV10GenerationSchemaSha256,
  type Gate18PhaseBV10Output,
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  GATE18_V10_TARGETED_PROBE_GENERATION_SCHEMA_VERSION,
  GATE18_V10_TARGETED_PROBE_ID,
  GATE18_V10_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
  GATE18_V10_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
  GATE18_V10_TARGETED_PROBE_QUESTION,
  assertGate18V10TargetedProbeSemantics,
  buildGate18V10TargetedProbeInput,
  gate18V10TargetedProbeGenerationSchemaSha256,
  gate18V10TargetedProbePromptSha256,
} from "../runtime/vnext/model-calibration-targeted-regression-v10";

function evidence(
  evidenceId: string,
  conflictStatus: string | null = null,
): Gate18V02EvidenceItem {
  return {
    evidence_id: evidenceId,
    claim_id: evidenceId,
    claim: evidenceId,
    value: `Synthetic ${evidenceId}.`,
    period: "FY2025",
    source_refs: [`SRC-${evidenceId.slice(2)}`],
    source_class: "S1",
    claim_fit: "HIGH",
    epistemic_type: "REPORTED",
    freshness_state: "CURRENT",
    limitations: null,
    conflict_status: conflictStatus,
    module_tags: ["MOAT_INPUTS"],
  };
}

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "adyen-probe",
    display_name: "Adyen",
    role: "DIFFICULT_MOAT_CONFLICTING_EVIDENCE",
    source_run_id: "adyen-probe",
    data_cutoff: "2026-09-19",
    source_integrity: {
      evidence_ledger_sha256: "a".repeat(64),
      conflict_ledger_sha256: "b".repeat(64),
    },
    evidence_items: [
      evidence("E-036", "C-005"),
      evidence("E-037", "C-005"),
      evidence("E-040"),
      evidence("E-041", "C-005"),
      evidence("E-042", "C-005"),
      evidence("E-043"),
      evidence("E-055", "C-010"),
      evidence("E-056", "C-010"),
    ],
    conflicts: [
      {
        conflict_id: "C-005",
        metric_claim: "Switching friction",
        value_a: "Low churn and integration",
        value_b: "Multi-PSP and provider replacement",
        conflict_type: "SWITCHING_FRICTION",
        reason: "Opposing mechanisms.",
        resolution: "UNRESOLVED_BUT_BOUNDED",
        resolution_note: "Carry forward.",
        materiality: "HIGH",
        evidence_refs: [
          "E-036",
          "E-037",
          "E-041",
          "E-042",
        ],
        affected_outputs: ["MOAT_INPUTS"],
      },
      {
        conflict_id: "C-010",
        metric_claim: "Reliability versus incident",
        value_a: "Peak uptime",
        value_b: "Documented incident",
        conflict_type: "PEAK_METRIC_VS_INCIDENT",
        reason: "Different scopes.",
        resolution: "RESOLVED_BY_SCOPE",
        resolution_note: "Peak uptime is not incident-free availability.",
        materiality: "MEDIUM",
        evidence_refs: ["E-055", "E-056"],
        affected_outputs: ["MOAT_INPUTS"],
      },
    ],
  };
}

function validOutput(): Gate18PhaseBV10Output {
  return {
    case_id: "adyen-probe",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim:
          "Low churn and integration indicate meaningful customer switching friction.",
        support_state: "MIXED",
        evidence_ids: ["E-036", "E-042"],
        conflict_ids: ["C-005"],
        causal_link:
          "Low churn and integration support relationship persistence.",
        evidence_qualifications: [],
        counterevidence_ids: ["E-037", "E-041"],
        counterevidence_link:
          "Multi-PSP behavior and provider replacement weaken switching friction.",
      },
      {
        claim:
          "Adyen experienced a documented service disruption during the measured period.",
        support_state: "SUPPORTED",
        evidence_ids: ["E-055"],
        conflict_ids: ["C-010"],
        causal_link:
          "E-055 directly documents the service disruption.",
        evidence_qualifications: [],
        counterevidence_ids: [],
        counterevidence_link: null,
      },
      {
        claim:
          "Adyen reports selected positive commercial outcomes.",
        support_state: "SUPPORTED",
        evidence_ids: ["E-040", "E-043"],
        conflict_ids: [],
        causal_link:
          "Selected enterprise wins and a customer case support the bounded report.",
        evidence_qualifications: [
          {
            evidence_id: "E-040",
            qualification:
              "Selected enterprise wins do not quantify revenue contribution.",
          },
          {
            evidence_id: "E-043",
            qualification:
              "The issuer-hosted selected customer case is not representative.",
          },
        ],
        counterevidence_ids: [],
        counterevidence_link: null,
      },
    ],
    material_conflicts: [
      {
        conflict_id: "C-005",
        implication:
          "Opposing switching mechanisms prevent a stronger persistence inference.",
        resolution_state: "UNRESOLVED_IN_PACKET",
      },
      {
        conflict_id: "C-010",
        implication:
          "Peak-event uptime and a separate incident are compatible scoped observations.",
        resolution_state: "RESOLVED_IN_PACKET",
      },
    ],
    weak_link_candidates: [
      {
        candidate: "Switching friction",
        evidence_ids: ["E-036", "E-037", "E-042"],
        conflict_ids: ["C-005"],
        why_uncertain:
          "Retention evidence coexists with mechanisms that enable volume migration.",
      },
    ],
    unresolved_points: [],
  };
}

test("targeted probe preserves v1.0 generation schema while changing only the calibration question", () => {
  assert.equal(
    GATE18_V10_TARGETED_PROBE_ID,
    "ADYEN_CLAIM_TARGET_CORE_001",
  );
  assert.equal(
    GATE18_V10_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
    "GATE18_ADYEN_CLAIM_TARGET_PROBE_V0_1",
  );
  assert.equal(
    GATE18_V10_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
    "0.1",
  );
  assert.equal(
    GATE18_V10_TARGETED_PROBE_GENERATION_SCHEMA_VERSION,
    GATE18_PHASE_B_V10_GENERATION_SCHEMA_VERSION,
  );
  assert.equal(
    gate18V10TargetedProbeGenerationSchemaSha256(),
    gate18PhaseBV10GenerationSchemaSha256(),
  );
  assert.match(
    GATE18_V10_TARGETED_PROBE_QUESTION,
    /Finding 1 must test switching friction/i,
  );
  assert.match(
    GATE18_V10_TARGETED_PROBE_QUESTION,
    /E-056 must not be counterevidence/i,
  );
  assert.match(
    GATE18_V10_TARGETED_PROBE_QUESTION,
    /E-040 and E-043/i,
  );
  assert.equal(
    gate18V10TargetedProbePromptSha256().length,
    64,
  );
});

test("targeted probe model input pins the exact regression roles before the unchanged packet", () => {
  const input = buildGate18V10TargetedProbeInput(packet());

  assert.match(input, /support_state MIXED/);
  assert.match(input, /evidence_ids E-036 and E-042/);
  assert.match(input, /counterevidence_ids E-037 and E-041/);
  assert.match(input, /support_state SUPPORTED, evidence_id E-055/);
  assert.match(input, /POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:/);
});

test("targeted probe accepts the intended same-target MIXED, scope resolution, and qualification map", () => {
  assert.doesNotThrow(() =>
    assertGate18V10TargetedProbeSemantics(
      packet(),
      validOutput(),
    ),
  );
});

test("targeted probe rejects E-056 as counterevidence to incident existence", () => {
  const output = validOutput();
  output.priority_findings[1].counterevidence_ids = [
    "E-056",
  ];
  output.priority_findings[1].counterevidence_link =
    "E-056 allegedly weakens incident existence.";

  assert.throws(
    () =>
      assertGate18V10TargetedProbeSemantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V10_TARGETED_PROBE_INCIDENT_COUNTER/,
  );
});

test("targeted probe rejects false unresolved status for scope-resolved C-010", () => {
  const output = validOutput();
  output.material_conflicts[1].resolution_state =
    "UNRESOLVED_IN_PACKET";

  assert.throws(
    () =>
      assertGate18V10TargetedProbeSemantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V10_TARGETED_PROBE_C010_RESOLUTION/,
  );
});

test("targeted probe requires qualifications for both E-040 and E-043", () => {
  const output = validOutput();
  output.priority_findings[2].evidence_qualifications =
    output.priority_findings[2].evidence_qualifications.filter(
      (item) => item.evidence_id !== "E-043",
    );

  assert.throws(
    () =>
      assertGate18V10TargetedProbeSemantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V10_TARGETED_PROBE_QUALIFICATION_MISSING:E-043/,
  );
});

test("Gate 18 runner exposes targeted probe without changing the default v1.0 path", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(source, /--targeted-regression-probe/);
  assert.match(source, /ADYEN_CLAIM_TARGET_CORE_001/);
  assert.match(source, /buildGate18V10TargetedProbeInput/);
  assert.match(source, /assertGate18V10TargetedProbeSemantics/);
});
