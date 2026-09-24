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
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_GENERATION_SCHEMA_VERSION,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_QUESTION,
  assertGate18V10BrookfieldTargetedProbeSemantics,
  buildGate18V10BrookfieldTargetedProbeInput,
  gate18V10BrookfieldTargetedProbeGenerationSchemaSha256,
  gate18V10BrookfieldTargetedProbePromptSha256,
} from "../runtime/vnext/model-calibration-targeted-brookfield-v10";

function evidence(
  evidenceId: string,
  limitations: string | null = null,
): Gate18V02EvidenceItem {
  return {
    evidence_id: evidenceId,
    claim_id: evidenceId,
    claim: evidenceId,
    value: `Synthetic ${evidenceId}.`,
    period: "Q2 2026",
    source_refs: [`SRC-${evidenceId.slice(2)}`],
    source_class: "S1",
    claim_fit: "HIGH",
    epistemic_type: "REPORTED",
    freshness_state: "CURRENT",
    limitations,
    conflict_status: null,
    module_tags: ["MOAT_INPUTS"],
  };
}

function packet(): Gate18V02EvidencePacket {
  return {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: "MOAT_INPUTS",
    case_id: "brookfield-probe",
    display_name: "Brookfield Corporation",
    role: "ACCOUNTING_HEAVY",
    source_run_id: "brookfield-probe",
    data_cutoff: "2026-09-19",
    source_integrity: {
      evidence_ledger_sha256: "a".repeat(64),
      conflict_ledger_sha256: "b".repeat(64),
    },
    evidence_items: [
      evidence("E-036"),
      evidence("E-037"),
      evidence(
        "E-039",
        "Peer accounting differs; evidence is for replicability tests.",
      ),
      evidence(
        "E-042",
        "Supports opportunity scale and strong competing capital.",
      ),
    ],
    conflicts: [],
  };
}

function validOutput(): Gate18PhaseBV10Output {
  return {
    case_id: "brookfield-probe",
    data_cutoff: "2026-09-19",
    priority_findings: [
      {
        claim:
          "Large-scale alternative-asset-management capability is present across Brookfield's major peers.",
        support_state: "SUPPORTED",
        evidence_ids: ["E-036", "E-037", "E-039"],
        conflict_ids: [],
        causal_link:
          "Multiple peers report large alternative-platform and insurance scale.",
        evidence_qualifications: [
          {
            evidence_id: "E-039",
            qualification:
              "Peer accounting differs and metrics are not forced to identical definitions.",
          },
        ],
        counterevidence_ids: [],
        counterevidence_link: null,
      },
      {
        claim:
          "Brookfield faces competing capital from major peer platforms in large AI-infrastructure financing initiatives.",
        support_state: "SUPPORTED",
        evidence_ids: ["E-042"],
        conflict_ids: [],
        causal_link:
          "Brookfield participates alongside several large competing capital providers.",
        evidence_qualifications: [],
        counterevidence_ids: [],
        counterevidence_link: null,
      },
    ],
    material_conflicts: [],
    weak_link_candidates: [],
    unresolved_points: [],
  };
}

test("Brookfield targeted probe preserves the v1.0 generation schema", () => {
  assert.equal(
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
    "BROOKFIELD_PEER_ROLE_CORE_001",
  );
  assert.equal(
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
    "GATE18_BROOKFIELD_PEER_ROLE_PROBE_V0_1",
  );
  assert.equal(
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
    "0.1",
  );
  assert.equal(
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_GENERATION_SCHEMA_VERSION,
    GATE18_PHASE_B_V10_GENERATION_SCHEMA_VERSION,
  );
  assert.equal(
    gate18V10BrookfieldTargetedProbeGenerationSchemaSha256(),
    gate18PhaseBV10GenerationSchemaSha256(),
  );
  assert.match(
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_QUESTION,
    /E-039 is peer-platform replicability evidence/i,
  );
  assert.match(
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_QUESTION,
    /E-042 simultaneously indicates opportunity scale and strong competing capital/i,
  );
  assert.equal(
    gate18V10BrookfieldTargetedProbePromptSha256().length,
    64,
  );
});

test("Brookfield targeted input pins the exact historical role regression", () => {
  const input =
    buildGate18V10BrookfieldTargetedProbeInput(packet());

  assert.match(
    input,
    /evidence_ids E-036, E-037, and E-039/i,
  );
  assert.match(
    input,
    /evidence_id E-042/i,
  );
  assert.match(input, /no counterevidence_ids/i);
  assert.match(
    input,
    /POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:/,
  );
});

test("Brookfield targeted validator accepts E-039 and E-042 as support", () => {
  assert.doesNotThrow(() =>
    assertGate18V10BrookfieldTargetedProbeSemantics(
      packet(),
      validOutput(),
    ),
  );
});

test("Brookfield targeted validator rejects E-039 as counterevidence", () => {
  const output = validOutput();
  output.priority_findings[0].evidence_ids = [
    "E-036",
    "E-037",
  ];
  output.priority_findings[0].support_state = "MIXED";
  output.priority_findings[0].counterevidence_ids = [
    "E-039",
  ];
  output.priority_findings[0].counterevidence_link =
    "E-039 allegedly weakens peer replicability.";

  assert.throws(
    () =>
      assertGate18V10BrookfieldTargetedProbeSemantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_STATE|VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_SUPPORT/,
  );
});

test("Brookfield targeted validator rejects E-042 as counterevidence", () => {
  const output = validOutput();
  output.priority_findings[1].evidence_ids = ["E-039"];
  output.priority_findings[1].support_state = "MIXED";
  output.priority_findings[1].counterevidence_ids = [
    "E-042",
  ];
  output.priority_findings[1].counterevidence_link =
    "E-042 allegedly weakens competing-capital evidence.";

  assert.throws(
    () =>
      assertGate18V10BrookfieldTargetedProbeSemantics(
        packet(),
        output,
      ),
    /VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_STATE|VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_SUPPORT/,
  );
});

test("Gate 18 runner exposes both targeted probe identities", () => {
  const source = readFileSync(
    "scripts/vnext-gate18-phase-b.ts",
    "utf8",
  );

  assert.match(
    source,
    /GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID/,
  );
  assert.match(
    source,
    /buildGate18V10BrookfieldTargetedProbeInput/,
  );
  assert.match(
    source,
    /assertGate18V10BrookfieldTargetedProbeSemantics/,
  );
});
