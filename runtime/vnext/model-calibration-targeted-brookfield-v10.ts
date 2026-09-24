import { createHash } from "node:crypto";

import type { Gate18V02EvidencePacket } from "./model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V10_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_V10_SYSTEM_PROMPT,
  assertGate18PhaseBV10Semantics,
  buildGate18PhaseBV10ModelInput,
  gate18PhaseBV10GenerationSchemaSha256,
  type Gate18PhaseBV10Output,
} from "./model-calibration-pilot-v10";

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID =
  "BROOKFIELD_PEER_ROLE_CORE_001" as const;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID =
  "GATE18_BROOKFIELD_PEER_ROLE_PROBE_V0_1" as const;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION =
  "0.1" as const;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT =
  GATE18_PHASE_B_V10_SYSTEM_PROMPT;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_GENERATION_SCHEMA_ID =
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_ID;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_GENERATION_SCHEMA_VERSION =
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_VERSION;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_MAX_OUTPUT_TOKENS =
  GATE18_PHASE_B_V10_MAX_OUTPUT_TOKENS;

const REQUIRED_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-039",
  "E-042",
] as const;

export const GATE18_V10_BROOKFIELD_TARGETED_PROBE_QUESTION = [
  "Run a calibration-only targeted regression probe on the supplied Brookfield Corporation MOAT_INPUTS packet.",
  "This probe is not a normal priority-selection task and is not comparison-admissible model-ranking evidence.",
  "Return exactly two priority_findings in the required order below.",
  "Finding 1 must state one atomic descriptive proposition that large-scale alternative-asset-management capability is present across Brookfield's major peers.",
  "Finding 1 must use support_state SUPPORTED, evidence_ids E-036, E-037, and E-039 in that order, no conflict_ids, and no counterevidence_ids.",
  "E-039 is peer-platform replicability evidence and must remain SUPPORT; its accounting-comparability limitation may be expressed as an evidence qualification but must not invert its direction.",
  "Finding 2 must state one atomic descriptive proposition that Brookfield faces competing capital from major peer platforms in large AI-infrastructure financing initiatives.",
  "Finding 2 must use support_state SUPPORTED, evidence_id E-042, no conflict_ids, and no counterevidence_ids.",
  "E-042 simultaneously indicates opportunity scale and strong competing capital; for this exact competition proposition it is SUPPORT, not COUNTEREVIDENCE.",
  "Do not add a third priority finding.",
  "material_conflicts may be empty because this probe targets evidence-role polarity rather than packet conflict resolution.",
  "Do not browse or use outside facts.",
  "Keep all normal v1.0 atomicity, evidence-role, counterfactual, scope, qualification, and human-judgment boundaries.",
].join(" ");

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function assertExactIds(
  actual: readonly string[],
  expected: readonly string[],
  code: string,
): void {
  if (
    actual.length !== expected.length ||
    actual.some((value, index) => value !== expected[index])
  ) {
    throw new Error(code);
  }
}

function assertProbePacket(packet: Gate18V02EvidencePacket): void {
  const evidence = new Set(
    packet.evidence_items.map((item) => item.evidence_id),
  );

  for (const id of REQUIRED_EVIDENCE_IDS) {
    if (!evidence.has(id)) {
      throw new Error(
        `VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_EVIDENCE_MISSING:${id}`,
      );
    }
  }
}

export function buildGate18V10BrookfieldTargetedProbeInput(
  packet: Gate18V02EvidencePacket,
): string {
  assertProbePacket(packet);

  const base = buildGate18PhaseBV10ModelInput(packet);
  const marker = "POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:";
  const packetStart = base.indexOf(marker);

  if (packetStart < 0) {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PACKET_MARKER_MISSING",
    );
  }

  return [
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_QUESTION,
    "",
    base.slice(packetStart),
  ].join("\n");
}

export function gate18V10BrookfieldTargetedProbePromptSha256(): string {
  return sha256Hex(
    JSON.stringify({
      system: GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT,
      question: GATE18_V10_BROOKFIELD_TARGETED_PROBE_QUESTION,
    }),
  );
}

export function gate18V10BrookfieldTargetedProbeGenerationSchemaSha256(): string {
  return gate18PhaseBV10GenerationSchemaSha256();
}

export function assertGate18V10BrookfieldTargetedProbeSemantics(
  packet: Gate18V02EvidencePacket,
  output: Gate18PhaseBV10Output,
): void {
  assertProbePacket(packet);
  assertGate18PhaseBV10Semantics(packet, output);

  if (output.priority_findings.length !== 2) {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_FINDING_COUNT",
    );
  }

  const [peerScale, aiCompetition] = output.priority_findings;

  if (peerScale.support_state !== "SUPPORTED") {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_STATE",
    );
  }
  assertExactIds(
    peerScale.evidence_ids,
    ["E-036", "E-037", "E-039"],
    "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_SUPPORT",
  );
  assertExactIds(
    peerScale.conflict_ids,
    [],
    "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_CONFLICT",
  );
  assertExactIds(
    peerScale.counterevidence_ids,
    [],
    "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_COUNTER",
  );
  if (peerScale.counterevidence_link !== null) {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_SCALE_LINK",
    );
  }

  if (aiCompetition.support_state !== "SUPPORTED") {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_STATE",
    );
  }
  assertExactIds(
    aiCompetition.evidence_ids,
    ["E-042"],
    "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_SUPPORT",
  );
  assertExactIds(
    aiCompetition.conflict_ids,
    [],
    "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_CONFLICT",
  );
  assertExactIds(
    aiCompetition.counterevidence_ids,
    [],
    "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_COUNTER",
  );
  if (aiCompetition.counterevidence_link !== null) {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_AI_LINK",
    );
  }

  const peerQualificationIds = new Set(
    peerScale.evidence_qualifications.map(
      (item) => item.evidence_id,
    ),
  );
  if (
    peerScale.evidence_qualifications.length > 0 &&
    !peerQualificationIds.has("E-039")
  ) {
    throw new Error(
      "VNEXT_GATE18_V10_BROOKFIELD_TARGETED_PROBE_PEER_QUALIFICATION_DIRECTION",
    );
  }
}
