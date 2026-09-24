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

export const GATE18_V10_TARGETED_PROBE_ID =
  "ADYEN_CLAIM_TARGET_CORE_001" as const;

export const GATE18_V10_TARGETED_PROBE_PROMPT_TEMPLATE_ID =
  "GATE18_ADYEN_CLAIM_TARGET_PROBE_V0_1" as const;

export const GATE18_V10_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION =
  "0.1" as const;

export const GATE18_V10_TARGETED_PROBE_SYSTEM_PROMPT =
  GATE18_PHASE_B_V10_SYSTEM_PROMPT;

export const GATE18_V10_TARGETED_PROBE_GENERATION_SCHEMA_ID =
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_ID;

export const GATE18_V10_TARGETED_PROBE_GENERATION_SCHEMA_VERSION =
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_VERSION;

export const GATE18_V10_TARGETED_PROBE_MAX_OUTPUT_TOKENS =
  GATE18_PHASE_B_V10_MAX_OUTPUT_TOKENS;

const REQUIRED_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-040",
  "E-041",
  "E-042",
  "E-043",
  "E-055",
  "E-056",
] as const;

const REQUIRED_CONFLICT_IDS = [
  "C-005",
  "C-010",
] as const;

export const GATE18_V10_TARGETED_PROBE_QUESTION = [
  "Run a calibration-only targeted regression probe on the supplied Adyen MOAT_INPUTS packet.",
  "This probe is not a normal priority-selection task and is not comparison-admissible model-ranking evidence.",
  "Return exactly three priority_findings in the required order below.",
  "Finding 1 must test switching friction at the inferential economic level: use support_state MIXED, evidence_ids E-036 and E-042, conflict_id C-005, and counterevidence_ids E-037 and E-041.",
  "Finding 1 must state one atomic proposition about meaningful customer switching friction or relationship persistence; E-037 and E-041 must weaken that same proposition rather than a different descriptive claim.",
  "Finding 2 must test incident existence under a scope-resolved conflict: state one atomic proposition that a documented service disruption occurred, use support_state SUPPORTED, evidence_id E-055, conflict_id C-010, and no counterevidence_ids.",
  "E-056 must not be counterevidence to Finding 2 because peak-event uptime can coexist with a separate documented incident.",
  "Finding 3 must test qualification orthogonality: state one atomic descriptive proposition that Adyen reports selected positive commercial outcomes, use support_state SUPPORTED, evidence_ids E-040 and E-043, no counterevidence_ids, and attach a separate evidence_qualification to both E-040 and E-043.",
  "The E-040 qualification must preserve that selected enterprise wins do not quantify revenue contribution.",
  "The E-043 qualification must preserve that the issuer-hosted selected case is not representative.",
  "material_conflicts must include C-005 as UNRESOLVED_IN_PACKET and C-010 as RESOLVED_IN_PACKET.",
  "Do not add a fourth priority finding.",
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
  const conflicts = new Set(
    packet.conflicts.map((item) => item.conflict_id),
  );

  for (const id of REQUIRED_EVIDENCE_IDS) {
    if (!evidence.has(id)) {
      throw new Error(
        `VNEXT_GATE18_V10_TARGETED_PROBE_EVIDENCE_MISSING:${id}`,
      );
    }
  }

  for (const id of REQUIRED_CONFLICT_IDS) {
    if (!conflicts.has(id)) {
      throw new Error(
        `VNEXT_GATE18_V10_TARGETED_PROBE_CONFLICT_MISSING:${id}`,
      );
    }
  }
}

export function buildGate18V10TargetedProbeInput(
  packet: Gate18V02EvidencePacket,
): string {
  assertProbePacket(packet);

  const base = buildGate18PhaseBV10ModelInput(packet);
  const marker = "POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:";
  const packetStart = base.indexOf(marker);

  if (packetStart < 0) {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_PACKET_MARKER_MISSING",
    );
  }

  return [
    GATE18_V10_TARGETED_PROBE_QUESTION,
    "",
    base.slice(packetStart),
  ].join("\n");
}

export function gate18V10TargetedProbePromptSha256(): string {
  return sha256Hex(
    JSON.stringify({
      system: GATE18_V10_TARGETED_PROBE_SYSTEM_PROMPT,
      question: GATE18_V10_TARGETED_PROBE_QUESTION,
    }),
  );
}

export function gate18V10TargetedProbeGenerationSchemaSha256(): string {
  return gate18PhaseBV10GenerationSchemaSha256();
}

export function assertGate18V10TargetedProbeSemantics(
  packet: Gate18V02EvidencePacket,
  output: Gate18PhaseBV10Output,
): void {
  assertProbePacket(packet);
  assertGate18PhaseBV10Semantics(packet, output);

  if (output.priority_findings.length !== 3) {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_FINDING_COUNT",
    );
  }

  const [switching, incident, qualification] =
    output.priority_findings;

  if (switching.support_state !== "MIXED") {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_SWITCHING_NOT_MIXED",
    );
  }
  assertExactIds(
    switching.evidence_ids,
    ["E-036", "E-042"],
    "VNEXT_GATE18_V10_TARGETED_PROBE_SWITCHING_SUPPORT",
  );
  assertExactIds(
    switching.counterevidence_ids,
    ["E-037", "E-041"],
    "VNEXT_GATE18_V10_TARGETED_PROBE_SWITCHING_COUNTER",
  );
  assertExactIds(
    switching.conflict_ids,
    ["C-005"],
    "VNEXT_GATE18_V10_TARGETED_PROBE_SWITCHING_CONFLICT",
  );
  if (switching.counterevidence_link === null) {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_SWITCHING_LINK",
    );
  }

  if (incident.support_state !== "SUPPORTED") {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_INCIDENT_STATE",
    );
  }
  assertExactIds(
    incident.evidence_ids,
    ["E-055"],
    "VNEXT_GATE18_V10_TARGETED_PROBE_INCIDENT_SUPPORT",
  );
  assertExactIds(
    incident.counterevidence_ids,
    [],
    "VNEXT_GATE18_V10_TARGETED_PROBE_INCIDENT_COUNTER",
  );
  assertExactIds(
    incident.conflict_ids,
    ["C-010"],
    "VNEXT_GATE18_V10_TARGETED_PROBE_INCIDENT_CONFLICT",
  );
  if (incident.counterevidence_link !== null) {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_INCIDENT_LINK",
    );
  }

  if (qualification.support_state !== "SUPPORTED") {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_QUALIFICATION_STATE",
    );
  }
  assertExactIds(
    qualification.evidence_ids,
    ["E-040", "E-043"],
    "VNEXT_GATE18_V10_TARGETED_PROBE_QUALIFICATION_SUPPORT",
  );
  assertExactIds(
    qualification.counterevidence_ids,
    [],
    "VNEXT_GATE18_V10_TARGETED_PROBE_QUALIFICATION_COUNTER",
  );

  const qualificationIds = new Set(
    qualification.evidence_qualifications.map(
      (item) => item.evidence_id,
    ),
  );
  for (const id of ["E-040", "E-043"]) {
    if (!qualificationIds.has(id)) {
      throw new Error(
        `VNEXT_GATE18_V10_TARGETED_PROBE_QUALIFICATION_MISSING:${id}`,
      );
    }
  }

  const c005 = output.material_conflicts.find(
    (item) => item.conflict_id === "C-005",
  );
  const c010 = output.material_conflicts.find(
    (item) => item.conflict_id === "C-010",
  );

  if (
    !c005 ||
    c005.resolution_state !== "UNRESOLVED_IN_PACKET"
  ) {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_C005_RESOLUTION",
    );
  }
  if (
    !c010 ||
    c010.resolution_state !== "RESOLVED_IN_PACKET"
  ) {
    throw new Error(
      "VNEXT_GATE18_V10_TARGETED_PROBE_C010_RESOLUTION",
    );
  }
}
