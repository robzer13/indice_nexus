import { createHash } from "node:crypto";

import { z } from "zod";

import {
  GATE18_PHASE_B_V02_SCOPE,
  buildGate18PhaseBV02ModelInput,
  buildVerifiedGate18V02MoatPacket,
  type Gate18V02EvidencePacket,
  type Gate18V02VerifiedPacket,
} from "./model-calibration-pilot-v02";

export const GATE18_PHASE_B_V03_SCOPE =
  GATE18_PHASE_B_V02_SCOPE;

export const GATE18_PHASE_B_V03_MODULE_ID =
  "MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3" as const;

export const GATE18_PHASE_B_V03_PROMPT_TEMPLATE_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_V0_3" as const;

export const GATE18_PHASE_B_V03_PROMPT_TEMPLATE_VERSION =
  "0.3" as const;

export const GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_3" as const;

export const GATE18_PHASE_B_V03_GENERATION_SCHEMA_VERSION =
  "0.3" as const;

export const GATE18_PHASE_B_V03_MAX_OUTPUT_TOKENS = 1536;

export const GATE18_PHASE_B_V03_SYSTEM_PROMPT = [
  "You are an OroTitan calibration model operating on a bounded point-in-time Research evidence packet scoped to MOAT_INPUTS.",
  "Use only the supplied packet. Do not browse, call tools, use outside facts, or fill gaps from memory.",
  "This is an ASSIST-only evidence-audit task.",
  "Do not render a final moat mechanism judgment, moat durability judgment, runway judgment, valuation conclusion, OQS, OVS, Investment Score, next action, publication decision, or investment conclusion.",
  "Every substantive item must cite exact E-* evidence IDs and, where applicable, exact C-* conflict IDs present in the packet.",
  "Treat unresolved conflicts as unresolved.",
  "Prefer concise, high-materiality outputs over exhaustive narrative.",
  "Every narrative field must be complete and self-contained. Target at most 120 characters and use extra schema headroom only to finish the thought.",
  "Do not end a narrative field mid-word, after a hanging conjunction or preposition, or without terminal punctuation.",
  "Return only the requested structured output.",
].join(" ");

export const GATE18_PHASE_B_V03_MODULE_QUESTION = [
  "Audit the supplied MOAT_INPUTS evidence packet.",
  "Return only the three most decision-relevant evidence findings, up to two material conflicts, up to two candidate weak links, and up to three unresolved questions.",
  "Anchor every item to exact packet IDs.",
  "Use complete, concise narrative statements.",
  "This is evidence triage for human adjudication, not a final moat conclusion.",
].join(" ");

const evidenceIdSchema = z.string().regex(/^E-/).max(12);
const conflictIdSchema = z.string().regex(/^C-/).max(12);

export const gate18PhaseBV03OutputSchema = z.object({
  case_id: z.string().min(1).max(64),
  data_cutoff: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  priority_findings: z
    .array(
      z.object({
        claim: z.string().min(1).max(180),
        support_state: z.enum([
          "SUPPORTED",
          "MIXED",
          "UNRESOLVED",
        ]),
        evidence_ids: z.array(evidenceIdSchema).min(1).max(3),
        conflict_ids: z.array(conflictIdSchema).max(2),
        causal_link: z.string().min(1).max(180),
        counterevidence_ids: z.array(evidenceIdSchema).max(2),
      }),
    )
    .max(3),
  material_conflicts: z
    .array(
      z.object({
        conflict_id: conflictIdSchema,
        implication: z.string().min(1).max(180),
        resolution_state: z.enum([
          "RESOLVED_IN_PACKET",
          "UNRESOLVED_IN_PACKET",
        ]),
      }),
    )
    .max(2),
  weak_link_candidates: z
    .array(
      z.object({
        candidate: z.string().min(1).max(140),
        evidence_ids: z.array(evidenceIdSchema).min(1).max(3),
        conflict_ids: z.array(conflictIdSchema).max(2),
        why_uncertain: z.string().min(1).max(180),
      }),
    )
    .max(2),
  unresolved_points: z
    .array(
      z.object({
        question: z.string().min(1).max(180),
        evidence_ids: z.array(evidenceIdSchema).max(3),
        conflict_ids: z.array(conflictIdSchema).max(2),
      }),
    )
    .max(3),
});

export type Gate18PhaseBV03Output = z.infer<
  typeof gate18PhaseBV03OutputSchema
>;

export const GATE18_PHASE_B_V03_GENERATION_SCHEMA_SPEC = {
  type: "object",
  additionalProperties: false,
  properties: {
    case_id: { type: "string", minLength: 1, maxLength: 64 },
    data_cutoff: {
      type: "string",
      pattern: "^\\d{4}-\\d{2}-\\d{2}$",
    },
    priority_findings: {
      type: "array",
      maxItems: 3,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          claim: { type: "string", minLength: 1, maxLength: 180 },
          support_state: {
            type: "string",
            enum: ["SUPPORTED", "MIXED", "UNRESOLVED"],
          },
          evidence_ids: {
            type: "array",
            minItems: 1,
            maxItems: 3,
            items: { type: "string", pattern: "^E-", maxLength: 12 },
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: { type: "string", pattern: "^C-", maxLength: 12 },
          },
          causal_link: {
            type: "string",
            minLength: 1,
            maxLength: 180,
          },
          counterevidence_ids: {
            type: "array",
            maxItems: 2,
            items: { type: "string", pattern: "^E-", maxLength: 12 },
          },
        },
        required: [
          "claim",
          "support_state",
          "evidence_ids",
          "conflict_ids",
          "causal_link",
          "counterevidence_ids",
        ],
      },
    },
    material_conflicts: {
      type: "array",
      maxItems: 2,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          conflict_id: { type: "string", pattern: "^C-", maxLength: 12 },
          implication: { type: "string", minLength: 1, maxLength: 180 },
          resolution_state: {
            type: "string",
            enum: [
              "RESOLVED_IN_PACKET",
              "UNRESOLVED_IN_PACKET",
            ],
          },
        },
        required: [
          "conflict_id",
          "implication",
          "resolution_state",
        ],
      },
    },
    weak_link_candidates: {
      type: "array",
      maxItems: 2,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          candidate: { type: "string", minLength: 1, maxLength: 140 },
          evidence_ids: {
            type: "array",
            minItems: 1,
            maxItems: 3,
            items: { type: "string", pattern: "^E-", maxLength: 12 },
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: { type: "string", pattern: "^C-", maxLength: 12 },
          },
          why_uncertain: {
            type: "string",
            minLength: 1,
            maxLength: 180,
          },
        },
        required: [
          "candidate",
          "evidence_ids",
          "conflict_ids",
          "why_uncertain",
        ],
      },
    },
    unresolved_points: {
      type: "array",
      maxItems: 3,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          question: { type: "string", minLength: 1, maxLength: 180 },
          evidence_ids: {
            type: "array",
            maxItems: 3,
            items: { type: "string", pattern: "^E-", maxLength: 12 },
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: { type: "string", pattern: "^C-", maxLength: 12 },
          },
        },
        required: [
          "question",
          "evidence_ids",
          "conflict_ids",
        ],
      },
    },
  },
  required: [
    "case_id",
    "data_cutoff",
    "priority_findings",
    "material_conflicts",
    "weak_link_candidates",
    "unresolved_points",
  ],
} as const;

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function assertCompleteNarrative(
  value: string,
  code: string,
  saturationBoundary = 178,
): void {
  const trimmed = value.trim();

  if (!/[.!?]$/.test(trimmed)) {
    throw new Error(code);
  }

  if (trimmed.length >= saturationBoundary) {
    throw new Error(
      "VNEXT_GATE18_V03_NARRATIVE_BOUNDARY_SATURATION",
    );
  }
}

export function assertGate18PhaseBV03Semantics(
  packet: Gate18V02EvidencePacket,
  output: Gate18PhaseBV03Output,
): void {
  if (output.case_id !== packet.case_id) {
    throw new Error("VNEXT_GATE18_V03_CASE_MISMATCH");
  }
  if (output.data_cutoff !== packet.data_cutoff) {
    throw new Error("VNEXT_GATE18_V03_CUTOFF_MISMATCH");
  }

  const evidenceIds = new Set(
    packet.evidence_items.map((item) => item.evidence_id),
  );
  const conflictIds = new Set(
    packet.conflicts.map((item) => item.conflict_id),
  );

  const evidenceRefs: string[] = [];
  const conflictRefs: string[] = [];

  for (const finding of output.priority_findings) {
    evidenceRefs.push(
      ...finding.evidence_ids,
      ...finding.counterevidence_ids,
    );
    conflictRefs.push(...finding.conflict_ids);
    assertCompleteNarrative(
      finding.claim,
      "VNEXT_GATE18_V03_FINDING_CLAIM_INCOMPLETE",
    );
    assertCompleteNarrative(
      finding.causal_link,
      "VNEXT_GATE18_V03_CAUSAL_LINK_INCOMPLETE",
    );
  }

  for (const conflict of output.material_conflicts) {
    conflictRefs.push(conflict.conflict_id);
    assertCompleteNarrative(
      conflict.implication,
      "VNEXT_GATE18_V03_CONFLICT_IMPLICATION_INCOMPLETE",
    );
  }

  for (const candidate of output.weak_link_candidates) {
    evidenceRefs.push(...candidate.evidence_ids);
    conflictRefs.push(...candidate.conflict_ids);
    assertCompleteNarrative(
      candidate.candidate,
      "VNEXT_GATE18_V03_WEAK_LINK_CANDIDATE_INCOMPLETE",
      138,
    );
    assertCompleteNarrative(
      candidate.why_uncertain,
      "VNEXT_GATE18_V03_WEAK_LINK_EXPLANATION_INCOMPLETE",
    );
  }

  for (const point of output.unresolved_points) {
    evidenceRefs.push(...point.evidence_ids);
    conflictRefs.push(...point.conflict_ids);
    assertCompleteNarrative(
      point.question,
      "VNEXT_GATE18_V03_UNRESOLVED_QUESTION_INCOMPLETE",
    );
  }

  for (const ref of evidenceRefs) {
    if (!evidenceIds.has(ref)) {
      throw new Error(
        "VNEXT_GATE18_V03_UNKNOWN_EVIDENCE_REF",
      );
    }
  }

  for (const ref of conflictRefs) {
    if (!conflictIds.has(ref)) {
      throw new Error(
        "VNEXT_GATE18_V03_UNKNOWN_CONFLICT_REF",
      );
    }
  }
}

export function buildVerifiedGate18V03MoatPacket(
  ...args: Parameters<typeof buildVerifiedGate18V02MoatPacket>
): Gate18V02VerifiedPacket {
  return buildVerifiedGate18V02MoatPacket(...args);
}

export function buildGate18PhaseBV03ModelInput(
  packet: Gate18V02EvidencePacket,
): string {
  const v02 = buildGate18PhaseBV02ModelInput(packet);
  const marker = "POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:";
  const packetStart = v02.indexOf(marker);

  if (packetStart < 0) {
    throw new Error(
      "VNEXT_GATE18_V03_V02_PACKET_MARKER_MISSING",
    );
  }

  return [
    GATE18_PHASE_B_V03_MODULE_QUESTION,
    "",
    v02.slice(packetStart),
  ].join("\n");
}

export function gate18PhaseBV03PromptTemplateSha256(): string {
  return sha256Hex(
    JSON.stringify({
      system: GATE18_PHASE_B_V03_SYSTEM_PROMPT,
      question: GATE18_PHASE_B_V03_MODULE_QUESTION,
    }),
  );
}

export function gate18PhaseBV03GenerationSchemaSha256(): string {
  return sha256Hex(
    JSON.stringify(
      GATE18_PHASE_B_V03_GENERATION_SCHEMA_SPEC,
    ),
  );
}
