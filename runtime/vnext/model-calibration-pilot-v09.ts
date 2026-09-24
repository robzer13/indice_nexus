import { createHash } from "node:crypto";

import { z } from "zod";

import type {
  Gate18V02EvidencePacket,
  Gate18V02VerifiedPacket,
} from "./model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_V03_MODULE_ID,
  GATE18_PHASE_B_V03_SCOPE,
  GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS,
  buildGate18PhaseBV03ModelInput,
  buildVerifiedGate18V03MoatPacket,
} from "./model-calibration-pilot-v05";

export const GATE18_PHASE_B_PROTOCOL_VERSION = "0.9" as const;

export const GATE18_PHASE_B_V09_MODULE_ID =
  GATE18_PHASE_B_V03_MODULE_ID;

export const GATE18_PHASE_B_V09_SCOPE =
  GATE18_PHASE_B_V03_SCOPE;

export const GATE18_PHASE_B_V09_PROMPT_TEMPLATE_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_V0_7" as const;

export const GATE18_PHASE_B_V09_PROMPT_TEMPLATE_VERSION =
  "0.7" as const;

export const GATE18_PHASE_B_V09_GENERATION_SCHEMA_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6" as const;

export const GATE18_PHASE_B_V09_GENERATION_SCHEMA_VERSION =
  "0.6" as const;

export const GATE18_PHASE_B_V09_MAX_OUTPUT_TOKENS =
  GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS;

export const GATE18_PHASE_B_V09_SYSTEM_PROMPT = [
  "You are an OroTitan calibration model operating on a bounded point-in-time Research evidence packet scoped to MOAT_INPUTS.",
  "Use only the supplied packet. Do not browse, call tools, use outside facts, or fill gaps from memory.",
  "This is an ASSIST-only evidence-audit task.",
  "Do not render a final moat mechanism judgment, moat durability judgment, runway judgment, valuation conclusion, OQS, OVS, Investment Score, next action, publication decision, or investment conclusion.",
  "Every evidence identifier must be copied exactly as one canonical packet ID in the form E-NNN. Never concatenate, quote-combine, abbreviate, or invent IDs.",
  "Every conflict identifier must be copied exactly as one canonical packet ID in the form C-NNN.",
  "For each priority finding, claim must state one atomic directional proposition only. Do not embed the challenge, counterevidence, conflict, caveat, or opposing context inside the claim wording.",
  "Do not write contrastive compound claims using framing such as but, despite, although, though, while, whereas, yet, however, nevertheless, nonetheless, or coexist/coexists with.",
  "For each priority finding, evidence_ids contains only evidence that directly supports the atomic claim wording.",
  "counterevidence_ids contains only evidence that directly weakens or contradicts that same atomic claim through an opposing mechanism, contrary observation, displacement, substitution, or economically inconsistent outcome.",
  "If evidence belongs to the opposing side, keep it outside claim wording and represent it only through counterevidence_ids, conflict_ids/material_conflicts, or evidence_qualifications as appropriate.",
  "Evidence direction and evidence qualification are separate dimensions. Direction is SUPPORT or COUNTEREVIDENCE; qualification describes limits on strength, representativeness, generalizability, measurement quality, validation quality, denominator certainty, or selection bias.",
  "evidence_qualifications contains optional qualification metadata for evidence already cited in evidence_ids or counterevidence_ids. A qualified item keeps its directional role.",
  "Judge direction relative to the exact claim wording. For a skeptical claim, substitution or weak lock-in may support the claim, while low churn or deep integration may be counterevidence.",
  "Issuer-hosted, selected, unaudited, non-representative, or denominator-limited evidence may still be SUPPORT if the observation supports the claim; record the limitation in evidence_qualifications rather than changing direction.",
  "A limitation on supportive evidence must not be placed in counterevidence_ids unless the underlying observation itself weakens the claim.",
  "Never place the same evidence ID in both evidence_ids and counterevidence_ids for one finding.",
  "Each evidence_qualifications entry must reference an ID already present in evidence_ids or counterevidence_ids and explain the limitation in one complete sentence.",
  "If counterevidence_ids is non-empty, counterevidence_link must explain in one complete sentence how those IDs weaken the claim. If it is empty, counterevidence_link must be null.",
  "Treat unresolved conflicts as unresolved.",
  "Prefer concise, high-materiality outputs over exhaustive narrative.",
  "Every narrative sentence field must be complete and self-contained. Target at most 120 characters and use extra schema headroom only to finish the thought.",
  "Do not end a narrative sentence field mid-word, after a hanging conjunction or preposition, or without terminal punctuation.",
  "weak_link_candidates.candidate is a short label, not a sentence, so terminal punctuation is optional.",
  "Return only the requested structured output.",
].join(" ");

export const GATE18_PHASE_B_V09_MODULE_QUESTION = [
  "Audit the supplied MOAT_INPUTS evidence packet.",
  "Return only the three most decision-relevant evidence findings, up to two material conflicts, up to two candidate weak links, and up to three unresolved questions.",
  "For every finding, write one atomic directional claim, then separate directional support from directional counterevidence and attach optional qualification metadata to cited evidence whose strength or scope is limited.",
  "Anchor every item to exact canonical packet IDs.",
  "This is evidence triage for human adjudication, not a final moat conclusion.",
].join(" ");

const evidenceIdSchema = z.string().regex(/^E-\d{3}$/);
const conflictIdSchema = z.string().regex(/^C-\d{3}$/);

export const gate18PhaseBV09OutputSchema = z.object({
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
        evidence_qualifications: z
          .array(
            z.object({
              evidence_id: evidenceIdSchema,
              qualification: z.string().min(1).max(180),
            }),
          )
          .max(3),
        counterevidence_ids: z.array(evidenceIdSchema).max(2),
        counterevidence_link: z
          .string()
          .min(1)
          .max(180)
          .nullable(),
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

export type Gate18PhaseBV09Output = z.infer<
  typeof gate18PhaseBV09OutputSchema
>;

const evidenceIdSpec = {
  type: "string",
  pattern: "^E-\\d{3}$",
} as const;

const conflictIdSpec = {
  type: "string",
  pattern: "^C-\\d{3}$",
} as const;

export const GATE18_PHASE_B_V09_GENERATION_SCHEMA_SPEC = {
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
            items: evidenceIdSpec,
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: conflictIdSpec,
          },
          causal_link: {
            type: "string",
            minLength: 1,
            maxLength: 180,
          },
          evidence_qualifications: {
            type: "array",
            maxItems: 3,
            items: {
              type: "object",
              additionalProperties: false,
              properties: {
                evidence_id: evidenceIdSpec,
                qualification: {
                  type: "string",
                  minLength: 1,
                  maxLength: 180,
                },
              },
              required: ["evidence_id", "qualification"],
            },
          },
          counterevidence_ids: {
            type: "array",
            maxItems: 2,
            items: evidenceIdSpec,
          },
          counterevidence_link: {
            anyOf: [
              {
                type: "string",
                minLength: 1,
                maxLength: 180,
              },
              { type: "null" },
            ],
          },
        },
        required: [
          "claim",
          "support_state",
          "evidence_ids",
          "conflict_ids",
          "causal_link",
          "evidence_qualifications",
          "counterevidence_ids",
          "counterevidence_link",
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
          conflict_id: conflictIdSpec,
          implication: {
            type: "string",
            minLength: 1,
            maxLength: 180,
          },
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
          candidate: {
            type: "string",
            minLength: 1,
            maxLength: 140,
          },
          evidence_ids: {
            type: "array",
            minItems: 1,
            maxItems: 3,
            items: evidenceIdSpec,
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: conflictIdSpec,
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
          question: {
            type: "string",
            minLength: 1,
            maxLength: 180,
          },
          evidence_ids: {
            type: "array",
            maxItems: 3,
            items: evidenceIdSpec,
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: conflictIdSpec,
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
      "VNEXT_GATE18_V09_NARRATIVE_BOUNDARY_SATURATION",
    );
  }
}

const NON_ATOMIC_CLAIM_PATTERNS: readonly RegExp[] = [
  /\bbut\b/i,
  /\bdespite\b/i,
  /\balthough\b/i,
  /\bthough\b/i,
  /\bwhile\b/i,
  /\bwhereas\b/i,
  /\byet\b/i,
  /\bhowever\b/i,
  /\bnevertheless\b/i,
  /\bnonetheless\b/i,
  /\bcoexist(?:s|ed|ing)?\b/i,
];

function assertAtomicClaim(value: string): void {
  if (
    NON_ATOMIC_CLAIM_PATTERNS.some((pattern) =>
      pattern.test(value),
    )
  ) {
    throw new Error(
      "VNEXT_GATE18_V09_NON_ATOMIC_CONTRASTIVE_CLAIM",
    );
  }
}

function assertUnique(
  values: readonly string[],
  code: string,
): void {
  if (new Set(values).size !== values.length) {
    throw new Error(code);
  }
}

export function assertGate18PhaseBV09Semantics(
  packet: Gate18V02EvidencePacket,
  output: Gate18PhaseBV09Output,
): void {
  if (output.case_id !== packet.case_id) {
    throw new Error("VNEXT_GATE18_V09_CASE_MISMATCH");
  }
  if (output.data_cutoff !== packet.data_cutoff) {
    throw new Error("VNEXT_GATE18_V09_CUTOFF_MISMATCH");
  }

  const evidenceIds = new Set(
    packet.evidence_items.map((item) => item.evidence_id),
  );
  const conflictsById = new Map(
    packet.conflicts.map((item) => [item.conflict_id, item]),
  );

  const evidenceRefs: string[] = [];
  const conflictRefs: string[] = [];

  for (const finding of output.priority_findings) {
    assertUnique(
      finding.evidence_ids,
      "VNEXT_GATE18_V09_DUPLICATE_SUPPORT_REF",
    );
    assertUnique(
      finding.counterevidence_ids,
      "VNEXT_GATE18_V09_DUPLICATE_COUNTEREVIDENCE_REF",
    );
    assertUnique(
      finding.conflict_ids,
      "VNEXT_GATE18_V09_DUPLICATE_FINDING_CONFLICT_REF",
    );
    assertUnique(
      finding.evidence_qualifications.map(
        (item) => item.evidence_id,
      ),
      "VNEXT_GATE18_V09_DUPLICATE_QUALIFICATION_REF",
    );

    const supportIds = new Set(finding.evidence_ids);
    const counterIds = new Set(finding.counterevidence_ids);
    if (
      finding.counterevidence_ids.some((id) =>
        supportIds.has(id),
      )
    ) {
      throw new Error(
        "VNEXT_GATE18_V09_DIRECTION_ROLE_OVERLAP",
      );
    }

    for (const qualification of finding.evidence_qualifications) {
      if (
        !supportIds.has(qualification.evidence_id) &&
        !counterIds.has(qualification.evidence_id)
      ) {
        throw new Error(
          "VNEXT_GATE18_V09_ORPHAN_EVIDENCE_QUALIFICATION",
        );
      }
    }

    if (
      finding.counterevidence_ids.length === 0 &&
      finding.counterevidence_link !== null
    ) {
      throw new Error(
        "VNEXT_GATE18_V09_COUNTEREVIDENCE_LINK_WITHOUT_IDS",
      );
    }

    if (
      finding.counterevidence_ids.length > 0 &&
      finding.counterevidence_link === null
    ) {
      throw new Error(
        "VNEXT_GATE18_V09_COUNTEREVIDENCE_LINK_REQUIRED",
      );
    }

    if (
      finding.support_state === "MIXED" &&
      finding.counterevidence_ids.length === 0 &&
      finding.conflict_ids.length === 0
    ) {
      throw new Error(
        "VNEXT_GATE18_V09_MIXED_REQUIRES_CHALLENGE",
      );
    }

    evidenceRefs.push(
      ...finding.evidence_ids,
      ...finding.counterevidence_ids,
      ...finding.evidence_qualifications.map(
        (item) => item.evidence_id,
      ),
    );
    conflictRefs.push(...finding.conflict_ids);

    assertCompleteNarrative(
      finding.claim,
      "VNEXT_GATE18_V09_FINDING_CLAIM_INCOMPLETE",
    );
    assertAtomicClaim(finding.claim);
    assertCompleteNarrative(
      finding.causal_link,
      "VNEXT_GATE18_V09_CAUSAL_LINK_INCOMPLETE",
    );

    for (const qualification of finding.evidence_qualifications) {
      assertCompleteNarrative(
        qualification.qualification,
        "VNEXT_GATE18_V09_EVIDENCE_QUALIFICATION_INCOMPLETE",
      );
    }

    if (finding.counterevidence_link !== null) {
      assertCompleteNarrative(
        finding.counterevidence_link,
        "VNEXT_GATE18_V09_COUNTEREVIDENCE_LINK_INCOMPLETE",
      );
    }

    const findingRefs = new Set([
      ...finding.evidence_ids,
      ...finding.counterevidence_ids,
    ]);

    for (const conflictId of finding.conflict_ids) {
      const conflict = conflictsById.get(conflictId);
      if (!conflict) {
        continue;
      }
      if (
        conflict.evidence_refs.length > 0 &&
        !conflict.evidence_refs.some((id) =>
          findingRefs.has(id),
        )
      ) {
        throw new Error(
          "VNEXT_GATE18_V09_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS",
        );
      }
    }
  }

  assertUnique(
    output.material_conflicts.map((item) => item.conflict_id),
    "VNEXT_GATE18_V09_DUPLICATE_MATERIAL_CONFLICT",
  );

  for (const conflict of output.material_conflicts) {
    conflictRefs.push(conflict.conflict_id);
    assertCompleteNarrative(
      conflict.implication,
      "VNEXT_GATE18_V09_CONFLICT_IMPLICATION_INCOMPLETE",
    );
  }

  for (const candidate of output.weak_link_candidates) {
    assertUnique(
      candidate.evidence_ids,
      "VNEXT_GATE18_V09_DUPLICATE_WEAK_LINK_EVIDENCE_REF",
    );
    assertUnique(
      candidate.conflict_ids,
      "VNEXT_GATE18_V09_DUPLICATE_WEAK_LINK_CONFLICT_REF",
    );
    evidenceRefs.push(...candidate.evidence_ids);
    conflictRefs.push(...candidate.conflict_ids);
    assertCompleteNarrative(
      candidate.why_uncertain,
      "VNEXT_GATE18_V09_WEAK_LINK_EXPLANATION_INCOMPLETE",
    );
  }

  for (const point of output.unresolved_points) {
    assertUnique(
      point.evidence_ids,
      "VNEXT_GATE18_V09_DUPLICATE_UNRESOLVED_EVIDENCE_REF",
    );
    assertUnique(
      point.conflict_ids,
      "VNEXT_GATE18_V09_DUPLICATE_UNRESOLVED_CONFLICT_REF",
    );
    evidenceRefs.push(...point.evidence_ids);
    conflictRefs.push(...point.conflict_ids);
    assertCompleteNarrative(
      point.question,
      "VNEXT_GATE18_V09_UNRESOLVED_QUESTION_INCOMPLETE",
    );
  }

  for (const ref of evidenceRefs) {
    if (!/^E-\d{3}$/.test(ref)) {
      throw new Error(
        "VNEXT_GATE18_V09_EVIDENCE_REF_FORMAT_INVALID",
      );
    }
    if (!evidenceIds.has(ref)) {
      throw new Error(
        "VNEXT_GATE18_V09_UNKNOWN_EVIDENCE_REF",
      );
    }
  }

  for (const ref of conflictRefs) {
    if (!/^C-\d{3}$/.test(ref)) {
      throw new Error(
        "VNEXT_GATE18_V09_CONFLICT_REF_FORMAT_INVALID",
      );
    }
    if (!conflictsById.has(ref)) {
      throw new Error(
        "VNEXT_GATE18_V09_UNKNOWN_CONFLICT_REF",
      );
    }
  }
}

export function buildVerifiedGate18V09MoatPacket(
  ...args: Parameters<typeof buildVerifiedGate18V03MoatPacket>
): Gate18V02VerifiedPacket {
  return buildVerifiedGate18V03MoatPacket(...args);
}

export function buildGate18PhaseBV09ModelInput(
  packet: Gate18V02EvidencePacket,
): string {
  const v03 = buildGate18PhaseBV03ModelInput(packet);
  const marker = "POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:";
  const packetStart = v03.indexOf(marker);

  if (packetStart < 0) {
    throw new Error(
      "VNEXT_GATE18_V09_PACKET_MARKER_MISSING",
    );
  }

  return [
    GATE18_PHASE_B_V09_MODULE_QUESTION,
    "",
    v03.slice(packetStart),
  ].join("\n");
}

export function gate18PhaseBV09PromptTemplateSha256(): string {
  return sha256Hex(
    JSON.stringify({
      system: GATE18_PHASE_B_V09_SYSTEM_PROMPT,
      question: GATE18_PHASE_B_V09_MODULE_QUESTION,
    }),
  );
}

export function gate18PhaseBV09GenerationSchemaSha256(): string {
  return sha256Hex(
    JSON.stringify(
      GATE18_PHASE_B_V09_GENERATION_SCHEMA_SPEC,
    ),
  );
}
