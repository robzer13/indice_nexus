import { createHash } from "node:crypto";

import { z } from "zod";

export const GATE18_PRIVATE_SOURCE_REPOSITORY =
  "robzer13/real-orotitan" as const;

export const GATE18_PHASE_B_MODULE_ID =
  "EVIDENCE_AUDIT_ASSISTED_V0_1" as const;

export const GATE18_PHASE_B_PROMPT_TEMPLATE_ID =
  "GATE18_EVIDENCE_AUDIT_V0_1" as const;

export const GATE18_PHASE_B_PROMPT_TEMPLATE_VERSION =
  "0.1" as const;

export const GATE18_PHASE_B_GENERATION_SCHEMA_ID =
  "GATE18_EVIDENCE_AUDIT_OUTPUT_V0_1" as const;

export const GATE18_PHASE_B_GENERATION_SCHEMA_VERSION =
  "0.1" as const;

export const GATE18_PHASE_B_MAX_OUTPUT_TOKENS = 1536;

export const GATE18_PHASE_B_SYSTEM_PROMPT = [
  "You are an OroTitan calibration model operating on a bounded point-in-time evidence packet.",
  "Use only the supplied packet. Do not use outside facts, browsing, tools, or unstated assumptions.",
  "This is an ASSIST-only calibration task. Do not render a final moat, durability, runway, valuation, OQS, OVS, Investment Score, next-action, publication, or investment conclusion.",
  "Every substantive finding must cite exact E-* evidence IDs and, where applicable, exact C-* conflict IDs from the packet.",
  "Treat unresolved conflicts as unresolved. Do not resolve a conflict in management's favor without packet evidence.",
  "Surface counter-evidence, weak-link candidates, and unresolved questions explicitly.",
  "Return only the requested structured output.",
].join(" ");

export const GATE18_PHASE_B_MODULE_QUESTION = [
  "Using only the supplied point-in-time Research Evidence Ledger and Conflict Ledger projection, produce an assisted evidence audit.",
  "Identify the most decision-relevant supported or mixed findings, material source conflicts, candidate weak links, and unresolved points.",
  "For each item, anchor the statement to exact evidence/conflict IDs and explain the causal or uncertainty link concisely.",
  "Do not make any final economic or investment judgment.",
].join(" ");

export const GATE18_PHASE_B_GENERATION_SCHEMA_SPEC = {
  type: "object",
  additionalProperties: false,
  properties: {
    case_id: { type: "string", minLength: 1 },
    data_cutoff: {
      type: "string",
      pattern: "^\\d{4}-\\d{2}-\\d{2}$",
    },
    findings: {
      type: "array",
      maxItems: 8,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          finding_id: { type: "string", minLength: 1 },
          claim: { type: "string", minLength: 1 },
          support_state: {
            type: "string",
            enum: ["SUPPORTED", "MIXED", "UNRESOLVED"],
          },
          evidence_ids: {
            type: "array",
            minItems: 1,
            items: { type: "string", pattern: "^E-" },
          },
          conflict_ids: {
            type: "array",
            items: { type: "string", pattern: "^C-" },
          },
          causal_link: { type: "string", minLength: 1 },
          counterevidence_ids: {
            type: "array",
            items: { type: "string", pattern: "^E-" },
          },
        },
        required: [
          "finding_id",
          "claim",
          "support_state",
          "evidence_ids",
          "conflict_ids",
          "causal_link",
          "counterevidence_ids",
        ],
      },
    },
    conflicts: {
      type: "array",
      maxItems: 8,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          conflict_id: { type: "string", pattern: "^C-" },
          implication: { type: "string", minLength: 1 },
          evidence_ids: {
            type: "array",
            items: { type: "string", pattern: "^E-" },
          },
          resolution_state: {
            type: "string",
            enum: ["RESOLVED_IN_PACKET", "UNRESOLVED_IN_PACKET"],
          },
        },
        required: [
          "conflict_id",
          "implication",
          "evidence_ids",
          "resolution_state",
        ],
      },
    },
    weak_link_candidates: {
      type: "array",
      maxItems: 5,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          candidate: { type: "string", minLength: 1 },
          evidence_ids: {
            type: "array",
            minItems: 1,
            items: { type: "string", pattern: "^E-" },
          },
          conflict_ids: {
            type: "array",
            items: { type: "string", pattern: "^C-" },
          },
          why_uncertain: { type: "string", minLength: 1 },
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
      maxItems: 8,
      items: {
        type: "object",
        additionalProperties: false,
        properties: {
          question: { type: "string", minLength: 1 },
          evidence_ids: {
            type: "array",
            items: { type: "string", pattern: "^E-" },
          },
          conflict_ids: {
            type: "array",
            items: { type: "string", pattern: "^C-" },
          },
        },
        required: ["question", "evidence_ids", "conflict_ids"],
      },
    },
  },
  required: [
    "case_id",
    "data_cutoff",
    "findings",
    "conflicts",
    "weak_link_candidates",
    "unresolved_points",
  ],
} as const;

export const gate18PhaseBOutputSchema = z.object({
  case_id: z.string().min(1),
  data_cutoff: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  findings: z
    .array(
      z.object({
        finding_id: z.string().min(1),
        claim: z.string().min(1),
        support_state: z.enum([
          "SUPPORTED",
          "MIXED",
          "UNRESOLVED",
        ]),
        evidence_ids: z.array(z.string().regex(/^E-/)).min(1),
        conflict_ids: z.array(z.string().regex(/^C-/)),
        causal_link: z.string().min(1),
        counterevidence_ids: z.array(z.string().regex(/^E-/)),
      }),
    )
    .max(8),
  conflicts: z
    .array(
      z.object({
        conflict_id: z.string().regex(/^C-/),
        implication: z.string().min(1),
        evidence_ids: z.array(z.string().regex(/^E-/)),
        resolution_state: z.enum([
          "RESOLVED_IN_PACKET",
          "UNRESOLVED_IN_PACKET",
        ]),
      }),
    )
    .max(8),
  weak_link_candidates: z
    .array(
      z.object({
        candidate: z.string().min(1),
        evidence_ids: z.array(z.string().regex(/^E-/)).min(1),
        conflict_ids: z.array(z.string().regex(/^C-/)),
        why_uncertain: z.string().min(1),
      }),
    )
    .max(5),
  unresolved_points: z
    .array(
      z.object({
        question: z.string().min(1),
        evidence_ids: z.array(z.string().regex(/^E-/)),
        conflict_ids: z.array(z.string().regex(/^C-/)),
      }),
    )
    .max(8),
});

export type Gate18PhaseBOutput = z.infer<
  typeof gate18PhaseBOutputSchema
>;

export interface Gate18ArtifactPin {
  artifact_id: string;
  version: number;
  sha256: string;
  repository: string;
  commit_sha: string;
  path: string;
}

export interface Gate18PilotCompany {
  role: string;
  display_name: string;
  source_run_id: string;
  data_cutoff: string;
  evidence_ledger: Gate18ArtifactPin;
  conflict_ledger: Gate18ArtifactPin;
}

export interface Gate18EvidencePacket {
  format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.1";
  case_id: string;
  display_name: string;
  role: string;
  source_run_id: string;
  data_cutoff: string;
  source_integrity: {
    evidence_ledger_sha256: string;
    conflict_ledger_sha256: string;
  };
  evidence_items: readonly Record<string, unknown>[];
  conflicts: readonly Record<string, unknown>[];
}

export interface Gate18VerifiedPacket {
  packet: Gate18EvidencePacket;
  packetSha256: string;
  evidenceLedgerSha256: string;
  conflictLedgerSha256: string;
}

type ArtifactReader = (
  pin: Gate18ArtifactPin,
) => Uint8Array;

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const COMMIT_PATTERN = /^[a-f0-9]{40}$/;

const EVIDENCE_FIELDS = [
  "evidence_id",
  "claim_id",
  "claim_metric",
  "value_statement",
  "period",
  "as_of_date",
  "source",
  "root_source_id",
  "independence_group",
  "source_class",
  "claim_fit",
  "source_date",
  "epistemic_type",
  "freshness_state",
  "limitations",
  "conflict_status",
  "used_in",
] as const;

const CONFLICT_FIELDS = [
  "conflict_id",
  "metric_claim",
  "source_a",
  "value_a",
  "source_b",
  "value_b",
  "conflict_type",
  "reason",
  "resolution",
  "resolution_note",
  "materiality",
  "affected_outputs",
] as const;

const FORBIDDEN_PACKET_KEYS = new Set([
  "oqs",
  "ovs",
  "investment_score",
  "valuation_conclusion",
  "next_action",
  "terminal_result",
  "published_v2_conclusion",
  "canonical_snapshot_payload",
]);

function sha256Hex(bytes: Uint8Array | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

function asRecord(
  value: unknown,
  code: string,
): Record<string, unknown> {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    throw new Error(code);
  }

  return value as Record<string, unknown>;
}

function asArray(
  value: unknown,
  code: string,
): unknown[] {
  if (!Array.isArray(value)) {
    throw new Error(code);
  }
  return value;
}

function assertPin(
  company: Gate18PilotCompany,
  pin: Gate18ArtifactPin,
  artifactType: "EVIDENCE_LEDGER" | "CONFLICT_LEDGER",
): void {
  if (pin.repository !== GATE18_PRIVATE_SOURCE_REPOSITORY) {
    throw new Error("VNEXT_GATE18_PRIVATE_REPOSITORY_MISMATCH");
  }
  if (!SHA256_PATTERN.test(pin.sha256)) {
    throw new Error("VNEXT_GATE18_PRIVATE_SHA256_INVALID");
  }
  if (!COMMIT_PATTERN.test(pin.commit_sha)) {
    throw new Error("VNEXT_GATE18_PRIVATE_COMMIT_INVALID");
  }
  if (!Number.isInteger(pin.version) || pin.version < 1) {
    throw new Error("VNEXT_GATE18_PRIVATE_VERSION_INVALID");
  }

  const expectedPrefix =
    `artifacts/orotitan-equity/runs/${company.source_run_id}/research/`;
  if (!pin.path.startsWith(expectedPrefix)) {
    throw new Error("VNEXT_GATE18_PRIVATE_PATH_RUN_MISMATCH");
  }

  if (!pin.path.includes(artifactType)) {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_TYPE_PATH_MISMATCH");
  }
}

function parsePinnedArtifact(
  bytes: Uint8Array,
  pin: Gate18ArtifactPin,
  company: Gate18PilotCompany,
  artifactType: "EVIDENCE_LEDGER" | "CONFLICT_LEDGER",
): Record<string, unknown> {
  const actualSha = sha256Hex(bytes);
  if (actualSha !== pin.sha256) {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_SHA256_MISMATCH");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_JSON_INVALID");
  }

  const record = asRecord(
    parsed,
    "VNEXT_GATE18_PRIVATE_ARTIFACT_OBJECT_REQUIRED",
  );

  if (record.artifact_type !== artifactType) {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_TYPE_MISMATCH");
  }
  if (record.run_id !== company.source_run_id) {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_RUN_MISMATCH");
  }
  if (record.data_cutoff !== company.data_cutoff) {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_CUTOFF_MISMATCH");
  }
  if (record.version !== pin.version) {
    throw new Error("VNEXT_GATE18_PRIVATE_ARTIFACT_VERSION_MISMATCH");
  }

  return record;
}

function projectRecord(
  raw: unknown,
  fields: readonly string[],
): Record<string, unknown> {
  const source = asRecord(
    raw,
    "VNEXT_GATE18_PRIVATE_LEDGER_ITEM_INVALID",
  );
  const projected: Record<string, unknown> = {};

  for (const field of fields) {
    if (Object.prototype.hasOwnProperty.call(source, field)) {
      projected[field] = source[field];
    }
  }

  return projected;
}

function assertNoForbiddenKeys(value: unknown): void {
  if (Array.isArray(value)) {
    for (const item of value) {
      assertNoForbiddenKeys(item);
    }
    return;
  }

  if (value === null || typeof value !== "object") {
    return;
  }

  for (const [key, child] of Object.entries(
    value as Record<string, unknown>,
  )) {
    if (FORBIDDEN_PACKET_KEYS.has(key.toLowerCase())) {
      throw new Error("VNEXT_GATE18_V2_LEAKAGE_FORBIDDEN");
    }
    assertNoForbiddenKeys(child);
  }
}

function sortByStringKey(
  items: readonly Record<string, unknown>[],
  key: string,
): readonly Record<string, unknown>[] {
  return [...items].sort((left, right) =>
    String(left[key] ?? "").localeCompare(
      String(right[key] ?? ""),
    ),
  );
}

export function buildVerifiedGate18EvidencePacket(
  company: Gate18PilotCompany,
  readArtifact: ArtifactReader,
): Gate18VerifiedPacket {
  assertPin(company, company.evidence_ledger, "EVIDENCE_LEDGER");
  assertPin(company, company.conflict_ledger, "CONFLICT_LEDGER");

  const evidenceBytes = readArtifact(company.evidence_ledger);
  const conflictBytes = readArtifact(company.conflict_ledger);

  const evidenceLedger = parsePinnedArtifact(
    evidenceBytes,
    company.evidence_ledger,
    company,
    "EVIDENCE_LEDGER",
  );
  const conflictLedger = parsePinnedArtifact(
    conflictBytes,
    company.conflict_ledger,
    company,
    "CONFLICT_LEDGER",
  );

  const evidenceItems = sortByStringKey(
    asArray(
      evidenceLedger.evidence_items,
      "VNEXT_GATE18_EVIDENCE_ITEMS_REQUIRED",
    ).map((item) => projectRecord(item, EVIDENCE_FIELDS)),
    "evidence_id",
  );

  const conflicts = sortByStringKey(
    asArray(
      conflictLedger.conflicts,
      "VNEXT_GATE18_CONFLICTS_REQUIRED",
    ).map((item) => projectRecord(item, CONFLICT_FIELDS)),
    "conflict_id",
  );

  const packet: Gate18EvidencePacket = {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.1",
    case_id: company.source_run_id,
    display_name: company.display_name,
    role: company.role,
    source_run_id: company.source_run_id,
    data_cutoff: company.data_cutoff,
    source_integrity: {
      evidence_ledger_sha256: company.evidence_ledger.sha256,
      conflict_ledger_sha256: company.conflict_ledger.sha256,
    },
    evidence_items: evidenceItems,
    conflicts,
  };

  assertNoForbiddenKeys(packet);

  return {
    packet,
    packetSha256: sha256Hex(JSON.stringify(packet)),
    evidenceLedgerSha256: sha256Hex(evidenceBytes),
    conflictLedgerSha256: sha256Hex(conflictBytes),
  };
}

export function buildGate18PhaseBModelInput(
  packet: Gate18EvidencePacket,
): string {
  return [
    GATE18_PHASE_B_MODULE_QUESTION,
    "",
    "POINT_IN_TIME_EVIDENCE_PACKET_JSON:",
    JSON.stringify(packet),
  ].join("\n");
}

export function gate18PhaseBPromptTemplateSha256(): string {
  return sha256Hex(
    JSON.stringify({
      system: GATE18_PHASE_B_SYSTEM_PROMPT,
      question: GATE18_PHASE_B_MODULE_QUESTION,
    }),
  );
}

export function gate18PhaseBGenerationSchemaSha256(): string {
  return sha256Hex(
    JSON.stringify(GATE18_PHASE_B_GENERATION_SCHEMA_SPEC),
  );
}

function collectRefs(
  output: Gate18PhaseBOutput,
): {
  evidence: string[];
  conflicts: string[];
} {
  const evidence: string[] = [];
  const conflicts: string[] = [];

  for (const finding of output.findings) {
    evidence.push(
      ...finding.evidence_ids,
      ...finding.counterevidence_ids,
    );
    conflicts.push(...finding.conflict_ids);
  }

  for (const conflict of output.conflicts) {
    evidence.push(...conflict.evidence_ids);
    conflicts.push(conflict.conflict_id);
  }

  for (const candidate of output.weak_link_candidates) {
    evidence.push(...candidate.evidence_ids);
    conflicts.push(...candidate.conflict_ids);
  }

  for (const point of output.unresolved_points) {
    evidence.push(...point.evidence_ids);
    conflicts.push(...point.conflict_ids);
  }

  return { evidence, conflicts };
}

export function assertGate18PhaseBSemantics(
  packet: Gate18EvidencePacket,
  output: Gate18PhaseBOutput,
): void {
  if (output.case_id !== packet.case_id) {
    throw new Error("VNEXT_GATE18_PHASE_B_CASE_MISMATCH");
  }
  if (output.data_cutoff !== packet.data_cutoff) {
    throw new Error("VNEXT_GATE18_PHASE_B_CUTOFF_MISMATCH");
  }

  const evidenceIds = new Set(
    packet.evidence_items.map((item) =>
      String(item.evidence_id ?? ""),
    ),
  );
  const conflictIds = new Set(
    packet.conflicts.map((item) =>
      String(item.conflict_id ?? ""),
    ),
  );

  const refs = collectRefs(output);

  for (const ref of refs.evidence) {
    if (!evidenceIds.has(ref)) {
      throw new Error("VNEXT_GATE18_PHASE_B_UNKNOWN_EVIDENCE_REF");
    }
  }

  for (const ref of refs.conflicts) {
    if (!conflictIds.has(ref)) {
      throw new Error("VNEXT_GATE18_PHASE_B_UNKNOWN_CONFLICT_REF");
    }
  }

  const findingIds = output.findings.map(
    (finding) => finding.finding_id,
  );
  if (new Set(findingIds).size !== findingIds.length) {
    throw new Error("VNEXT_GATE18_PHASE_B_DUPLICATE_FINDING_ID");
  }
}
