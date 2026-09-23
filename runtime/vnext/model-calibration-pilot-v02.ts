import { createHash } from "node:crypto";

import { z } from "zod";

import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "./model-calibration-pilot";

export const GATE18_PHASE_B_V02_SCOPE =
  "MOAT_INPUTS" as const;

export const GATE18_PHASE_B_V02_MODULE_ID =
  "MOAT_EVIDENCE_AUDIT_ASSISTED_V0_2" as const;

export const GATE18_PHASE_B_V02_PROMPT_TEMPLATE_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_V0_2" as const;

export const GATE18_PHASE_B_V02_PROMPT_TEMPLATE_VERSION =
  "0.2" as const;

export const GATE18_PHASE_B_V02_GENERATION_SCHEMA_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_2" as const;

export const GATE18_PHASE_B_V02_GENERATION_SCHEMA_VERSION =
  "0.2" as const;

export const GATE18_PHASE_B_V02_MAX_OUTPUT_TOKENS = 1536;

export const GATE18_PHASE_B_V02_SYSTEM_PROMPT = [
  "You are an OroTitan calibration model operating on a bounded point-in-time Research evidence packet scoped to MOAT_INPUTS.",
  "Use only the supplied packet. Do not browse, call tools, use outside facts, or fill gaps from memory.",
  "This is an ASSIST-only evidence-audit task.",
  "Do not render a final moat mechanism judgment, moat durability judgment, runway judgment, valuation conclusion, OQS, OVS, Investment Score, next action, publication decision, or investment conclusion.",
  "Every substantive item must cite exact E-* evidence IDs and, where applicable, exact C-* conflict IDs present in the packet.",
  "Treat unresolved conflicts as unresolved.",
  "Prefer concise, high-materiality outputs over exhaustive narrative.",
  "Return only the requested structured output.",
].join(" ");

export const GATE18_PHASE_B_V02_MODULE_QUESTION = [
  "Audit the supplied MOAT_INPUTS evidence packet.",
  "Return only the three most decision-relevant evidence findings, up to two material conflicts, up to two candidate weak links, and up to three unresolved questions.",
  "Anchor every item to exact packet IDs.",
  "This is evidence triage for human adjudication, not a final moat conclusion.",
].join(" ");

const evidenceIdSchema = z
  .string()
  .regex(/^E-/)
  .max(12);

const conflictIdSchema = z
  .string()
  .regex(/^C-/)
  .max(12);

export const gate18PhaseBV02OutputSchema = z.object({
  case_id: z.string().min(1).max(64),
  data_cutoff: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  priority_findings: z
    .array(
      z.object({
        claim: z.string().min(1).max(120),
        support_state: z.enum([
          "SUPPORTED",
          "MIXED",
          "UNRESOLVED",
        ]),
        evidence_ids: z.array(evidenceIdSchema).min(1).max(3),
        conflict_ids: z.array(conflictIdSchema).max(2),
        causal_link: z.string().min(1).max(120),
        counterevidence_ids: z.array(evidenceIdSchema).max(2),
      }),
    )
    .max(3),
  material_conflicts: z
    .array(
      z.object({
        conflict_id: conflictIdSchema,
        implication: z.string().min(1).max(120),
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
        candidate: z.string().min(1).max(100),
        evidence_ids: z.array(evidenceIdSchema).min(1).max(3),
        conflict_ids: z.array(conflictIdSchema).max(2),
        why_uncertain: z.string().min(1).max(120),
      }),
    )
    .max(2),
  unresolved_points: z
    .array(
      z.object({
        question: z.string().min(1).max(120),
        evidence_ids: z.array(evidenceIdSchema).max(3),
        conflict_ids: z.array(conflictIdSchema).max(2),
      }),
    )
    .max(3),
});

export type Gate18PhaseBV02Output = z.infer<
  typeof gate18PhaseBV02OutputSchema
>;

export const GATE18_PHASE_B_V02_GENERATION_SCHEMA_SPEC = {
  type: "object",
  additionalProperties: false,
  properties: {
    case_id: {
      type: "string",
      minLength: 1,
      maxLength: 64,
    },
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
          claim: {
            type: "string",
            minLength: 1,
            maxLength: 120,
          },
          support_state: {
            type: "string",
            enum: ["SUPPORTED", "MIXED", "UNRESOLVED"],
          },
          evidence_ids: {
            type: "array",
            minItems: 1,
            maxItems: 3,
            items: {
              type: "string",
              pattern: "^E-",
              maxLength: 12,
            },
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: {
              type: "string",
              pattern: "^C-",
              maxLength: 12,
            },
          },
          causal_link: {
            type: "string",
            minLength: 1,
            maxLength: 120,
          },
          counterevidence_ids: {
            type: "array",
            maxItems: 2,
            items: {
              type: "string",
              pattern: "^E-",
              maxLength: 12,
            },
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
          conflict_id: {
            type: "string",
            pattern: "^C-",
            maxLength: 12,
          },
          implication: {
            type: "string",
            minLength: 1,
            maxLength: 120,
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
            maxLength: 100,
          },
          evidence_ids: {
            type: "array",
            minItems: 1,
            maxItems: 3,
            items: {
              type: "string",
              pattern: "^E-",
              maxLength: 12,
            },
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: {
              type: "string",
              pattern: "^C-",
              maxLength: 12,
            },
          },
          why_uncertain: {
            type: "string",
            minLength: 1,
            maxLength: 120,
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
            maxLength: 120,
          },
          evidence_ids: {
            type: "array",
            maxItems: 3,
            items: {
              type: "string",
              pattern: "^E-",
              maxLength: 12,
            },
          },
          conflict_ids: {
            type: "array",
            maxItems: 2,
            items: {
              type: "string",
              pattern: "^C-",
              maxLength: 12,
            },
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

export interface Gate18V02EvidenceItem {
  evidence_id: string;
  claim_id: string | null;
  claim: string | null;
  value: string;
  period: string | null;
  source_refs: string[];
  source_class: string | null;
  claim_fit: string | null;
  epistemic_type: string | null;
  freshness_state: string | null;
  limitations: string | null;
  conflict_status: string | null;
  module_tags: string[];
}

export interface Gate18V02Conflict {
  conflict_id: string;
  metric_claim: string | null;
  value_a: string | null;
  value_b: string | null;
  conflict_type: string | null;
  reason: string | null;
  resolution: string | null;
  resolution_note: string | null;
  materiality: string | null;
  evidence_refs: string[];
  affected_outputs: string[];
}

export interface Gate18V02EvidencePacket {
  format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2";
  scope: typeof GATE18_PHASE_B_V02_SCOPE;
  case_id: string;
  display_name: string;
  role: string;
  source_run_id: string;
  data_cutoff: string;
  source_integrity: {
    evidence_ledger_sha256: string;
    conflict_ledger_sha256: string;
  };
  evidence_items: Gate18V02EvidenceItem[];
  conflicts: Gate18V02Conflict[];
}

export interface Gate18V02VerifiedPacket {
  packet: Gate18V02EvidencePacket;
  packetSha256: string;
  evidenceLedgerSha256: string;
  conflictLedgerSha256: string;
  sourceEvidenceCount: number;
  sourceConflictCount: number;
}

type ArtifactReader = (
  pin: Gate18ArtifactPin,
) => Uint8Array;

const SHA256_PATTERN = /^[a-f0-9]{64}$/;
const COMMIT_PATTERN = /^[a-f0-9]{40}$/;
const EVIDENCE_REF_PATTERN = /E-\d+/g;

const MODULE_ALIASES: Readonly<Record<string, string>> = {
  MOAT: "MOAT_INPUTS",
  RUNWAY: "RUNWAY_INPUTS",
  RETURN_QUALITY: "RETURN_QUALITY_INPUTS",
  RISK: "RISK_RESILIENCE_INPUTS",
  RISK_RESILIENCE: "RISK_RESILIENCE_INPUTS",
  BUSINESS_MODEL: "BUSINESS_MODEL_INPUTS",
  ECONOMIC_QUALITY: "ECONOMIC_QUALITY_INPUTS",
  CAPITAL_ALLOCATION: "CAPITAL_ALLOCATION_INPUTS",
  VALUATION: "VALUATION_INPUTS",
  FCF: "FCF_FORENSIC_INPUTS",
};

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

function stableValue(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (
    value === null ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return JSON.stringify(value.map((item) => {
      if (
        item !== null &&
        typeof item === "object" &&
        !Array.isArray(item)
      ) {
        return Object.fromEntries(
          Object.entries(item as Record<string, unknown>)
            .sort(([left], [right]) =>
              left.localeCompare(right),
            ),
        );
      }
      return item;
    }));
  }
  if (typeof value === "object") {
    return JSON.stringify(
      Object.fromEntries(
        Object.entries(value as Record<string, unknown>)
          .sort(([left], [right]) =>
            left.localeCompare(right),
          ),
      ),
    );
  }
  return String(value ?? "");
}

function optionalString(value: unknown): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "string") {
    return value;
  }
  return stableValue(value);
}

function stringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item));
  }
  if (value === null || value === undefined) {
    return [];
  }
  return [String(value)];
}

function normalizeModuleTag(tag: string): string {
  return MODULE_ALIASES[tag] ?? tag;
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort((left, right) =>
    left.localeCompare(right),
  );
}

function assertPin(
  company: Gate18PilotCompany,
  pin: Gate18ArtifactPin,
  artifactType: "EVIDENCE_LEDGER" | "CONFLICT_LEDGER",
): void {
  if (pin.repository !== "robzer13/real-orotitan") {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_REPOSITORY_MISMATCH");
  }
  if (!SHA256_PATTERN.test(pin.sha256)) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_SHA256_INVALID");
  }
  if (!COMMIT_PATTERN.test(pin.commit_sha)) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_COMMIT_INVALID");
  }
  if (!Number.isInteger(pin.version) || pin.version < 1) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_VERSION_INVALID");
  }

  const expectedPrefix =
    `artifacts/orotitan-equity/runs/${company.source_run_id}/research/`;
  if (!pin.path.startsWith(expectedPrefix)) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_PATH_RUN_MISMATCH");
  }
  if (!pin.path.includes(artifactType)) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_TYPE_PATH_MISMATCH");
  }
}

function parsePinnedArtifact(
  bytes: Uint8Array,
  pin: Gate18ArtifactPin,
  company: Gate18PilotCompany,
  artifactType: "EVIDENCE_LEDGER" | "CONFLICT_LEDGER",
): Record<string, unknown> {
  if (sha256Hex(bytes) !== pin.sha256) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_SHA256_MISMATCH");
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_JSON_INVALID");
  }

  const record = asRecord(
    parsed,
    "VNEXT_GATE18_V02_PRIVATE_ARTIFACT_OBJECT_REQUIRED",
  );

  if (record.artifact_type !== artifactType) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_TYPE_MISMATCH");
  }
  if (record.run_id !== company.source_run_id) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_RUN_MISMATCH");
  }
  if (record.data_cutoff !== company.data_cutoff) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_CUTOFF_MISMATCH");
  }
  if (record.version !== pin.version) {
    throw new Error("VNEXT_GATE18_V02_PRIVATE_ARTIFACT_VERSION_MISMATCH");
  }

  return record;
}

function selectUniqueArray(
  record: Record<string, unknown>,
  keys: readonly string[],
  code: string,
): unknown[] {
  const candidates = keys
    .map((key) => ({ key, value: record[key] }))
    .filter(({ value }) => Array.isArray(value));

  if (candidates.length !== 1) {
    throw new Error(code);
  }

  return candidates[0].value as unknown[];
}

function normalizeEvidenceItem(
  raw: unknown,
): Gate18V02EvidenceItem {
  const item = asRecord(
    raw,
    "VNEXT_GATE18_V02_EVIDENCE_ITEM_INVALID",
  );

  const evidenceId =
    optionalString(item.evidence_id) ??
    optionalString(item.id);

  if (!evidenceId || !/^E-/.test(evidenceId)) {
    throw new Error("VNEXT_GATE18_V02_EVIDENCE_ID_INVALID");
  }

  const valueSource =
    item.value_statement ??
    item.value ??
    item.claim ??
    item.claim_metric;

  if (valueSource === undefined) {
    throw new Error("VNEXT_GATE18_V02_EVIDENCE_VALUE_MISSING");
  }

  const sourceRefs = uniqueSorted([
    ...stringArray(item.source),
    ...stringArray(item.source_id),
    ...stringArray(item.sources),
  ]);

  const moduleTags = uniqueSorted([
    ...stringArray(item.used_in),
    ...stringArray(item.blocks),
  ].map(normalizeModuleTag));

  return {
    evidence_id: evidenceId,
    claim_id: optionalString(item.claim_id),
    claim:
      optionalString(item.claim_metric) ??
      optionalString(item.claim),
    value: stableValue(valueSource),
    period:
      optionalString(item.period) ??
      optionalString(item.period_as_of_date) ??
      optionalString(item.as_of_date),
    source_refs: sourceRefs,
    source_class: optionalString(item.source_class),
    claim_fit: optionalString(item.claim_fit),
    epistemic_type: optionalString(item.epistemic_type),
    freshness_state: optionalString(item.freshness_state),
    limitations: optionalString(item.limitations),
    conflict_status:
      optionalString(item.conflict_status) ??
      optionalString(item.conflict),
    module_tags: moduleTags,
  };
}

function extractEvidenceRefs(
  item: Record<string, unknown>,
): string[] {
  const explicit = [
    ...stringArray(item.evidence_id_a),
    ...stringArray(item.evidence_id_b),
  ].filter((value) => /^E-/.test(value));

  const embedded = [
    optionalString(item.source_a),
    optionalString(item.source_b),
  ]
    .filter((value): value is string => value !== null)
    .flatMap((value) => value.match(EVIDENCE_REF_PATTERN) ?? []);

  return uniqueSorted([...explicit, ...embedded]);
}

function normalizeConflict(
  raw: unknown,
): Gate18V02Conflict {
  const item = asRecord(
    raw,
    "VNEXT_GATE18_V02_CONFLICT_ITEM_INVALID",
  );

  const conflictId =
    optionalString(item.conflict_id) ??
    optionalString(item.id);

  if (!conflictId || !/^C-/.test(conflictId)) {
    throw new Error("VNEXT_GATE18_V02_CONFLICT_ID_INVALID");
  }

  return {
    conflict_id: conflictId,
    metric_claim: optionalString(item.metric_claim),
    value_a: optionalString(item.value_a),
    value_b: optionalString(item.value_b),
    conflict_type: optionalString(item.conflict_type),
    reason: optionalString(item.reason),
    resolution: optionalString(item.resolution),
    resolution_note: optionalString(item.resolution_note),
    materiality: optionalString(item.materiality),
    evidence_refs: extractEvidenceRefs(item),
    affected_outputs: uniqueSorted(
      stringArray(item.affected_outputs).map(normalizeModuleTag),
    ),
  };
}

export function buildVerifiedGate18V02MoatPacket(
  company: Gate18PilotCompany,
  readArtifact: ArtifactReader,
): Gate18V02VerifiedPacket {
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

  const allEvidence = selectUniqueArray(
    evidenceLedger,
    ["evidence_items", "items", "evidence"],
    "VNEXT_GATE18_V02_EVIDENCE_ARRAY_AMBIGUOUS_OR_MISSING",
  ).map(normalizeEvidenceItem);

  const allConflicts = selectUniqueArray(
    conflictLedger,
    ["conflicts"],
    "VNEXT_GATE18_V02_CONFLICT_ARRAY_AMBIGUOUS_OR_MISSING",
  ).map(normalizeConflict);

  const scopedConflicts = allConflicts.filter((conflict) =>
    conflict.affected_outputs.includes(
      GATE18_PHASE_B_V02_SCOPE,
    ),
  );

  const requiredEvidenceIds = new Set<string>();
  for (const item of allEvidence) {
    if (
      item.module_tags.includes(GATE18_PHASE_B_V02_SCOPE) ||
      item.module_tags.includes("ALL_DD_INPUTS")
    ) {
      requiredEvidenceIds.add(item.evidence_id);
    }
  }
  for (const conflict of scopedConflicts) {
    for (const evidenceId of conflict.evidence_refs) {
      requiredEvidenceIds.add(evidenceId);
    }
  }

  const scopedEvidence = allEvidence
    .filter((item) =>
      requiredEvidenceIds.has(item.evidence_id),
    )
    .sort((left, right) =>
      left.evidence_id.localeCompare(right.evidence_id),
    );

  const scopedEvidenceIds = new Set(
    scopedEvidence.map((item) => item.evidence_id),
  );

  for (const conflict of scopedConflicts) {
    for (const evidenceId of conflict.evidence_refs) {
      if (!scopedEvidenceIds.has(evidenceId)) {
        throw new Error(
          "VNEXT_GATE18_V02_CONFLICT_EVIDENCE_REF_MISSING",
        );
      }
    }
  }

  if (scopedEvidence.length === 0) {
    throw new Error(
      "VNEXT_GATE18_V02_MOAT_SCOPE_HAS_NO_EVIDENCE",
    );
  }

  const packet: Gate18V02EvidencePacket = {
    format: "OROTITAN_GATE18_EVIDENCE_PACKET_V0.2",
    scope: GATE18_PHASE_B_V02_SCOPE,
    case_id: company.source_run_id,
    display_name: company.display_name,
    role: company.role,
    source_run_id: company.source_run_id,
    data_cutoff: company.data_cutoff,
    source_integrity: {
      evidence_ledger_sha256: company.evidence_ledger.sha256,
      conflict_ledger_sha256: company.conflict_ledger.sha256,
    },
    evidence_items: scopedEvidence,
    conflicts: scopedConflicts.sort((left, right) =>
      left.conflict_id.localeCompare(right.conflict_id),
    ),
  };

  return {
    packet,
    packetSha256: sha256Hex(JSON.stringify(packet)),
    evidenceLedgerSha256: sha256Hex(evidenceBytes),
    conflictLedgerSha256: sha256Hex(conflictBytes),
    sourceEvidenceCount: allEvidence.length,
    sourceConflictCount: allConflicts.length,
  };
}

export function buildGate18PhaseBV02ModelInput(
  packet: Gate18V02EvidencePacket,
): string {
  return [
    GATE18_PHASE_B_V02_MODULE_QUESTION,
    "",
    "POINT_IN_TIME_MOAT_EVIDENCE_PACKET_JSON:",
    JSON.stringify(packet),
  ].join("\n");
}

export function gate18PhaseBV02PromptTemplateSha256(): string {
  return sha256Hex(
    JSON.stringify({
      system: GATE18_PHASE_B_V02_SYSTEM_PROMPT,
      question: GATE18_PHASE_B_V02_MODULE_QUESTION,
    }),
  );
}

export function gate18PhaseBV02GenerationSchemaSha256(): string {
  return sha256Hex(
    JSON.stringify(
      GATE18_PHASE_B_V02_GENERATION_SCHEMA_SPEC,
    ),
  );
}

function collectRefs(
  output: Gate18PhaseBV02Output,
): {
  evidence: string[];
  conflicts: string[];
} {
  const evidence: string[] = [];
  const conflicts: string[] = [];

  for (const finding of output.priority_findings) {
    evidence.push(
      ...finding.evidence_ids,
      ...finding.counterevidence_ids,
    );
    conflicts.push(...finding.conflict_ids);
  }
  for (const conflict of output.material_conflicts) {
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

export function assertGate18PhaseBV02Semantics(
  packet: Gate18V02EvidencePacket,
  output: Gate18PhaseBV02Output,
): void {
  if (output.case_id !== packet.case_id) {
    throw new Error("VNEXT_GATE18_V02_CASE_MISMATCH");
  }
  if (output.data_cutoff !== packet.data_cutoff) {
    throw new Error("VNEXT_GATE18_V02_CUTOFF_MISMATCH");
  }

  const evidenceIds = new Set(
    packet.evidence_items.map((item) => item.evidence_id),
  );
  const conflictIds = new Set(
    packet.conflicts.map((item) => item.conflict_id),
  );
  const refs = collectRefs(output);

  for (const evidenceId of refs.evidence) {
    if (!evidenceIds.has(evidenceId)) {
      throw new Error(
        "VNEXT_GATE18_V02_UNKNOWN_EVIDENCE_REF",
      );
    }
  }
  for (const conflictId of refs.conflicts) {
    if (!conflictIds.has(conflictId)) {
      throw new Error(
        "VNEXT_GATE18_V02_UNKNOWN_CONFLICT_REF",
      );
    }
  }
}

export function buildGate18PhaseBV02MaximalOutputFixture(
  caseId: string,
  dataCutoff: string,
): Gate18PhaseBV02Output {
  const evidenceIds = ["E-999999999", "E-888888888", "E-777777777"];
  const conflictIds = ["C-999999999", "C-888888888"];
  const max120 = "x".repeat(120);
  const max100 = "x".repeat(100);

  return {
    case_id: caseId.slice(0, 64),
    data_cutoff: dataCutoff,
    priority_findings: Array.from({ length: 3 }, () => ({
      claim: max120,
      support_state: "UNRESOLVED" as const,
      evidence_ids: evidenceIds,
      conflict_ids: conflictIds,
      causal_link: max120,
      counterevidence_ids: evidenceIds.slice(0, 2),
    })),
    material_conflicts: Array.from({ length: 2 }, () => ({
      conflict_id: conflictIds[0],
      implication: max120,
      resolution_state: "UNRESOLVED_IN_PACKET" as const,
    })),
    weak_link_candidates: Array.from({ length: 2 }, () => ({
      candidate: max100,
      evidence_ids: evidenceIds,
      conflict_ids: conflictIds,
      why_uncertain: max120,
    })),
    unresolved_points: Array.from({ length: 3 }, () => ({
      question: max120,
      evidence_ids: evidenceIds,
      conflict_ids: conflictIds,
    })),
  };
}
