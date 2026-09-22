import { createHash } from "node:crypto";

import { z } from "zod";

export const GATE18_CONFLICT_REASONING_PROBE_ID =
  "GATE18_CONFLICT_REASONING_PROBE_V0.1" as const;

export const GATE18_CONFLICT_REASONING_PROMPT_VERSION =
  "0.1.0" as const;

export const GATE18_CONFLICT_REASONING_PROMPT = `
You are performing an OroTitan Gate 18 calibration-only evidence reasoning probe.

Use ONLY the supplied point-in-time RESEARCH EVIDENCE_LEDGER and CONFLICT_LEDGER.
Do not use external knowledge.
Do not infer facts that are not supported by the supplied evidence.
Do not perform valuation.
Do not issue an investment recommendation or portfolio action.
Do not produce OQS, OVS, Investment Score, certification, or OroTitan terminal status.
Do not use model self-confidence as evidence.

Task:
1. Identify up to six material analytical findings where evidence interpretation or conflict handling matters.
2. Every finding must cite one or more existing EVIDENCE_LEDGER IDs.
3. Cite CONFLICT_LEDGER IDs whenever a supplied conflict is relevant.
4. Classify each finding as SUPPORTED, CONTESTED, or UNRESOLVED.
5. Explain the causal reasoning in bounded terms.
6. State the strongest counter-evidence when one exists.
7. Identify the single weakest link in the supplied evidence base.
8. List unresolved questions that materially limit a stronger conclusion.

The output is a calibration artifact only. It is not a canonical OroTitan analytical output.
`.trim();

export const GATE18_PROBE_STATUS = [
  "SUPPORTED",
  "CONTESTED",
  "UNRESOLVED",
] as const;

export const gate18ConflictReasoningOutputSchema = z.object({
  probeId: z.literal(GATE18_CONFLICT_REASONING_PROBE_ID),
  findings: z
    .array(
      z.object({
        findingId: z.string().min(1),
        status: z.enum(GATE18_PROBE_STATUS),
        claim: z.string().min(1),
        evidenceIds: z.array(z.string().min(1)).min(1),
        conflictIds: z.array(z.string().min(1)),
        causalReasoning: z.string().min(1),
        strongestCounterEvidence: z.string().min(1).nullable(),
      }),
    )
    .min(1)
    .max(6),
  weakestLink: z.object({
    claim: z.string().min(1),
    evidenceIds: z.array(z.string().min(1)).min(1),
    conflictIds: z.array(z.string().min(1)),
    rationale: z.string().min(1),
  }),
  unresolvedQuestions: z
    .array(
      z.object({
        question: z.string().min(1),
        evidenceIds: z.array(z.string().min(1)),
        conflictIds: z.array(z.string().min(1)),
      }),
    )
    .max(6),
});

export type Gate18ConflictReasoningOutput = z.infer<
  typeof gate18ConflictReasoningOutputSchema
>;

export const GATE18_CONFLICT_REASONING_SCHEMA_DESCRIPTOR = {
  probeId: GATE18_CONFLICT_REASONING_PROBE_ID,
  version: "0.1.0",
  fields: [
    "probeId",
    "findings[].findingId",
    "findings[].status",
    "findings[].claim",
    "findings[].evidenceIds[]",
    "findings[].conflictIds[]",
    "findings[].causalReasoning",
    "findings[].strongestCounterEvidence",
    "weakestLink.claim",
    "weakestLink.evidenceIds[]",
    "weakestLink.conflictIds[]",
    "weakestLink.rationale",
    "unresolvedQuestions[].question",
    "unresolvedQuestions[].evidenceIds[]",
    "unresolvedQuestions[].conflictIds[]",
  ],
} as const;

export interface Gate18ProbeSourceIdentity {
  runId: string;
  dataCutoff: string;
  evidenceIds: ReadonlySet<string>;
  conflictIds: ReadonlySet<string>;
}

export interface Gate18ProbeSemanticValidation {
  valid: boolean;
  issues: readonly string[];
}

export function sha256Utf8(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export const GATE18_CONFLICT_REASONING_PROMPT_SHA256 =
  sha256Utf8(GATE18_CONFLICT_REASONING_PROMPT);

export const GATE18_CONFLICT_REASONING_SCHEMA_SHA256 =
  sha256Utf8(
    JSON.stringify(GATE18_CONFLICT_REASONING_SCHEMA_DESCRIPTOR),
  );

function assertRecord(
  value: unknown,
  errorCode: string,
): asserts value is Record<string, unknown> {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    throw new Error(errorCode);
  }
}

function stringField(
  value: Record<string, unknown>,
  key: string,
  errorCode: string,
): string {
  const candidate = value[key];

  if (
    typeof candidate !== "string" ||
    candidate.trim().length === 0
  ) {
    throw new Error(errorCode);
  }

  return candidate;
}

function collectIds(
  value: Record<string, unknown>,
  key: string,
  errorCode: string,
): ReadonlySet<string> {
  const candidate = value[key];

  if (!Array.isArray(candidate)) {
    throw new Error(errorCode);
  }

  const ids = new Set<string>();

  for (const item of candidate) {
    assertRecord(item, errorCode);
    const id = stringField(item, "id", errorCode);

    if (ids.has(id)) {
      throw new Error(errorCode);
    }

    ids.add(id);
  }

  return ids;
}

export function parseGate18ProbeSources(
  evidenceLedgerRaw: string,
  conflictLedgerRaw: string,
): Gate18ProbeSourceIdentity {
  const evidenceUnknown: unknown = JSON.parse(evidenceLedgerRaw);
  const conflictUnknown: unknown = JSON.parse(conflictLedgerRaw);

  assertRecord(
    evidenceUnknown,
    "VNEXT_GATE18_EVIDENCE_LEDGER_INVALID",
  );
  assertRecord(
    conflictUnknown,
    "VNEXT_GATE18_CONFLICT_LEDGER_INVALID",
  );

  const evidenceRunId = stringField(
    evidenceUnknown,
    "run_id",
    "VNEXT_GATE18_EVIDENCE_RUN_ID_INVALID",
  );
  const conflictRunId = stringField(
    conflictUnknown,
    "run_id",
    "VNEXT_GATE18_CONFLICT_RUN_ID_INVALID",
  );

  if (evidenceRunId !== conflictRunId) {
    throw new Error("VNEXT_GATE18_SOURCE_RUN_ID_MISMATCH");
  }

  const evidenceCutoff = stringField(
    evidenceUnknown,
    "data_cutoff",
    "VNEXT_GATE18_EVIDENCE_DATA_CUTOFF_INVALID",
  );
  const conflictCutoff = stringField(
    conflictUnknown,
    "data_cutoff",
    "VNEXT_GATE18_CONFLICT_DATA_CUTOFF_INVALID",
  );

  if (evidenceCutoff !== conflictCutoff) {
    throw new Error("VNEXT_GATE18_SOURCE_DATA_CUTOFF_MISMATCH");
  }

  return {
    runId: evidenceRunId,
    dataCutoff: evidenceCutoff,
    evidenceIds: collectIds(
      evidenceUnknown,
      "items",
      "VNEXT_GATE18_EVIDENCE_ITEMS_INVALID",
    ),
    conflictIds: collectIds(
      conflictUnknown,
      "conflicts",
      "VNEXT_GATE18_CONFLICT_ITEMS_INVALID",
    ),
  };
}

function pushUnknownRefs(
  refs: readonly string[],
  allowed: ReadonlySet<string>,
  prefix: string,
  issues: string[],
): void {
  for (const ref of refs) {
    if (!allowed.has(ref)) {
      issues.push(`${prefix}:${ref}`);
    }
  }
}

export function validateGate18ProbeReferences(
  output: Gate18ConflictReasoningOutput,
  sources: Gate18ProbeSourceIdentity,
): Gate18ProbeSemanticValidation {
  const issues: string[] = [];
  const findingIds = new Set<string>();

  for (const finding of output.findings) {
    if (findingIds.has(finding.findingId)) {
      issues.push(
        `DUPLICATE_FINDING_ID:${finding.findingId}`,
      );
    }

    findingIds.add(finding.findingId);

    pushUnknownRefs(
      finding.evidenceIds,
      sources.evidenceIds,
      "UNKNOWN_EVIDENCE_ID",
      issues,
    );
    pushUnknownRefs(
      finding.conflictIds,
      sources.conflictIds,
      "UNKNOWN_CONFLICT_ID",
      issues,
    );
  }

  pushUnknownRefs(
    output.weakestLink.evidenceIds,
    sources.evidenceIds,
    "UNKNOWN_EVIDENCE_ID",
    issues,
  );
  pushUnknownRefs(
    output.weakestLink.conflictIds,
    sources.conflictIds,
    "UNKNOWN_CONFLICT_ID",
    issues,
  );

  for (const question of output.unresolvedQuestions) {
    pushUnknownRefs(
      question.evidenceIds,
      sources.evidenceIds,
      "UNKNOWN_EVIDENCE_ID",
      issues,
    );
    pushUnknownRefs(
      question.conflictIds,
      sources.conflictIds,
      "UNKNOWN_CONFLICT_ID",
      issues,
    );
  }

  return {
    valid: issues.length === 0,
    issues,
  };
}

export function buildGate18ConflictReasoningUserPrompt(
  displayName: string,
  dataCutoff: string,
  evidenceLedgerRaw: string,
  conflictLedgerRaw: string,
): string {
  return [
    `COMPANY: ${displayName}`,
    `DATA_CUTOFF: ${dataCutoff}`,
    "",
    "RESEARCH EVIDENCE_LEDGER:",
    evidenceLedgerRaw,
    "",
    "RESEARCH CONFLICT_LEDGER:",
    conflictLedgerRaw,
  ].join("\n");
}
