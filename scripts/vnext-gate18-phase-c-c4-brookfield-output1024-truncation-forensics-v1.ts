import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const EXPECTED_MATRIX_CELL_ID =
  "C4_BROOKFIELD_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_BROOKFIELD_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1024_TIMEOUT480_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "Brookfield Corporation";
const EXPECTED_DONE_REASON = "length";
const EXPECTED_EVAL_COUNT = 1024;
const EXPECTED_MAX_OUTPUT_TOKENS = 1024;
const EXPECTED_SCHEMA_ERROR_PREFIX =
  "Unterminated string in JSON at position 3764";

interface CliOptions {
  privateOutputPath: string;
}

interface PrivateRunArtifact {
  status?: string;
  invocation?: {
    matrixCellId?: string;
    attemptId?: string;
    company?: string;
  };
  localRuntime?: {
    generation?: {
      maxOutputTokens?: number;
      clientTimeoutMs?: number;
    };
  };
  execution?: {
    wallClockMs?: number;
    doneReason?: string | null;
    promptEvalCount?: number | null;
    evalCount?: number | null;
    runtimeError?: string | null;
    schemaValid?: boolean;
    schemaError?: string | null;
    semanticValid?: boolean;
    semanticError?: string | null;
  };
  response?: {
    rawText?: string | null;
    parsedJson?: unknown;
  };
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateOutputPath = "";

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--private-output-path") {
      privateOutputPath = argv[++index] ?? "";
      continue;
    }
    throw new Error(
      `VNEXT_GATE18_C4_BROOKFIELD_TRUNCATION_FORENSICS_UNKNOWN_ARG:${arg}`,
    );
  }

  if (!privateOutputPath.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_TRUNCATION_FORENSICS_PRIVATE_OUTPUT_REQUIRED",
    );
  }

  return { privateOutputPath };
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function countPattern(value: string, pattern: RegExp): number {
  return Array.from(value.matchAll(pattern)).length;
}

function lineAndColumnAt(value: string, index: number): {
  line: number;
  column: number;
} {
  const safe = Math.max(0, Math.min(index, value.length));
  const prefix = value.slice(0, safe);
  const lines = prefix.split("\n");
  return {
    line: lines.length,
    column: (lines.at(-1)?.length ?? 0) + 1,
  };
}

function scanTerminalStructure(value: string): {
  inString: boolean;
  escapePending: boolean;
  objectDepth: number;
  arrayDepth: number;
} {
  let inString = false;
  let escapePending = false;
  let objectDepth = 0;
  let arrayDepth = 0;

  for (const char of value) {
    if (inString) {
      if (escapePending) {
        escapePending = false;
        continue;
      }
      if (char === "\\") {
        escapePending = true;
        continue;
      }
      if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }
    if (char === "{") objectDepth += 1;
    if (char === "}") objectDepth -= 1;
    if (char === "[") arrayDepth += 1;
    if (char === "]") arrayDepth -= 1;
  }

  return {
    inString,
    escapePending,
    objectDepth,
    arrayDepth,
  };
}

function marker(value: string, key: string) {
  const literal = `"${key}"`;
  const index = value.indexOf(literal);
  return {
    key,
    present: index >= 0,
    index: index >= 0 ? index : null,
    location:
      index >= 0 ? lineAndColumnAt(value, index) : null,
  };
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const absolute = resolve(options.privateOutputPath);

  if (!existsSync(absolute)) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_TRUNCATION_FORENSICS_PRIVATE_OUTPUT_NOT_FOUND",
    );
  }

  const run = JSON.parse(
    readFileSync(absolute, "utf8"),
  ) as PrivateRunArtifact;

  if (
    run.status !== "FAIL" ||
    run.invocation?.matrixCellId !== EXPECTED_MATRIX_CELL_ID ||
    run.invocation?.attemptId !== EXPECTED_ATTEMPT_ID ||
    run.invocation?.company !== EXPECTED_COMPANY ||
    run.execution?.doneReason !== EXPECTED_DONE_REASON ||
    run.execution?.evalCount !== EXPECTED_EVAL_COUNT ||
    run.localRuntime?.generation?.maxOutputTokens !==
      EXPECTED_MAX_OUTPUT_TOKENS ||
    run.execution?.runtimeError !== null ||
    run.execution?.schemaValid !== false ||
    !(run.execution?.schemaError ?? "").startsWith(
      EXPECTED_SCHEMA_ERROR_PREFIX,
    )
  ) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_TRUNCATION_FORENSICS_RUN_IDENTITY_MISMATCH",
    );
  }

  const rawText = run.response?.rawText;
  if (typeof rawText !== "string" || rawText.length === 0) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_TRUNCATION_FORENSICS_RAW_TEXT_MISSING",
    );
  }

  const sectionKeys = [
    "case_id",
    "data_cutoff",
    "priority_findings",
    "material_conflicts",
    "weak_link_candidates",
    "unresolved_points",
  ];
  const sectionMarkers = sectionKeys.map((key) =>
    marker(rawText, key),
  );
  const lastPresentSection =
    [...sectionMarkers]
      .filter((item) => item.present)
      .sort(
        (a, b) =>
          (b.index ?? -1) - (a.index ?? -1),
      )[0] ?? null;

  const parsePositionMatch =
    (run.execution?.schemaError ?? "").match(
      /position\s+(\d+)/i,
    );
  const parsePosition = parsePositionMatch
    ? Number(parsePositionMatch[1])
    : null;

  const wallClockMs = run.execution?.wallClockMs ?? null;
  const evalCount = run.execution?.evalCount ?? null;
  const wallClockPerGeneratedTokenUpperBoundMs =
    typeof wallClockMs === "number" &&
    typeof evalCount === "number" &&
    evalCount > 0
      ? wallClockMs / evalCount
      : null;

  const projections =
    wallClockPerGeneratedTokenUpperBoundMs === null
      ? null
      : [1152, 1280, 1536].map((tokens) => ({
          maxOutputTokens: tokens,
          linearUpperBoundWallClockMs: Math.ceil(
            wallClockPerGeneratedTokenUpperBoundMs * tokens,
          ),
          note:
            "Diagnostic upper bound only; prompt/load time is partly fixed, so actual runtime need not scale linearly.",
        }));

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1024_TRUNCATION_FORENSICS_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C4_OUTPUT_BUDGET_FORENSICS",
        status:
          "PASS_TRUNCATION_STRUCTURE_MEASURED_NO_INFERENCE",
        mode: "LOCAL_READ_ONLY_NO_INFERENCE",
        sourceRun: {
          matrixCellId: EXPECTED_MATRIX_CELL_ID,
          attemptId: EXPECTED_ATTEMPT_ID,
          company: EXPECTED_COMPANY,
          wallClockMs,
          doneReason: run.execution?.doneReason ?? null,
          promptEvalCount:
            run.execution?.promptEvalCount ?? null,
          evalCount,
          maxOutputTokens:
            run.localRuntime?.generation?.maxOutputTokens ??
            null,
          clientTimeoutMs:
            run.localRuntime?.generation?.clientTimeoutMs ??
            null,
          runtimeError: run.execution?.runtimeError ?? null,
          schemaValid: run.execution?.schemaValid ?? null,
          schemaError: run.execution?.schemaError ?? null,
        },
        rawOutput: {
          chars: rawText.length,
          bytes: Buffer.byteLength(rawText, "utf8"),
          sha256: sha256(rawText),
          parseErrorPosition: parsePosition,
          parseErrorLocation:
            parsePosition === null
              ? null
              : lineAndColumnAt(rawText, parsePosition),
          terminalStructure: scanTerminalStructure(rawText),
        },
        structuralProgress: {
          sectionMarkers,
          lastPresentSection:
            lastPresentSection?.key ?? null,
          priorityFindingClaimCount: countPattern(
            rawText,
            /"claim"\s*:/g,
          ),
          priorityFindingCausalLinkCount: countPattern(
            rawText,
            /"causal_link"\s*:/g,
          ),
          materialConflictItemCount: countPattern(
            rawText,
            /"conflict_id"\s*:/g,
          ),
          weakLinkCandidateCount: countPattern(
            rawText,
            /"candidate"\s*:/g,
          ),
          unresolvedQuestionCount: countPattern(
            rawText,
            /"question"\s*:/g,
          ),
        },
        diagnosticRuntimeProjection: {
          wallClockPerGeneratedTokenUpperBoundMs,
          projections,
        },
        interpretationBoundary: {
          outputBudgetInadequacyProven: true,
          exactAdditionalOutputTokensRequiredConcluded: false,
          nextOutputBudgetSelected: false,
          timeoutChangeJustified: false,
          retryAuthorized: false,
          autoRepairAuthorized: false,
          rawOutputPersistedPublicly: false,
          rawOutputIncludedInConsole: false,
          nextAction:
            "Review structural progress and size the smallest defensible output-budget remediation before any retry.",
        },
        safety: {
          externalNetworkAccessRequested: false,
          ollamaApiCalled: false,
          modelInferenceExecuted: false,
          modelLoadRequested: false,
          modelSwitchExecuted: false,
          productionMutation: false,
          publicationAuthority: false,
        },
      },
      null,
      2,
    ),
  );
}

main();
