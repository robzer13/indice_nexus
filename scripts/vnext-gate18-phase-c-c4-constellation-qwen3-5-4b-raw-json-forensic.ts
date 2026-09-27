import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const AUTHORIZATION_ID =
  "G18-PHASEC-C4-CONSTELLATION-QWEN3_5-4B-RAW-JSON-FORENSIC-AUTH-001";
const AUTHORIZATION_PATH =
  "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_QWEN3_5_4B_RAW_JSON_FORENSIC_AUTH_001.json";

const EXPECTED_MATRIX_CELL_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_QWEN3_5_4B_V1_1_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_QWEN3_5_4B_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "Constellation Software";
const EXPECTED_MODEL = "qwen3.5:4b-q4_K_M";
const EXPECTED_DONE_REASON = "stop";
const EXPECTED_EVAL_COUNT = 842;
const EXPECTED_MAX_OUTPUT_TOKENS = 1024;
const EXPECTED_SCHEMA_ERROR = "Unexpected end of JSON input";

interface CliOptions {
  privateOutputPath: string;
  authorizationId: string | null;
}

interface AuthorizationArtifact {
  authorization_id?: string;
  status?: string;
  scope?: {
    read_existing_private_artifact?: boolean;
    inspect_raw_json_structure?: boolean;
    execute_model_inference?: boolean;
    call_ollama?: boolean;
    external_network_access?: boolean;
    mutate_source_artifact?: boolean;
  };
}

interface PrivateRunArtifact {
  status?: string;
  invocation?: {
    matrixCellId?: string;
    attemptId?: string;
    company?: string;
    model?: string;
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
  validationV11?: unknown;
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateOutputPath = "";
  let authorizationId: string | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--private-output-path") {
      privateOutputPath = argv[++index] ?? "";
      continue;
    }
    if (arg === "--authorization-id") {
      authorizationId = argv[++index] ?? null;
      continue;
    }
    throw new Error(`VNEXT_GATE18_QWEN35_JSON_FORENSIC_UNKNOWN_ARG:${arg}`);
  }

  if (!privateOutputPath.trim()) {
    throw new Error("VNEXT_GATE18_QWEN35_JSON_FORENSIC_PRIVATE_OUTPUT_REQUIRED");
  }

  return { privateOutputPath, authorizationId };
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
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

type Delimiter = "{" | "[";

function scanStructure(value: string) {
  let inString = false;
  let escapePending = false;
  let mismatchDetected = false;
  const stack: Delimiter[] = [];

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
    if (char === "{" || char === "[") {
      stack.push(char);
      continue;
    }
    if (char === "}" || char === "]") {
      const expected = char === "}" ? "{" : "[";
      if (stack.at(-1) !== expected) {
        mismatchDetected = true;
      } else {
        stack.pop();
      }
    }
  }

  return {
    inString,
    escapePending,
    mismatchDetected,
    openDelimiterCount: stack.length,
    openDelimiters: stack,
  };
}

function terminalTokenClass(value: string): string {
  const trimmed = value.trimEnd();
  if (!trimmed) return "EMPTY";
  const char = trimmed.at(-1) ?? "";
  if (char === '"') return "QUOTE";
  if (char === "}") return "OBJECT_CLOSE";
  if (char === "]") return "ARRAY_CLOSE";
  if (char === ",") return "COMMA";
  if (char === ":") return "COLON";
  if (/\s/.test(char)) return "WHITESPACE";
  if (/[A-Za-z0-9._-]/.test(char)) return "ALPHANUMERIC_OR_WORD";
  return "OTHER";
}

function marker(value: string, key: string) {
  const literal = `"${key}"`;
  const index = value.indexOf(literal);
  return {
    key,
    present: index >= 0,
    index: index >= 0 ? index : null,
    location: index >= 0 ? lineAndColumnAt(value, index) : null,
  };
}

function countPattern(value: string, pattern: RegExp): number {
  return Array.from(value.matchAll(pattern)).length;
}


interface ShapeEntry {
  path: string;
  kind: string;
  stringLength?: number;
  startsWithObject?: boolean;
  sha256?: string;
}

function collectArtifactShape(
  value: unknown,
  path = "$",
  depth = 0,
  out: ShapeEntry[] = [],
): ShapeEntry[] {
  if (depth > 5) return out;

  if (typeof value === "string") {
    if (
      value.length >= 256 ||
      /(raw|response|output|content|text|provider|generation)/i.test(path)
    ) {
      out.push({
        path,
        kind: "string",
        stringLength: value.length,
        startsWithObject: value.trimStart().startsWith("{"),
        sha256: sha256(value),
      });
    }
    return out;
  }

  if (value === null) {
    if (/(raw|response|output|content|text|provider|generation)/i.test(path)) {
      out.push({ path, kind: "null" });
    }
    return out;
  }

  if (Array.isArray(value)) {
    out.push({ path, kind: "array" });
    value.slice(0, 5).forEach((item, index) => {
      collectArtifactShape(item, `${path}[${index}]`, depth + 1, out);
    });
    return out;
  }

  if (typeof value === "object") {
    if (depth <= 2 || /(raw|response|output|content|text|provider|generation)/i.test(path)) {
      out.push({ path, kind: "object" });
    }
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      collectArtifactShape(child, `${path}.${key}`, depth + 1, out);
    }
    return out;
  }

  return out;
}

function structuralClosureProbe(rawText: string) {
  const scan = scanStructure(rawText);
  let suffix = "";

  if (scan.inString) {
    if (scan.escapePending) {
      suffix += "\\";
    }
    suffix += '"';
  }

  for (const delimiter of [...scan.openDelimiters].reverse()) {
    suffix += delimiter === "{" ? "}" : "]";
  }

  if (suffix.length === 0) {
    return {
      attempted: false,
      suffixLength: 0,
      parseableAfterStructuralClosureOnly: false,
      parseErrorAfterClosure: null,
    };
  }

  try {
    JSON.parse(rawText + suffix);
    return {
      attempted: true,
      suffixLength: suffix.length,
      parseableAfterStructuralClosureOnly: true,
      parseErrorAfterClosure: null,
    };
  } catch (error) {
    return {
      attempted: true,
      suffixLength: suffix.length,
      parseableAfterStructuralClosureOnly: false,
      parseErrorAfterClosure:
        error instanceof Error ? error.message : String(error),
    };
  }
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));

  if (options.authorizationId !== AUTHORIZATION_ID) {
    throw new Error("VNEXT_GATE18_QWEN35_JSON_FORENSIC_AUTHORIZATION_ID_MISMATCH");
  }

  const authAbsolute = resolve(process.cwd(), AUTHORIZATION_PATH);
  if (!existsSync(authAbsolute)) {
    throw new Error("VNEXT_GATE18_QWEN35_JSON_FORENSIC_AUTHORIZATION_MISSING");
  }

  const auth = JSON.parse(
    readFileSync(authAbsolute, "utf8"),
  ) as AuthorizationArtifact;

  if (
    auth.authorization_id !== AUTHORIZATION_ID ||
    auth.status !== "AUTHORIZED_READ_ONLY_NO_INFERENCE" ||
    auth.scope?.read_existing_private_artifact !== true ||
    auth.scope.inspect_raw_json_structure !== true ||
    auth.scope.execute_model_inference !== false ||
    auth.scope.call_ollama !== false ||
    auth.scope.external_network_access !== false ||
    auth.scope.mutate_source_artifact !== false
  ) {
    throw new Error("VNEXT_GATE18_QWEN35_JSON_FORENSIC_AUTHORIZATION_INVALID");
  }

  const absolute = resolve(options.privateOutputPath);
  if (!existsSync(absolute)) {
    throw new Error("VNEXT_GATE18_QWEN35_JSON_FORENSIC_PRIVATE_OUTPUT_NOT_FOUND");
  }

  const run = JSON.parse(
    readFileSync(absolute, "utf8"),
  ) as PrivateRunArtifact;

  if (
    run.status !== "FAIL" ||
    run.invocation?.matrixCellId !== EXPECTED_MATRIX_CELL_ID ||
    run.invocation?.attemptId !== EXPECTED_ATTEMPT_ID ||
    run.invocation?.company !== EXPECTED_COMPANY ||
    run.invocation?.model !== EXPECTED_MODEL ||
    run.execution?.doneReason !== EXPECTED_DONE_REASON ||
    run.execution?.evalCount !== EXPECTED_EVAL_COUNT ||
    run.localRuntime?.generation?.maxOutputTokens !== EXPECTED_MAX_OUTPUT_TOKENS ||
    run.execution?.runtimeError !== null ||
    run.execution?.schemaValid !== false ||
    run.execution?.schemaError !== EXPECTED_SCHEMA_ERROR
  ) {
    throw new Error("VNEXT_GATE18_QWEN35_JSON_FORENSIC_RUN_IDENTITY_MISMATCH");
  }

  const rawText = run.response?.rawText;
  if (typeof rawText !== "string" || rawText.length === 0) {
    const root = run as unknown as Record<string, unknown>;
    const responseObject =
      root.response && typeof root.response === "object"
        ? (root.response as Record<string, unknown>)
        : null;

    console.log(
      JSON.stringify(
        {
          format:
            "OROTITAN_GATE18_PHASE_C_C4_QWEN3_5_4B_RAW_JSON_TERMINATION_FORENSIC_V0.2",
          status:
            "PASS_ARTIFACT_SHAPE_DISCOVERY_RAW_TEXT_PATH_UNRESOLVED",
          mode: "LOCAL_READ_ONLY_NO_INFERENCE",
          sourceRun: {
            matrixCellId: EXPECTED_MATRIX_CELL_ID,
            attemptId: EXPECTED_ATTEMPT_ID,
            company: EXPECTED_COMPANY,
            model: EXPECTED_MODEL,
            doneReason: run.execution?.doneReason ?? null,
            evalCount: run.execution?.evalCount ?? null,
            maxOutputTokens:
              run.localRuntime?.generation?.maxOutputTokens ?? null,
            schemaError: run.execution?.schemaError ?? null,
          },
          artifactShape: {
            topLevelKeys: Object.keys(root).sort(),
            responseKeys: responseObject
              ? Object.keys(responseObject).sort()
              : [],
            interestingPaths: collectArtifactShape(run),
          },
          diagnosis: {
            expectedRawTextPath: "$.response.rawText",
            expectedRawTextPresent: false,
            forensicToolingSchemaMismatchObserved: true,
            modelFailureReclassified: false,
            retryAutomaticallyJustified: false,
          },
          interpretationBoundary: {
            sourceArtifactMutated: false,
            rawValuesPrinted: false,
            rawOutputPublished: false,
            nextAction:
              "Use artifact-shape metadata to locate the persisted raw model response path, then rerun the structural forensic without new inference.",
          },
          safety: {
            externalNetworkAccessRequested: false,
            ollamaApiCalled: false,
            modelInferenceExecuted: false,
            modelLoadRequested: false,
            productionMutation: false,
            publicationAuthority: false,
          },
        },
        null,
        2,
      ),
    );
    return;
  }

  const sectionKeys = [
    "case_id",
    "data_cutoff",
    "priority_findings",
    "material_conflicts",
    "weak_link_candidates",
    "unresolved_points",
  ];
  const sectionMarkers = sectionKeys.map((key) => marker(rawText, key));
  const missingRequiredSections = sectionMarkers
    .filter((item) => !item.present)
    .map((item) => item.key);

  const structure = scanStructure(rawText);
  const closureProbe = structuralClosureProbe(rawText);

  const classification =
    run.execution?.doneReason === "stop" &&
    (run.execution?.evalCount ?? EXPECTED_MAX_OUTPUT_TOKENS) <
      EXPECTED_MAX_OUTPUT_TOKENS
      ? "MODEL_STOPPED_WITH_INCOMPLETE_JSON_BEFORE_OUTPUT_BUDGET_EXHAUSTION"
      : "INCOMPLETE_JSON_CAUSE_NOT_ISOLATED";

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_QWEN3_5_4B_RAW_JSON_TERMINATION_FORENSIC_V0.1",
        status: "PASS_READ_ONLY_FORENSIC_COMPLETE",
        mode: "LOCAL_READ_ONLY_NO_INFERENCE",
        sourceRun: {
          matrixCellId: EXPECTED_MATRIX_CELL_ID,
          attemptId: EXPECTED_ATTEMPT_ID,
          company: EXPECTED_COMPANY,
          model: EXPECTED_MODEL,
          wallClockMs: run.execution?.wallClockMs ?? null,
          doneReason: run.execution?.doneReason ?? null,
          promptEvalCount: run.execution?.promptEvalCount ?? null,
          evalCount: run.execution?.evalCount ?? null,
          maxOutputTokens:
            run.localRuntime?.generation?.maxOutputTokens ?? null,
          outputTokenMargin:
            typeof run.execution?.evalCount === "number"
              ? EXPECTED_MAX_OUTPUT_TOKENS - run.execution.evalCount
              : null,
          runtimeError: run.execution?.runtimeError ?? null,
          schemaValid: run.execution?.schemaValid ?? null,
          schemaError: run.execution?.schemaError ?? null,
          semanticValid: run.execution?.semanticValid ?? null,
        },
        rawOutput: {
          chars: rawText.length,
          bytes: Buffer.byteLength(rawText, "utf8"),
          sha256: sha256(rawText),
          startsWithObject: rawText.trimStart().startsWith("{"),
          terminalTokenClass: terminalTokenClass(rawText),
          structure,
          closureProbe,
        },
        structuralProgress: {
          requiredSectionMarkers: sectionMarkers,
          missingRequiredSections,
          priorityFindingClaimCount: countPattern(rawText, /"claim"\s*:/g),
          priorityFindingCausalLinkCount: countPattern(rawText, /"causal_link"\s*:/g),
          materialConflictItemCount: countPattern(rawText, /"conflict_id"\s*:/g),
          weakLinkCandidateCount: countPattern(rawText, /"candidate"\s*:/g),
          unresolvedQuestionCount: countPattern(rawText, /"question"\s*:/g),
        },
        diagnosis: {
          classification,
          outputBudgetExhaustionProven: false,
          timeoutFailureProven: false,
          runtimeFailureProven: false,
          structuredOutputTerminationFailureObserved: true,
          exactSemanticQualityAssessable: false,
          retryAutomaticallyJustified: false,
        },
        interpretationBoundary: {
          sourceRunStatusChanged: false,
          retroactivePassAllowed: false,
          sourceArtifactMutated: false,
          rawOutputPublished: false,
          humanQualityAdjudicationPossibleBeforeValidJson: false,
          nextAction:
            "Use structural progress to decide whether one bounded same-cell reliability retry is informative or whether Qwen3.5 should be stopped as structured-output unreliable.",
        },
        safety: {
          externalNetworkAccessRequested: false,
          ollamaApiCalled: false,
          modelInferenceExecuted: false,
          modelLoadRequested: false,
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
