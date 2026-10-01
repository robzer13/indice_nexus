import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { gate18PhaseBV10OutputSchema } from "../runtime/vnext/model-calibration-pilot-v10";

const AUTHORIZATION_ID =
  "G18-PHASEC-C4-CONSTELLATION-LLAMA3_2-3B-RAW-SCHEMA-FORENSIC-AUTH-001";
const AUTHORIZATION_PATH =
  "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_CONSTELLATION_LLAMA3_2_3B_RAW_SCHEMA_FORENSIC_AUTH_001.json";
const EXPECTED_MATRIX_CELL_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_LLAMA3_2_3B_V1_1_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_LLAMA3_2_3B_V1_1_CONTEXT16384_OUTPUT1024_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "Constellation Software";
const EXPECTED_MODEL = "llama3.2:3b-instruct-q4_K_M";
const EXPECTED_DONE_REASON = "stop";
const EXPECTED_PROMPT_EVAL_COUNT = 3157;
const EXPECTED_EVAL_COUNT = 895;
const EXPECTED_MAX_OUTPUT_TOKENS = 1024;
const EXPECTED_SCHEMA_ERROR = "VNEXT_GATE18_V11_RAW_SCHEMA_INVALID";

const REQUIRED_TOP_LEVEL_KEYS = [
  "case_id",
  "data_cutoff",
  "priority_findings",
  "material_conflicts",
  "weak_link_candidates",
  "unresolved_points",
] as const;

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
    inspect_schema_issue_paths_and_codes?: boolean;
    report_section_cardinalities?: boolean;
    compute_raw_text_hash?: boolean;
    print_raw_generated_values?: boolean;
    mutate_source_artifact?: boolean;
    execute_model_inference?: boolean;
    call_ollama?: boolean;
    external_network_access?: boolean;
    retry_inference?: boolean;
  };
  constraints?: {
    authorized_run_count?: number;
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
  validationV11?: {
    rawSchemaPass?: boolean;
    rawSchemaError?: string | null;
    substantiveValidation?: {
      status?: string;
      pass?: boolean | null;
      error?: string | null;
    };
  } | null;
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
    throw new Error(
      `VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_UNKNOWN_ARG:${arg}`,
    );
  }

  if (!privateOutputPath.trim()) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_PRIVATE_OUTPUT_REQUIRED",
    );
  }

  return { privateOutputPath, authorizationId };
}

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function kind(value: unknown): string {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  return typeof value;
}

function arrayCount(value: unknown): number | null {
  return Array.isArray(value) ? value.length : null;
}

function sanitizeIssue(issue: unknown): Record<string, unknown> {
  const source =
    issue && typeof issue === "object"
      ? (issue as Record<string, unknown>)
      : {};

  const out: Record<string, unknown> = {
    path: Array.isArray(source.path)
      ? source.path.map((part) => String(part)).join(".")
      : "",
    code: typeof source.code === "string" ? source.code : "UNKNOWN",
  };

  if (typeof source.expected === "string") {
    out.expected = source.expected;
  }
  if (typeof source.origin === "string") {
    out.origin = source.origin;
  }
  if (typeof source.format === "string") {
    out.format = source.format;
  }
  if (typeof source.minimum === "number") {
    out.minimum = source.minimum;
  }
  if (typeof source.maximum === "number") {
    out.maximum = source.maximum;
  }
  if (typeof source.inclusive === "boolean") {
    out.inclusive = source.inclusive;
  }
  if (Array.isArray(source.values)) {
    out.expectedValueCount = source.values.length;
  }
  if (Array.isArray(source.keys)) {
    out.unrecognizedKeyCount = source.keys.length;
  }
  if (Array.isArray(source.errors)) {
    out.unionBranchCount = source.errors.length;
  }

  return out;
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));

  if (options.authorizationId !== AUTHORIZATION_ID) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_AUTHORIZATION_ID_MISMATCH",
    );
  }

  const authAbsolute = resolve(process.cwd(), AUTHORIZATION_PATH);
  if (!existsSync(authAbsolute)) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_AUTHORIZATION_MISSING",
    );
  }

  const auth = JSON.parse(
    readFileSync(authAbsolute, "utf8"),
  ) as AuthorizationArtifact;

  if (
    auth.authorization_id !== AUTHORIZATION_ID ||
    auth.status !== "AUTHORIZED_READ_ONLY_NO_INFERENCE" ||
    auth.constraints?.authorized_run_count !== 1 ||
    auth.scope?.read_existing_private_artifact !== true ||
    auth.scope.inspect_raw_json_structure !== true ||
    auth.scope.inspect_schema_issue_paths_and_codes !== true ||
    auth.scope.report_section_cardinalities !== true ||
    auth.scope.compute_raw_text_hash !== true ||
    auth.scope.print_raw_generated_values !== false ||
    auth.scope.mutate_source_artifact !== false ||
    auth.scope.execute_model_inference !== false ||
    auth.scope.call_ollama !== false ||
    auth.scope.external_network_access !== false ||
    auth.scope.retry_inference !== false
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_AUTHORIZATION_INVALID",
    );
  }

  const absolute = resolve(options.privateOutputPath);
  if (!existsSync(absolute)) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_PRIVATE_OUTPUT_NOT_FOUND",
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
    run.invocation?.model !== EXPECTED_MODEL ||
    run.execution?.doneReason !== EXPECTED_DONE_REASON ||
    run.execution?.promptEvalCount !== EXPECTED_PROMPT_EVAL_COUNT ||
    run.execution?.evalCount !== EXPECTED_EVAL_COUNT ||
    run.localRuntime?.generation?.maxOutputTokens !==
      EXPECTED_MAX_OUTPUT_TOKENS ||
    run.execution?.runtimeError !== null ||
    run.execution?.schemaValid !== false ||
    run.execution?.schemaError !== EXPECTED_SCHEMA_ERROR ||
    run.execution?.semanticError !== "NOT_EVALUATED_SCHEMA_FAILURE" ||
    run.validationV11?.rawSchemaPass !== false ||
    run.validationV11?.rawSchemaError !== EXPECTED_SCHEMA_ERROR ||
    run.validationV11?.substantiveValidation?.status !==
      "NOT_EVALUATED_SCHEMA_FAILURE"
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_RUN_IDENTITY_MISMATCH",
    );
  }

  const rawText = run.response?.rawText;
  const parsedJson = run.response?.parsedJson;

  if (typeof rawText !== "string" || rawText.length === 0) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_RAW_TEXT_MISSING",
    );
  }

  let reparsed: unknown;
  try {
    reparsed = JSON.parse(rawText);
  } catch {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_EXPECTED_SYNTACTIC_JSON",
    );
  }

  if (
    parsedJson === null ||
    parsedJson === undefined ||
    JSON.stringify(reparsed) !== JSON.stringify(parsedJson)
  ) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_PARSED_JSON_IDENTITY_MISMATCH",
    );
  }

  const parsedObject =
    parsedJson && typeof parsedJson === "object" && !Array.isArray(parsedJson)
      ? (parsedJson as Record<string, unknown>)
      : null;

  const schema = gate18PhaseBV10OutputSchema.safeParse(parsedJson);
  if (schema.success) {
    throw new Error(
      "VNEXT_GATE18_LLAMA32_RAW_SCHEMA_FORENSIC_EXPECTED_SCHEMA_FAILURE",
    );
  }

  const issues = schema.error.issues.map(sanitizeIssue);
  const issuePaths = [...new Set(
    issues.map((issue) => String(issue.path ?? "")),
  )].sort();

  const topLevelKeys = parsedObject
    ? Object.keys(parsedObject).sort()
    : [];
  const requiredTopLevelPresence = Object.fromEntries(
    REQUIRED_TOP_LEVEL_KEYS.map((key) => [
      key,
      parsedObject ? Object.hasOwn(parsedObject, key) : false,
    ]),
  );
  const extraTopLevelKeyCount = parsedObject
    ? topLevelKeys.filter(
        (key) =>
          !(REQUIRED_TOP_LEVEL_KEYS as readonly string[]).includes(key),
      ).length
    : null;

  const sectionCardinalities = {
    priority_findings: parsedObject
      ? arrayCount(parsedObject.priority_findings)
      : null,
    material_conflicts: parsedObject
      ? arrayCount(parsedObject.material_conflicts)
      : null,
    weak_link_candidates: parsedObject
      ? arrayCount(parsedObject.weak_link_candidates)
      : null,
    unresolved_points: parsedObject
      ? arrayCount(parsedObject.unresolved_points)
      : null,
  };

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_LLAMA3_2_3B_RAW_SCHEMA_FORENSIC_V0.1",
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
            EXPECTED_MAX_OUTPUT_TOKENS - EXPECTED_EVAL_COUNT,
          runtimeError: run.execution?.runtimeError ?? null,
          schemaValid: run.execution?.schemaValid ?? null,
          schemaError: run.execution?.schemaError ?? null,
          semanticValid: run.execution?.semanticValid ?? null,
          semanticError: run.execution?.semanticError ?? null,
        },
        rawJson: {
          chars: rawText.length,
          bytes: Buffer.byteLength(rawText, "utf8"),
          sha256: sha256(rawText),
          syntacticallyValid: true,
          parsedJsonKind: kind(parsedJson),
          topLevelKeys,
          requiredTopLevelPresence,
          extraTopLevelKeyCount,
          sectionCardinalities,
        },
        schemaDiagnosis: {
          classification:
            "SYNTACTIC_JSON_VALID_RAW_SCHEMA_CONTRACT_FAILURE",
          issueCount: issues.length,
          uniqueIssuePathCount: issuePaths.length,
          issuePaths,
          issues,
          outputBudgetExhaustionProven: false,
          timeoutFailureProven: false,
          runtimeFailureProven: false,
          semanticQualityAssessable: false,
          retryAutomaticallyJustified: false,
        },
        interpretationBoundary: {
          sourceRunStatusChanged: false,
          retroactivePassAllowed: false,
          sourceArtifactMutated: false,
          rawGeneratedValuesPrinted: false,
          rawOutputPublished: false,
          humanQualityAdjudicationPossibleBeforeSchemaPass: false,
          nextAction:
            "Use sanitized schema issue paths and codes to decide whether one bounded reliability remediation is informative or whether Llama 3.2 should stop at this C4 failure.",
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
