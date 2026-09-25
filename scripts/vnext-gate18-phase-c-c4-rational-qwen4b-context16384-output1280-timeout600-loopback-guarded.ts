import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import { acquireWindowsSystemRequiredGuard } from "../runtime/vnext/windows-sleep-guard";
import { requestLoopbackJson } from "../runtime/vnext/loopback-http-json-client";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
  GATE18_PHASE_B_V10_MODULE_ID,
  GATE18_PHASE_B_V10_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V10_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V10_SYSTEM_PROMPT,
  assertGate18PhaseBV10Semantics,
  buildGate18PhaseBV10ModelInput,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10GenerationSchemaSha256,
  gate18PhaseBV10OutputSchema,
  gate18PhaseBV10PromptTemplateSha256,
} from "../runtime/vnext/model-calibration-pilot-v10";

const MODEL_NAME = "qwen3:4b-instruct";
const MODEL_DIGEST =
  "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const CONTEXT_TOKENS = 16384;
const MAX_OUTPUT_TOKENS = 1280;
const CLIENT_TIMEOUT_MS = 600_000;
const AUTHORIZATION_ID =
  "G18-PHASEC-C4-RATIONAL-QWEN4B-CONTEXT16384-OUTPUT1280-TIMEOUT600-LOOPBACK-AUTH-001";
const AUTHORIZATION_PATH =
  "calibration/vnext/OROTITAN_GATE18_PHASE_C_C4_RATIONAL_QWEN3_4B_CONTEXT16384_OUTPUT1280_TIMEOUT600_LOOPBACK_AUTH_001.json";
const PRIVATE_OUTPUT_DIR =
  "calibration/vnext/private-runs";
const MATRIX_CELL_ID =
  "C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001";
const ATTEMPT_ID =
  "C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_PROMPT_SHA256 =
  "70265015372600e619010150a72e72dad5df32973f61c01a79658994ea02c0a5";
const EXPECTED_PROMPT_BYTES = 35756;
const EXPECTED_EVIDENCE_COUNT = 56;
const EXPECTED_CONFLICT_COUNT = 9;
const EXPECTED_PACKET_SHA256 =
  "3dc89f69ff39bee857595624cfe65b7772a3591fda538de05af0c757f83ba879";
const EXPECTED_REQUEST_SHA256 =
  "0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d";

interface CliOptions {
  privateRepoRoot: string;
  execute: boolean;
  authorizationId: string | null;
}

interface AuthorizationArtifact {
  authorization_id?: string;
  status?: string;
  c4_inference?: {
    authorized?: boolean;
    model_name?: string;
    model_digest?: string;
    company?: string;
    module_id?: string;
    execution_id?: string;
    attempt_id?: string;
    sleep_guard_required?: boolean;
    prompt_sha256?: string;
    context_tokens?: number;
    max_output_tokens?: number;
    temperature?: number;
    client_timeout_ms?: number;
    transport?: string;
  };
}

interface OllamaTag {
  name?: string;
  model?: string;
  digest?: string;
  size?: number;
  details?: {
    format?: string;
    family?: string;
    parameter_size?: string;
    quantization_level?: string;
  };
}

interface OllamaGenerateResponse {
  model?: string;
  created_at?: string;
  response?: string;
  done?: boolean;
  done_reason?: string;
  total_duration?: number;
  load_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  eval_count?: number;
  eval_duration?: number;
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateRepoRoot = "";
  let execute = false;
  let authorizationId: string | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--private-repo-root") {
      privateRepoRoot = argv[++index] ?? "";
      continue;
    }

    if (arg === "--execute") {
      execute = true;
      continue;
    }

    if (arg === "--authorization-id") {
      authorizationId = argv[++index] ?? null;
      continue;
    }

    throw new Error(
      `VNEXT_GATE18_PHASE_C_C4_LOCAL_UNKNOWN_ARG:${arg}`,
    );
  }

  if (privateRepoRoot.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  return {
    privateRepoRoot,
    execute,
    authorizationId,
  };
}

function findRationalAg(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) =>
      company.display_name === "RATIONAL AG",
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_RATIONAL_NOT_UNIQUE",
    );
  }

  return matches[0] as Gate18PilotCompany;
}

function artifactReader(
  privateRepoRoot: string,
): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = resolve(privateRepoRoot)
      .replaceAll("\\", "/");
    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");
    const normalizedPrefix = normalizedRoot.endsWith("/")
      ? normalizedRoot
      : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(normalizedPrefix)) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C4_LOCAL_PRIVATE_PATH_ESCAPE",
      );
    }

    try {
      execFileSync(
        "git",
        [
          "-C",
          privateRepoRoot,
          "cat-file",
          "-e",
          `${pin.commit_sha}^{commit}`,
        ],
        {
          stdio: ["ignore", "ignore", "pipe"],
        },
      );

      return execFileSync(
        "git",
        [
          "-C",
          privateRepoRoot,
          "show",
          `${pin.commit_sha}:${pin.path.replaceAll("\\", "/")}`,
        ],
        {
          encoding: "buffer",
          maxBuffer: 20 * 1024 * 1024,
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
              .replace(/[\r\n\t]+/g, " ")
              .slice(0, 300)
          : "UNKNOWN_GIT_READ_ERROR";

      throw new Error(
        `VNEXT_GATE18_PHASE_C_C4_LOCAL_PINNED_GIT_READ_FAILED:${message}`,
      );
    }
  };
}

function readAndAssertAuthorization(
  options: CliOptions,
): AuthorizationArtifact {
  if (!options.execute) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_EXECUTE_FLAG_REQUIRED",
    );
  }

  if (options.authorizationId !== AUTHORIZATION_ID) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_AUTHORIZATION_ID_MISMATCH",
    );
  }

  const absolute = resolve(process.cwd(), AUTHORIZATION_PATH);

  if (!existsSync(absolute)) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_AUTHORIZATION_ARTIFACT_MISSING",
    );
  }

  const artifact = JSON.parse(
    readFileSync(absolute, "utf8"),
  ) as AuthorizationArtifact;

  if (
    artifact.authorization_id !== AUTHORIZATION_ID ||
    artifact.status !== "AUTHORIZED_SINGLE_LOCAL_C4_CELL_INFERENCE" ||
    artifact.c4_inference?.authorized !== true ||
    artifact.c4_inference.model_name !== MODEL_NAME ||
    artifact.c4_inference.model_digest !== MODEL_DIGEST ||
    artifact.c4_inference.company !== "RATIONAL AG" ||
    artifact.c4_inference.module_id !== GATE18_PHASE_B_V10_MODULE_ID ||
    artifact.c4_inference.execution_id !== MATRIX_CELL_ID ||
    artifact.c4_inference.attempt_id !== ATTEMPT_ID ||
    artifact.c4_inference.sleep_guard_required !== true ||
    artifact.c4_inference.prompt_sha256 !== EXPECTED_PROMPT_SHA256 ||
    artifact.c4_inference.context_tokens !== CONTEXT_TOKENS ||
    artifact.c4_inference.max_output_tokens !== MAX_OUTPUT_TOKENS ||
    artifact.c4_inference.temperature !== 0 ||
    artifact.c4_inference.client_timeout_ms !== CLIENT_TIMEOUT_MS ||
    artifact.c4_inference.transport !== "NODE_HTTP_REQUEST_LOOPBACK"
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_AUTHORIZATION_ARTIFACT_INVALID",
    );
  }

  return artifact;
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 15_000,
): Promise<T> {
  const headers: Record<string, string> = {};
  if (init?.headers instanceof Headers) {
    init.headers.forEach((value, key) => {
      headers[key] = value;
    });
  } else if (Array.isArray(init?.headers)) {
    for (const [key, value] of init.headers) {
      headers[key] = String(value);
    }
  } else if (init?.headers && typeof init.headers === "object") {
    for (const [key, value] of Object.entries(init.headers)) {
      if (value !== undefined) {
        headers[key] = String(value);
      }
    }
  }

  let body: string | undefined;
  if (typeof init?.body === "string") {
    body = init.body;
  } else if (init?.body !== undefined && init.body !== null) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOOPBACK_HTTP_UNSUPPORTED_REQUEST_BODY",
    );
  }

  return await requestLoopbackJson<T>(
    `${OLLAMA_ENDPOINT}${path}`,
    {
      method: init?.method ?? "GET",
      headers,
      body,
      timeoutMs,
    },
  );
}

function assertNoLoadedModels(): void {
  try {
    const raw = execFileSync("ollama", ["ps"], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
      timeout: 15_000,
      maxBuffer: 1024 * 1024,
    }).trim();

    const lines = raw
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);

    if (lines.length > 1) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C4_QWEN4B_OTHER_MODEL_ALREADY_LOADED",
      );
    }
  } catch (error) {
    if (
      error instanceof Error &&
      error.message ===
        "VNEXT_GATE18_PHASE_C_C4_QWEN4B_OTHER_MODEL_ALREADY_LOADED"
    ) {
      throw error;
    }

    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_QWEN4B_OLLAMA_PS_CHECK_FAILED",
    );
  }
}

async function assertInstalledModel(): Promise<OllamaTag> {
  const tags = await fetchJson<{ models?: OllamaTag[] }>(
    "/api/tags",
  );

  const model = (tags.models ?? []).find(
    (item) => (item.name ?? item.model) === MODEL_NAME,
  );

  if (!model) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_MODEL_NOT_INSTALLED",
    );
  }

  if (model.digest !== MODEL_DIGEST) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_MODEL_DIGEST_MISMATCH",
    );
  }

  return model;
}

function isoFileSafe(value: Date): string {
  return value.toISOString().replace(/[:.]/g, "");
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const authorization = readAndAssertAuthorization(options);
  assertNoLoadedModels();
  const installedModel = await assertInstalledModel();

  const company = findRationalAg();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(options.privateRepoRoot),
  );
  if (verified.packet.evidence_items.length !== EXPECTED_EVIDENCE_COUNT) {
    throw new Error("VNEXT_GATE18_PHASE_C_C4_LOCAL_EVIDENCE_COUNT_MISMATCH");
  }
  if (verified.packet.conflicts.length !== EXPECTED_CONFLICT_COUNT) {
    throw new Error("VNEXT_GATE18_PHASE_C_C4_LOCAL_CONFLICT_COUNT_MISMATCH");
  }

  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_PACKET_IDENTITY_MISMATCH",
    );
  }

  const modelInput = buildGate18PhaseBV10ModelInput(verified.packet);
  const promptSha256 = sha256Hex(modelInput);
  const promptBytes = Buffer.byteLength(modelInput, "utf8");

  if (
    promptSha256 !== EXPECTED_PROMPT_SHA256 ||
    promptBytes !== EXPECTED_PROMPT_BYTES
  ) {
    throw new Error("VNEXT_GATE18_PHASE_C_C4_LOCAL_PROMPT_IDENTITY_MISMATCH");
  }

  const requestBody = {
    model: MODEL_NAME,
    system: GATE18_PHASE_B_V10_SYSTEM_PROMPT,
    prompt: modelInput,
    stream: false,
    format: GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
    keep_alive: "0s",
    options: {
      temperature: 0,
      num_ctx: CONTEXT_TOKENS,
      num_predict: MAX_OUTPUT_TOKENS,
    },
  };

  const requestSha256 = sha256Hex(
    JSON.stringify(requestBody),
  );

  if (requestSha256 !== EXPECTED_REQUEST_SHA256) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C4_LOCAL_REQUEST_IDENTITY_MISMATCH",
    );
  }

  const sleepGuard = await acquireWindowsSystemRequiredGuard();
  let sleepGuardReleaseError: string | null = null;
  const startedAt = Date.now();

  let providerResponse: OllamaGenerateResponse | null = null;
  let rawText: string | null = null;
  let parsedJson: unknown = null;
  let schemaValid = false;
  let semanticValid = false;
  let schemaError: string | null = null;
  let semanticError: string | null = null;
  let runtimeError: string | null = null;

  try {
    providerResponse =
      await fetchJson<OllamaGenerateResponse>(
        "/api/generate",
        {
          method: "POST",
          headers: {
            "content-type": "application/json",
          },
          body: JSON.stringify(requestBody),
        },
        CLIENT_TIMEOUT_MS,
      );

    if (providerResponse.model !== MODEL_NAME) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C4_LOCAL_RESPONSE_MODEL_MISMATCH",
      );
    }

    if (providerResponse.done !== true) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C4_LOCAL_GENERATION_NOT_DONE",
      );
    }

    if (typeof providerResponse.response !== "string") {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C4_LOCAL_RESPONSE_TEXT_MISSING",
      );
    }

    rawText = providerResponse.response;

    try {
      parsedJson = JSON.parse(rawText);
    } catch (error) {
      schemaError =
        error instanceof Error ? error.message : String(error);
    }

    if (parsedJson !== null) {
      const parsed =
        gate18PhaseBV10OutputSchema.safeParse(parsedJson);

      if (!parsed.success) {
        schemaError = parsed.error.message;
      } else {
        schemaValid = true;

        try {
          assertGate18PhaseBV10Semantics(
            verified.packet,
            parsed.data,
          );
          semanticValid = true;
        } catch (error) {
          semanticError =
            error instanceof Error
              ? error.message
              : String(error);
        }
      }
    }
  } catch (error) {
    runtimeError =
      error instanceof Error ? error.message : String(error);
  } finally {
    try {
      await sleepGuard.release();
    } catch (error) {
      sleepGuardReleaseError =
        error instanceof Error ? error.message : String(error);
      if (runtimeError === null) {
        runtimeError = sleepGuardReleaseError;
      }
    }
  }

  const wallClockMs = Date.now() - startedAt;
  const completedAt = new Date();
  const status =
    runtimeError === null &&
    schemaValid &&
    semanticValid
      ? "PASS"
      : "FAIL";

  const privateArtifact = {
    format:
      "OROTITAN_GATE18_PHASE_C_C4_LOCAL_LOOPBACK_TRANSPORT_REMEDIATION_PRIVATE_RUN_V0.1",
    privateArtifact: true,
    publication: false,
    productionMutation: false,
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C4_DIVERSIFIED_COMPANY_MODULE_MATRIX",
    status,
    authorization: {
      id: AUTHORIZATION_ID,
      artifact: authorization,
    },
    invocation: {
      matrixCellId: MATRIX_CELL_ID,
      attemptId: ATTEMPT_ID,
      company: "RATIONAL AG",
      moduleId: GATE18_PHASE_B_V10_MODULE_ID,
      phaseCExecutionId: MATRIX_CELL_ID,
      model: MODEL_NAME,
      modelDigest: MODEL_DIGEST,
      promptTemplateId: GATE18_PHASE_B_V10_PROMPT_TEMPLATE_ID,
      promptTemplateVersion: GATE18_PHASE_B_V10_PROMPT_TEMPLATE_VERSION,
      promptTemplateSha256: gate18PhaseBV10PromptTemplateSha256(),
      promptSha256,
      promptBytes,
      generationSchemaSha256:
        gate18PhaseBV10GenerationSchemaSha256(),
      fullPacketSha256: verified.packetSha256,
      evidenceCount: verified.packet.evidence_items.length,
      conflictCount: verified.packet.conflicts.length,
      requestSha256,
    },
    localRuntime: {
      provider: "OLLAMA",
      transport: "NODE_HTTP_REQUEST_LOOPBACK",
      globalFetchUsed: false,
      hiddenUndiciHeadersTimeoutConfigured: false,
      endpoint: OLLAMA_ENDPOINT,
      endpointLoopbackOnly: true,
      externalModelApiCall: false,
      externalModelApiCostUsd: 0,
      modelDownloadExecuted: false,
      installedModel: {
        name: MODEL_NAME,
        digest: MODEL_DIGEST,
        sizeBytes: installedModel.size ?? null,
        format: installedModel.details?.format ?? null,
        family: installedModel.details?.family ?? null,
        parameterSize:
          installedModel.details?.parameter_size ?? null,
        quantization:
          installedModel.details?.quantization_level ?? null,
      },
      generation: {
        temperature: 0,
        contextTokens: CONTEXT_TOKENS,
        maxOutputTokens: MAX_OUTPUT_TOKENS,
        keepAlive: "0s",
        clientTimeoutMs: CLIENT_TIMEOUT_MS,
      },
    },
    sleepGuard: {
      required: true,
      api: "SetThreadExecutionState",
      acquireFlags: ["ES_CONTINUOUS", "ES_SYSTEM_REQUIRED"],
      releaseFlags: ["ES_CONTINUOUS"],
      readyObserved: sleepGuard.telemetry.readyObserved,
      releasedObserved: sleepGuard.telemetry.releasedObserved,
      acquirePriorFlagsRaw: sleepGuard.telemetry.acquirePriorFlagsRaw,
      releasePriorFlagsRaw: sleepGuard.telemetry.releasePriorFlagsRaw,
      helperPid: sleepGuard.telemetry.helperPid,
      helperExitCode: sleepGuard.telemetry.helperExitCode,
      helperStderr: sleepGuard.telemetry.helperStderr,
      forcedTermination: sleepGuard.telemetry.forcedTermination,
      releaseError: sleepGuardReleaseError,
      lidOrManualSleepPreventionConcluded: false,
    },
    execution: {
      completedAt: completedAt.toISOString(),
      wallClockMs,
      doneReason: providerResponse?.done_reason ?? null,
      totalDurationNs:
        providerResponse?.total_duration ?? null,
      loadDurationNs:
        providerResponse?.load_duration ?? null,
      promptEvalCount:
        providerResponse?.prompt_eval_count ?? null,
      promptEvalDurationNs:
        providerResponse?.prompt_eval_duration ?? null,
      evalCount: providerResponse?.eval_count ?? null,
      evalDurationNs:
        providerResponse?.eval_duration ?? null,
      runtimeError,
      schemaValid,
      schemaError,
      semanticValid,
      semanticError,
    },
    response: {
      rawText,
      parsedJson,
    },
    evaluationContract: {
      packetMode: "FULL_PINNED_PACKET",
      modelDependentModule: GATE18_PHASE_B_V10_MODULE_ID,
      deterministicSemanticValidator: "assertGate18PhaseBV10Semantics",
      humanAdjudicationRequired: true,
      terminalPunctuationReliabilityCarryTracked: true,
    },
    interpretationBoundary: {
      comparisonAdmissible: false,
      modelRankingAuthority: false,
      routingAuthority: false,
      productionCandidateDecisionAuthority: false,
      humanAdjudicationRequired: true,
    },
  };

  const privateDir = resolve(
    process.cwd(),
    PRIVATE_OUTPUT_DIR,
  );
  mkdirSync(privateDir, { recursive: true });

  const outputPath = resolve(
    privateDir,
    `${isoFileSafe(completedAt)}__C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001.json`,
  );

  writeFileSync(
    outputPath,
    JSON.stringify(privateArtifact, null, 2) + "\n",
    "utf8",
  );

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_LOCAL_LOOPBACK_TRANSPORT_REMEDIATION_EXECUTION_SUMMARY_V0.1",
        status,
        matrixCellId: MATRIX_CELL_ID,
        attemptId: ATTEMPT_ID,
        company: "RATIONAL AG",
        moduleId: GATE18_PHASE_B_V10_MODULE_ID,
        phaseCExecutionId: MATRIX_CELL_ID,
        model: MODEL_NAME,
        modelDigest: MODEL_DIGEST,
        externalModelApiCostUsd: 0,
        modelDownloadExecuted: false,
        wallClockMs,
        doneReason:
          providerResponse?.done_reason ?? null,
        promptEvalCount:
          providerResponse?.prompt_eval_count ?? null,
        evalCount:
          providerResponse?.eval_count ?? null,
        runtimeError,
        schemaValid,
        schemaError,
        semanticValid,
        semanticError,
        privateOutputPath: outputPath,
        productionMutation: false,
        publicationAuthority: false,
        comparisonAdmissible: false,
        modelRankingAuthority: false,
      },
      null,
      2,
    ),
  );

  if (status !== "PASS") {
    process.exitCode = 1;
  }
}

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : String(error);

  console.error(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_LOCAL_LOOPBACK_TRANSPORT_REMEDIATION_EXECUTION_SUMMARY_V0.1",
        status: "BLOCKED_BEFORE_INFERENCE",
        externalModelApiCostUsd: 0,
        modelDownloadExecuted: false,
        productionMutation: false,
        publicationAuthority: false,
        error: message,
      },
      null,
      2,
    ),
  );

  process.exitCode = 1;
});
