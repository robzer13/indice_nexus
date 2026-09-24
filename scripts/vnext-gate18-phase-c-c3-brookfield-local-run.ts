import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import type { Gate18V02EvidencePacket } from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
  gate18PhaseBV10OutputSchema,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10GenerationSchemaSha256,
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT,
  buildGate18V10BrookfieldTargetedProbeInput,
  gate18V10BrookfieldTargetedProbePromptSha256,
  assertGate18V10BrookfieldTargetedProbeSemantics,
} from "../runtime/vnext/model-calibration-targeted-brookfield-v10";

const MODEL_NAME = "qwen3:1.7b";
const MODEL_DIGEST =
  "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7";
const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const CONTEXT_TOKENS = 8192;
const MAX_OUTPUT_TOKENS = 768;
const AUTHORIZATION_ID =
  "G18-PHASEC-C3-BROOKFIELD-QWEN17-INFERENCE-AUTH-002";
const AUTHORIZATION_PATH =
  "calibration/vnext/OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_QWEN3_1_7B_INFERENCE_AUTH_002.json";
const PRIVATE_OUTPUT_DIR =
  "calibration/vnext/private-runs";
const REQUIRED_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-039",
  "E-042",
] as const;

interface CliOptions {
  privateRepoRoot: string;
  execute: boolean;
  authorizationId: string | null;
}

interface AuthorizationArtifact {
  authorization_id?: string;
  status?: string;
  c3_inference?: {
    authorized?: boolean;
    model_name?: string;
    model_digest?: string;
    semantic_probe_id?: string;
    execution_id?: string;
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
      `VNEXT_GATE18_PHASE_C_C3_LOCAL_UNKNOWN_ARG:${arg}`,
    );
  }

  if (privateRepoRoot.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  return {
    privateRepoRoot,
    execute,
    authorizationId,
  };
}

function findBrookfield(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) =>
      company.display_name === "Brookfield Corporation",
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_BROOKFIELD_NOT_UNIQUE",
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
        "VNEXT_GATE18_PHASE_C_C3_LOCAL_PRIVATE_PATH_ESCAPE",
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
        `VNEXT_GATE18_PHASE_C_C3_LOCAL_PINNED_GIT_READ_FAILED:${message}`,
      );
    }
  };
}

function buildCompactPacket(
  source: Gate18V02EvidencePacket,
): Gate18V02EvidencePacket {
  const evidenceItems = source.evidence_items.filter((item) =>
    REQUIRED_EVIDENCE_IDS.includes(
      item.evidence_id as (typeof REQUIRED_EVIDENCE_IDS)[number],
    ),
  );

  for (const id of REQUIRED_EVIDENCE_IDS) {
    if (!evidenceItems.some((item) => item.evidence_id === id)) {
      throw new Error(
        `VNEXT_GATE18_PHASE_C_C3_LOCAL_EVIDENCE_MISSING:${id}`,
      );
    }
  }

  return {
    ...source,
    evidence_items: evidenceItems,
    conflicts: [],
  };
}

function readAndAssertAuthorization(
  options: CliOptions,
): AuthorizationArtifact {
  if (!options.execute) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_EXECUTE_FLAG_REQUIRED",
    );
  }

  if (options.authorizationId !== AUTHORIZATION_ID) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_AUTHORIZATION_ID_MISMATCH",
    );
  }

  const absolute = resolve(process.cwd(), AUTHORIZATION_PATH);

  if (!existsSync(absolute)) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_AUTHORIZATION_ARTIFACT_MISSING",
    );
  }

  const artifact = JSON.parse(
    readFileSync(absolute, "utf8"),
  ) as AuthorizationArtifact;

  if (
    artifact.authorization_id !== AUTHORIZATION_ID ||
    artifact.status !== "AUTHORIZED_SINGLE_LOCAL_INFERENCE" ||
    artifact.c3_inference?.authorized !== true ||
    artifact.c3_inference.model_name !== MODEL_NAME ||
    artifact.c3_inference.model_digest !== MODEL_DIGEST ||
    artifact.c3_inference.semantic_probe_id !==
      GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID ||
    artifact.c3_inference.execution_id !==
      "BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT"
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_AUTHORIZATION_ARTIFACT_INVALID",
    );
  }

  return artifact;
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 15_000,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(
      `${OLLAMA_ENDPOINT}${path}`,
      {
        ...init,
        signal: controller.signal,
      },
    );

    if (!response.ok) {
      const errorBody = (await response.text())
        .replace(/[\r\n\t]+/g, " ")
        .trim()
        .slice(0, 2000);
      throw new Error(
        [
          `VNEXT_GATE18_PHASE_C_C3_LOCAL_OLLAMA_HTTP_${response.status}`,
          errorBody.length > 0 ? errorBody : "NO_RESPONSE_BODY",
        ].join(":"),
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
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
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_MODEL_NOT_INSTALLED",
    );
  }

  if (model.digest !== MODEL_DIGEST) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_LOCAL_MODEL_DIGEST_MISMATCH",
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
  const installedModel = await assertInstalledModel();

  const company = findBrookfield();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(options.privateRepoRoot),
  );
  const compactPacket = buildCompactPacket(verified.packet);
  const modelInput =
    buildGate18V10BrookfieldTargetedProbeInput(compactPacket);

  const requestBody = {
    model: MODEL_NAME,
    system:
      GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT,
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
        180_000,
      );

    if (providerResponse.model !== MODEL_NAME) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C3_LOCAL_RESPONSE_MODEL_MISMATCH",
      );
    }

    if (providerResponse.done !== true) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C3_LOCAL_GENERATION_NOT_DONE",
      );
    }

    if (typeof providerResponse.response !== "string") {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C3_LOCAL_RESPONSE_TEXT_MISSING",
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
          assertGate18V10BrookfieldTargetedProbeSemantics(
            compactPacket,
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
      "OROTITAN_GATE18_PHASE_C_C3_LOCAL_PRIVATE_RUN_V0.1",
    privateArtifact: true,
    publication: false,
    productionMutation: false,
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_HISTORICAL_SEMANTIC_REGRESSION_SUITE",
    status,
    authorization: {
      id: AUTHORIZATION_ID,
      artifact: authorization,
    },
    invocation: {
      semanticProbeId:
        GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
      phaseCExecutionId:
        "BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT",
      model: MODEL_NAME,
      modelDigest: MODEL_DIGEST,
      promptTemplateId:
        GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
      promptTemplateVersion:
        GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
      promptTemplateSha256:
        gate18V10BrookfieldTargetedProbePromptSha256(),
      generationSchemaSha256:
        gate18PhaseBV10GenerationSchemaSha256(),
      fullPacketSha256: verified.packetSha256,
      compactPacketSha256: sha256Hex(
        JSON.stringify(compactPacket),
      ),
      requestSha256,
    },
    localRuntime: {
      provider: "OLLAMA",
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
      },
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
    expectedSemantics: {
      findingCount: 2,
      finding1: {
        supportState: "SUPPORTED",
        evidenceIds: ["E-036", "E-037", "E-039"],
        conflictIds: [],
        counterevidenceIds: [],
      },
      finding2: {
        supportState: "SUPPORTED",
        evidenceIds: ["E-042"],
        conflictIds: [],
        counterevidenceIds: [],
      },
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
    `${isoFileSafe(completedAt)}__BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT.json`,
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
          "OROTITAN_GATE18_PHASE_C_C3_LOCAL_EXECUTION_SUMMARY_V0.1",
        status,
        semanticProbeId:
          GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
        phaseCExecutionId:
          "BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT",
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
          "OROTITAN_GATE18_PHASE_C_C3_LOCAL_EXECUTION_SUMMARY_V0.1",
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
