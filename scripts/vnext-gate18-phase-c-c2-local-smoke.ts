import { execFileSync } from "node:child_process";
import os from "node:os";

const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const MODEL_NAME = "phi:latest";
const MODEL_DIGEST =
  "e2fd6321a5fe6bb3ac8a4e6f1cf04477fd2dea2924cf53237a995387e152ee9c";

interface OllamaTagModel {
  name?: string;
  model?: string;
  digest?: string;
}

interface GenerateResponse {
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

interface SmokePayload {
  status: "PASS";
  evidence_ids: ["E-001"];
  counterevidence_ids: [];
  qualification: "synthetic-local-smoke";
}

function runText(
  command: string,
  args: readonly string[],
): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 10_000,
      maxBuffer: 2 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
}

function nvidiaFreeMiB(): number | null {
  const output = runText("nvidia-smi", [
    "--query-gpu=memory.free",
    "--format=csv,noheader,nounits",
  ]);
  if (!output) {
    return null;
  }

  const first = Number(output.split(/\r?\n/)[0]?.trim());
  return Number.isFinite(first) ? first : null;
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);

  try {
    const response = await fetch(`${OLLAMA_ENDPOINT}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(
        `VNEXT_GATE18_PHASE_C_C2_OLLAMA_HTTP_${response.status}`,
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

function assertExactSmokePayload(
  value: unknown,
): asserts value is SmokePayload {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    throw new Error("VNEXT_GATE18_PHASE_C_C2_OUTPUT_NOT_OBJECT");
  }

  const object = value as Record<string, unknown>;
  const keys = Object.keys(object).sort();
  const expectedKeys = [
    "counterevidence_ids",
    "evidence_ids",
    "qualification",
    "status",
  ];

  if (
    keys.length !== expectedKeys.length ||
    keys.some((key, index) => key !== expectedKeys[index])
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_OUTPUT_KEYSET_MISMATCH",
    );
  }

  if (object.status !== "PASS") {
    throw new Error("VNEXT_GATE18_PHASE_C_C2_STATUS_MISMATCH");
  }

  if (
    !Array.isArray(object.evidence_ids) ||
    object.evidence_ids.length !== 1 ||
    object.evidence_ids[0] !== "E-001"
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_EVIDENCE_IDS_MISMATCH",
    );
  }

  if (
    !Array.isArray(object.counterevidence_ids) ||
    object.counterevidence_ids.length !== 0
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_COUNTEREVIDENCE_MISMATCH",
    );
  }

  if (object.qualification !== "synthetic-local-smoke") {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_QUALIFICATION_MISMATCH",
    );
  }
}

async function main() {
  const tags = await fetchJson<{ models?: OllamaTagModel[] }>(
    "/api/tags",
  );

  const installed = (tags.models ?? []).find(
    (model) =>
      (model.name ?? model.model) === MODEL_NAME,
  );

  if (!installed) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_PINNED_MODEL_NOT_INSTALLED",
    );
  }

  if (installed.digest !== MODEL_DIGEST) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_PINNED_MODEL_DIGEST_MISMATCH",
    );
  }

  const freeRamGiBBefore =
    Math.round((os.freemem() / 1024 ** 3) * 100) / 100;
  const freeVramMiBBefore = nvidiaFreeMiB();
  const startedAt = Date.now();

  const result = await fetchJson<GenerateResponse>(
    "/api/generate",
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        prompt: [
          "Return only one JSON object.",
          "Use exactly these four keys and exact values:",
          'status = "PASS"',
          'evidence_ids = ["E-001"]',
          "counterevidence_ids = []",
          'qualification = "synthetic-local-smoke"',
          "Do not add markdown or any other key.",
        ].join("\n"),
        stream: false,
        format: "json",
        keep_alive: "0s",
        options: {
          temperature: 0,
          num_ctx: 2048,
          num_predict: 128,
        },
      }),
    },
  );

  const wallClockMs = Date.now() - startedAt;

  if (result.model !== MODEL_NAME) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_RESPONSE_MODEL_MISMATCH",
    );
  }

  if (result.done !== true) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_GENERATION_NOT_DONE",
    );
  }

  if (typeof result.response !== "string") {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_RESPONSE_TEXT_MISSING",
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(result.response);
  } catch {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C2_RESPONSE_JSON_PARSE_FAILED",
    );
  }

  assertExactSmokePayload(parsed);

  const freeRamGiBAfter =
    Math.round((os.freemem() / 1024 ** 3) * 100) / 100;
  const freeVramMiBAfter = nvidiaFreeMiB();

  const output = {
    format:
      "OROTITAN_GATE18_PHASE_C_C2_LOCAL_STRUCTURED_OUTPUT_SMOKE_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C2_STRUCTURED_OUTPUT_SMOKE",
    status: "PASS",
    mode: "LOCAL_INFERENCE_ONLY",
    endpoint: OLLAMA_ENDPOINT,
    endpointLoopbackOnly: true,
    externalModelApiCall: false,
    modelDownloadExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
    model: {
      name: MODEL_NAME,
      digest: MODEL_DIGEST,
      role: "RUNTIME_SMOKE_ONLY_NOT_PRODUCTION_CANDIDATE",
    },
    generation: {
      format: "json",
      temperature: 0,
      numCtx: 2048,
      numPredict: 128,
      keepAlive: "0s",
      doneReason: result.done_reason ?? null,
    },
    validation: {
      jsonParseValid: true,
      exactKeysetValid: true,
      exactValuesValid: true,
      exactModelIdentityValid: true,
      semanticSmokeValid: true,
    },
    metrics: {
      wallClockMs,
      totalDurationNs: result.total_duration ?? null,
      loadDurationNs: result.load_duration ?? null,
      promptEvalCount: result.prompt_eval_count ?? null,
      promptEvalDurationNs: result.prompt_eval_duration ?? null,
      evalCount: result.eval_count ?? null,
      evalDurationNs: result.eval_duration ?? null,
      freeRamGiBBefore,
      freeRamGiBAfter,
      freeVramMiBBefore,
      freeVramMiBAfter,
    },
    syntheticOutput: parsed,
    interpretationBoundary: {
      ollamaRuntimeValidated: true,
      structuredJsonPathValidated: true,
      productionModelAdmitted: false,
      phiPromotedToProductionCandidate: false,
      c3SemanticRegressionReady: true,
      nextAction:
        "Select/download a production candidate only after this smoke is reviewed.",
    },
  };

  console.log(JSON.stringify(output, null, 2));
}

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : String(error);

  console.error(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C2_LOCAL_STRUCTURED_OUTPUT_SMOKE_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C2_STRUCTURED_OUTPUT_SMOKE",
        status: "FAIL",
        externalModelApiCall: false,
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
