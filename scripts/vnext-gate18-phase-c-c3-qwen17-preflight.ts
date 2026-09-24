import { execFileSync } from "node:child_process";
import os from "node:os";

const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const MODEL_NAME = "qwen3:1.7b";
const MODEL_DIGEST =
  "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7";

interface OllamaTagModel {
  name?: string;
  model?: string;
  digest?: string;
  size?: number;
  details?: {
    parent_model?: string;
    format?: string;
    family?: string;
    families?: string[];
    parameter_size?: string;
    quantization_level?: string;
  };
}

interface OllamaShowResponse {
  license?: string;
  modelfile?: string;
  parameters?: string;
  template?: string;
  details?: Record<string, unknown>;
  model_info?: Record<string, unknown>;
  capabilities?: string[];
}

interface OllamaPsModel {
  name?: string;
  model?: string;
  size?: number;
  size_vram?: number;
  expires_at?: string;
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

function nvidiaMemory(): {
  freeMiB: number | null;
  totalMiB: number | null;
} {
  const output = runText("nvidia-smi", [
    "--query-gpu=memory.free,memory.total",
    "--format=csv,noheader,nounits",
  ]);
  if (!output) {
    return { freeMiB: null, totalMiB: null };
  }

  const first = output.split(/\r?\n/)[0]?.trim();
  if (!first) {
    return { freeMiB: null, totalMiB: null };
  }

  const [freeRaw, totalRaw] = first.split(",").map((part) => part.trim());
  const free = Number(freeRaw);
  const total = Number(totalRaw);

  return {
    freeMiB: Number.isFinite(free) ? free : null,
    totalMiB: Number.isFinite(total) ? total : null,
  };
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);

  try {
    const response = await fetch(`${OLLAMA_ENDPOINT}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(
        `VNEXT_GATE18_PHASE_C_C3_PREFLIGHT_HTTP_${response.status}`,
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

function findContextLength(
  modelInfo: Record<string, unknown> | undefined,
): number | null {
  if (!modelInfo) {
    return null;
  }

  for (const [key, value] of Object.entries(modelInfo)) {
    if (!key.endsWith(".context_length")) {
      continue;
    }

    if (typeof value === "number" && Number.isFinite(value)) {
      return value;
    }

    if (typeof value === "string") {
      const parsed = Number(value);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return null;
}

async function main() {
  const tags = await fetchJson<{ models?: OllamaTagModel[] }>(
    "/api/tags",
  );
  const installed = (tags.models ?? []).find(
    (model) => (model.name ?? model.model) === MODEL_NAME,
  );

  if (!installed) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_PREFLIGHT_PINNED_MODEL_NOT_INSTALLED",
    );
  }

  if (installed.digest !== MODEL_DIGEST) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_PREFLIGHT_PINNED_MODEL_DIGEST_MISMATCH",
    );
  }

  const show = await fetchJson<OllamaShowResponse>(
    "/api/show",
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        verbose: false,
      }),
    },
  );

  const ps = await fetchJson<{ models?: OllamaPsModel[] }>(
    "/api/ps",
  );

  const loaded = (ps.models ?? []).filter(
    (model) => (model.name ?? model.model) === MODEL_NAME,
  );

  const gpu = nvidiaMemory();
  const freeRamGiB =
    Math.round((os.freemem() / 1024 ** 3) * 100) / 100;
  const totalRamGiB =
    Math.round((os.totalmem() / 1024 ** 3) * 100) / 100;
  const contextLength = findContextLength(show.model_info);

  const recommendedFirstC3Mode =
    contextLength !== null && contextLength >= 4096
      ? "TARGETED_REGRESSION_COMPACT_FIRST"
      : "BLOCK_CONTEXT_CAPABILITY_INSUFFICIENT";

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_C3_QWEN3_1_7B_PREFLIGHT_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_HISTORICAL_SEMANTIC_REGRESSION_SUITE",
    status:
      recommendedFirstC3Mode ===
      "TARGETED_REGRESSION_COMPACT_FIRST"
        ? "PASS"
        : "BLOCKED",
    mode: "LOCAL_METADATA_PREFLIGHT_ONLY",
    endpoint: OLLAMA_ENDPOINT,
    endpointLoopbackOnly: true,
    externalModelApiCall: false,
    modelInferenceExecuted: false,
    modelDownloadExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
    model: {
      name: MODEL_NAME,
      digest: MODEL_DIGEST,
      sizeBytes: installed.size ?? null,
      format: installed.details?.format ?? null,
      family: installed.details?.family ?? null,
      parameterSize: installed.details?.parameter_size ?? null,
      quantization:
        installed.details?.quantization_level ?? null,
      contextLength,
      capabilities: show.capabilities ?? [],
    },
    runtimeState: {
      currentlyLoaded: loaded.length > 0,
      loadedInstances: loaded.map((model) => ({
        name: model.name ?? model.model ?? null,
        sizeBytes: model.size ?? null,
        sizeVramBytes: model.size_vram ?? null,
        expiresAt: model.expires_at ?? null,
      })),
    },
    memory: {
      totalRamGiB,
      freeRamGiB,
      totalVramMiB: gpu.totalMiB,
      freeVramMiB: gpu.freeMiB,
      lowSystemRamWarning: freeRamGiB < 1,
    },
    c3Planning: {
      recommendedFirstMode: recommendedFirstC3Mode,
      rationale:
        "Start with compact targeted semantic regressions before any full-packet normal-path run. Full Phase B packets previously required roughly 4k-4.5k input tokens plus output headroom.",
      minimumPlannedContextTokens: 4096,
      preferredTargetedContextTokens: 4096,
      fullPacketContextNotYetAuthorized: true,
      c3InferenceAuthorized: false,
    },
    nextAction:
      "Human review. No inference is authorized by this preflight.",
  };

  console.log(JSON.stringify(payload, null, 2));

  if (payload.status !== "PASS") {
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
          "OROTITAN_GATE18_PHASE_C_C3_QWEN3_1_7B_PREFLIGHT_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C3_HISTORICAL_SEMANTIC_REGRESSION_SUITE",
        status: "FAIL",
        externalModelApiCall: false,
        modelInferenceExecuted: false,
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
