import { execFileSync } from "node:child_process";
import os from "node:os";

const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const MODEL_NAME = "qwen3:1.7b";
const MODEL_DIGEST =
  "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7";

interface OllamaTag {
  name?: string;
  model?: string;
  digest?: string;
  size?: number;
  details?: Record<string, unknown>;
}

interface OllamaShow {
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
  details?: Record<string, unknown>;
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
  totalMiB: number | null;
  usedMiB: number | null;
  freeMiB: number | null;
} {
  const output = runText("nvidia-smi", [
    "--query-gpu=memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
  ]);

  if (!output) {
    return {
      totalMiB: null,
      usedMiB: null,
      freeMiB: null,
    };
  }

  const first = output.split(/\r?\n/)[0]?.trim();
  if (!first) {
    return {
      totalMiB: null,
      usedMiB: null,
      freeMiB: null,
    };
  }

  const [totalRaw, usedRaw, freeRaw] = first
    .split(",")
    .map((part) => part.trim());

  const total = Number(totalRaw);
  const used = Number(usedRaw);
  const free = Number(freeRaw);

  return {
    totalMiB: Number.isFinite(total) ? total : null,
    usedMiB: Number.isFinite(used) ? used : null,
    freeMiB: Number.isFinite(free) ? free : null,
  };
}

async function fetchJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);

  try {
    const response = await fetch(
      `${OLLAMA_ENDPOINT}${path}`,
      {
        ...init,
        signal: controller.signal,
      },
    );

    if (!response.ok) {
      const body = (await response.text())
        .replace(/[\r\n\t]+/g, " ")
        .trim()
        .slice(0, 2000);

      throw new Error(
        [
          `VNEXT_GATE18_PHASE_C_C3_DIAGNOSTIC_OLLAMA_HTTP_${response.status}`,
          body.length > 0 ? body : "NO_RESPONSE_BODY",
        ].join(":"),
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

function numberBySuffix(
  record: Record<string, unknown> | undefined,
  suffix: string,
): number | null {
  if (!record) {
    return null;
  }

  for (const [key, value] of Object.entries(record)) {
    if (!key.endsWith(suffix)) {
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
  const version = await fetchJson<{ version?: string }>(
    "/api/version",
  );
  const tags = await fetchJson<{ models?: OllamaTag[] }>(
    "/api/tags",
  );
  const show = await fetchJson<OllamaShow>("/api/show", {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: MODEL_NAME,
      verbose: false,
    }),
  });
  const ps = await fetchJson<{ models?: OllamaPsModel[] }>(
    "/api/ps",
  );

  const model = (tags.models ?? []).find(
    (item) => (item.name ?? item.model) === MODEL_NAME,
  );

  if (!model) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_DIAGNOSTIC_MODEL_NOT_INSTALLED",
    );
  }

  if (model.digest !== MODEL_DIGEST) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_DIAGNOSTIC_MODEL_DIGEST_MISMATCH",
    );
  }

  const loaded = (ps.models ?? []).filter(
    (item) => (item.name ?? item.model) === MODEL_NAME,
  );

  const gpu = nvidiaMemory();
  const totalRamGiB =
    Math.round((os.totalmem() / 1024 ** 3) * 100) / 100;
  const freeRamGiB =
    Math.round((os.freemem() / 1024 ** 3) * 100) / 100;

  const modelInfo = show.model_info;

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_RUNTIME_DIAGNOSTIC_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_RUNTIME_FAILURE_DIAGNOSTIC",
    mode: "LOCAL_METADATA_ONLY_NO_INFERENCE",
    externalModelApiCall: false,
    modelInferenceExecuted: false,
    modelDownloadExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
    runtime: {
      provider: "OLLAMA",
      endpoint: OLLAMA_ENDPOINT,
      endpointLoopbackOnly: true,
      version: version.version ?? null,
    },
    model: {
      name: MODEL_NAME,
      digest: MODEL_DIGEST,
      sizeBytes: model.size ?? null,
      details: model.details ?? null,
      capabilities: show.capabilities ?? [],
      contextLength: numberBySuffix(
        modelInfo,
        ".context_length",
      ),
      blockCount: numberBySuffix(
        modelInfo,
        ".block_count",
      ),
      embeddingLength: numberBySuffix(
        modelInfo,
        ".embedding_length",
      ),
      attentionHeads: numberBySuffix(
        modelInfo,
        ".attention.head_count",
      ),
      kvHeads: numberBySuffix(
        modelInfo,
        ".attention.head_count_kv",
      ),
    },
    loadedState: {
      loaded: loaded.length > 0,
      instances: loaded.map((item) => ({
        name: item.name ?? item.model ?? null,
        sizeBytes: item.size ?? null,
        sizeVramBytes: item.size_vram ?? null,
        expiresAt: item.expires_at ?? null,
      })),
    },
    memory: {
      totalRamGiB,
      freeRamGiB,
      totalVramMiB: gpu.totalMiB,
      usedVramMiB: gpu.usedMiB,
      freeVramMiB: gpu.freeMiB,
      lowSystemRamWarning: freeRamGiB < 1,
    },
    interpretationBoundary: {
      diagnosesSemanticQuality: false,
      authorizesRetry: false,
      authorizesModelSwitch: false,
      authorizesDecodingChange: false,
      nextAction:
        "Use metadata plus preserved Ollama HTTP body from any future separately authorized run to classify the runtime failure.",
    },
  };

  console.log(JSON.stringify(payload, null, 2));
}

void main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : String(error);

  console.error(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_RUNTIME_DIAGNOSTIC_V0.1",
        status: "FAIL",
        mode: "LOCAL_METADATA_ONLY_NO_INFERENCE",
        modelInferenceExecuted: false,
        externalModelApiCall: false,
        error: message,
      },
      null,
      2,
    ),
  );

  process.exitCode = 1;
});
