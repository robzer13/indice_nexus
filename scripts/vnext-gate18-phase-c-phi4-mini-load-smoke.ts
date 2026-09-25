import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

const MODEL = "phi4-mini:3.8b-q4_K_M";
const EXPECTED_DIGEST =
  "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753";
const CONTEXT_TOKENS = 4096;
const AUTHORIZATION_ID = "G18-PHASEC-PHI4-MINI-LOAD-SMOKE-AUTH-001";
const AUTH_PATH = path.resolve(
  process.cwd(),
  "calibration/vnext/OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_AUTH_001.json",
);
const OLLAMA = "http://127.0.0.1:11434";

type JsonRecord = Record<string, unknown>;

function runText(
  command: string,
  args: readonly string[],
  timeout = 15_000,
): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout,
      maxBuffer: 2 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
}

function gib(bytes: number): number {
  return Math.round((bytes / 1024 ** 3) * 100) / 100;
}

function parseNumber(value: string | undefined): number | null {
  if (value === undefined) return null;
  const parsed = Number(value.trim());
  return Number.isFinite(parsed) ? parsed : null;
}

function requireAuthorization(): void {
  if (!existsSync(AUTH_PATH)) {
    throw new Error("PHI4_MINI_LOAD_SMOKE_AUTHORIZATION_ARTIFACT_MISSING");
  }

  const parsed = JSON.parse(readFileSync(AUTH_PATH, "utf8")) as JsonRecord;
  const authority =
    parsed.authority && typeof parsed.authority === "object"
      ? (parsed.authority as JsonRecord)
      : {};

  if (
    parsed.authorization_id !== AUTHORIZATION_ID ||
    parsed.status !== "AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED" ||
    authority.phi4_mini_load_smoke_authorized !== true ||
    authority.phi4_mini_inference_authorized !== false
  ) {
    throw new Error("PHI4_MINI_LOAD_SMOKE_AUTHORIZATION_NOT_ACTIVE");
  }
}

function gpuSnapshot() {
  const raw = runText("nvidia-smi", [
    "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,pci.bus_id",
    "--format=csv,noheader,nounits",
  ]);
  const fields = raw?.split(/\r?\n/)[0]?.split(",").map((v) => v.trim()) ?? [];
  return {
    raw,
    name: fields[0] ?? null,
    driverVersion: fields[1] ?? null,
    memoryTotalMiB: parseNumber(fields[2]),
    memoryFreeMiB: parseNumber(fields[3]),
    memoryUsedMiB: parseNumber(fields[4]),
    pciBusId: fields[5] ?? null,
  };
}

function ollamaPs() {
  const raw = runText("ollama", ["ps"]);
  const lines =
    raw?.split(/\r?\n/).map((line) => line.trim()).filter(Boolean) ?? [];
  return {
    reachable: raw !== null,
    raw,
    rows: lines.length > 1 ? lines.slice(1) : [],
  };
}

async function apiJson(pathname: string, body: JsonRecord): Promise<JsonRecord> {
  const response = await fetch(`${OLLAMA}${pathname}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP_${response.status}:${text}`);
  }
  return JSON.parse(text) as JsonRecord;
}

function ramSnapshot() {
  return {
    totalRamGiB: gib(os.totalmem()),
    freeRamGiB: gib(os.freemem()),
  };
}

async function main(): Promise<void> {
  requireAuthorization();

  const tagsResponse = await fetch(`${OLLAMA}/api/tags`);
  if (!tagsResponse.ok) {
    throw new Error(`TAGS_HTTP_${tagsResponse.status}`);
  }
  const tags = (await tagsResponse.json()) as {
    models?: Array<{ name?: string; digest?: string }>;
  };
  const model = tags.models?.find((item) => item.name === MODEL);
  if (!model || model.digest !== EXPECTED_DIGEST) {
    throw new Error("PHI4_MINI_IDENTITY_MISMATCH");
  }

  const before = {
    ram: ramSnapshot(),
    gpu: gpuSnapshot(),
    ollamaPs: ollamaPs(),
  };

  const loadResponse = await apiJson("/api/generate", {
    model: MODEL,
    stream: false,
    keep_alive: "2m",
    options: {
      num_ctx: CONTEXT_TOKENS,
    },
  });

  const loadOnlyConfirmed =
    loadResponse.response === "" &&
    loadResponse.done === true &&
    (loadResponse.eval_count === undefined || loadResponse.eval_count === 0);

  if (!loadOnlyConfirmed) {
    throw new Error("PHI4_MINI_LOAD_ONLY_GUARD_FAILED");
  }

  await new Promise((resolve) => setTimeout(resolve, 1000));

  const loaded = {
    ram: ramSnapshot(),
    gpu: gpuSnapshot(),
    ollamaPs: ollamaPs(),
  };

  const unloadResponse = await apiJson("/api/generate", {
    model: MODEL,
    stream: false,
    keep_alive: 0,
  });

  const unloadConfirmed =
    unloadResponse.response === "" &&
    unloadResponse.done === true;

  if (!unloadConfirmed) {
    throw new Error("PHI4_MINI_UNLOAD_GUARD_FAILED");
  }

  await new Promise((resolve) => setTimeout(resolve, 1000));

  const after = {
    ram: ramSnapshot(),
    gpu: gpuSnapshot(),
    ollamaPs: ollamaPs(),
  };

  const payload = {
    format: "OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_V0.1",
    authorizationId: AUTHORIZATION_ID,
    status: "PASS_LOAD_ONLY_MEASURED",
    mode: "LOCAL_MODEL_LOAD_WITHOUT_SEMANTIC_INFERENCE",
    target: {
      model: MODEL,
      digest: EXPECTED_DIGEST,
      contextTokens: CONTEXT_TOKENS,
      promptProvided: false,
      inferenceRequested: false,
    },
    before,
    load: {
      response: loadResponse.response ?? null,
      done: loadResponse.done ?? null,
      doneReason: loadResponse.done_reason ?? null,
      evalCount: loadResponse.eval_count ?? null,
      loadDurationNs: loadResponse.load_duration ?? null,
      totalDurationNs: loadResponse.total_duration ?? null,
      loadOnlyConfirmed,
    },
    loaded,
    unload: {
      response: unloadResponse.response ?? null,
      done: unloadResponse.done ?? null,
      doneReason: unloadResponse.done_reason ?? null,
      unloadConfirmed,
    },
    after,
    safety: {
      networkScope: "LOOPBACK_ONLY",
      modelDownloadExecuted: false,
      promptProvided: false,
      messagesProvided: false,
      semanticInferenceExecuted: false,
      automaticRetryExecuted: false,
      automaticModelSwitchExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    interpretationBoundary: {
      inferenceFitConcluded: false,
      semanticQualityAssessed: false,
      phi4MiniInferenceAuthorized: false,
      nextAction:
        "Persist load-only RAM/VRAM measurements and review fit before any separate bounded inference authorization.",
    },
  };

  console.log(JSON.stringify(payload, null, 2));
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        format: "OROTITAN_GATE18_PHASE_C_PHI4_MINI_LOAD_SMOKE_V0.1",
        status: "BLOCKED",
        error: error instanceof Error ? error.message : String(error),
        safety: {
          modelDownloadExecuted: false,
          semanticInferenceExecuted: false,
          automaticRetryExecuted: false,
          automaticModelSwitchExecuted: false,
          productionMutation: false,
          publicationAuthority: false,
        },
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
});
