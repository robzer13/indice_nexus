import { execFileSync } from "node:child_process";
import os from "node:os";

const MODEL = "qwen3:4b-instruct";
const EXPECTED_DIGEST =
  "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
const OLLAMA = "http://127.0.0.1:11434";

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

async function apiJson(
  path: string,
  body: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const response = await fetch(`${OLLAMA}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP_${response.status}:${text}`);
  }
  return JSON.parse(text) as Record<string, unknown>;
}

function ramSnapshot() {
  return {
    totalRamGiB: gib(os.totalmem()),
    freeRamGiB: gib(os.freemem()),
  };
}

async function main(): Promise<void> {
  const tagsResponse = await fetch(`${OLLAMA}/api/tags`);
  if (!tagsResponse.ok) {
    throw new Error(`TAGS_HTTP_${tagsResponse.status}`);
  }
  const tags = (await tagsResponse.json()) as {
    models?: Array<{ name?: string; digest?: string }>;
  };
  const model = tags.models?.find((item) => item.name === MODEL);
  if (!model || model.digest !== EXPECTED_DIGEST) {
    throw new Error("QWEN3_4B_IDENTITY_MISMATCH");
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
  });

  const loadOnlyConfirmed =
    loadResponse.response === "" &&
    loadResponse.done === true &&
    (loadResponse.eval_count === undefined || loadResponse.eval_count === 0);

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

  await new Promise((resolve) => setTimeout(resolve, 1000));

  const after = {
    ram: ramSnapshot(),
    gpu: gpuSnapshot(),
    ollamaPs: ollamaPs(),
  };

  const payload = {
    format: "OROTITAN_GATE18_PHASE_C_QWEN3_4B_LOAD_SMOKE_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_CAPABILITY_ESCALATION_LOAD_SMOKE",
    status: loadOnlyConfirmed ? "PASS_LOAD_ONLY_MEASURED" : "FAIL_LOAD_ONLY_GUARD",
    mode: "LOCAL_MODEL_LOAD_WITHOUT_INFERENCE",
    target: {
      model: MODEL,
      digest: EXPECTED_DIGEST,
      contextTokens: null,
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
      done: unloadResponse.done ?? null,
      doneReason: unloadResponse.done_reason ?? null,
      response: unloadResponse.response ?? null,
    },
    after,
    safety: {
      networkScope: "LOOPBACK_ONLY",
      modelDownloadExecuted: false,
      promptProvided: false,
      modelInferenceExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    interpretationBoundary: {
      inferenceFitConcluded: false,
      semanticQualityAssessed: false,
      c3InferenceAuthorized: false,
      nextAction:
        "Human review of load-only RAM/VRAM measurements before any bounded Qwen3 4B inference authorization.",
    },
  };

  console.log(JSON.stringify(payload, null, 2));

  if (!loadOnlyConfirmed) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        format: "OROTITAN_GATE18_PHASE_C_QWEN3_4B_LOAD_SMOKE_V0.1",
        status: "BLOCKED",
        error: error instanceof Error ? error.message : String(error),
        safety: {
          modelDownloadExecuted: false,
          modelInferenceExecuted: false,
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
