import { execFileSync } from "node:child_process";
import os from "node:os";

const TARGET_CANDIDATE_ID = "QWEN3_4B_INSTRUCT_OLLAMA_Q4_K_M";
const TARGET_MODEL = "qwen3:4b-instruct";
const TARGET_ARTIFACT_CLASS_GIB = 2.5;
const EXPECTED_GPU = "NVIDIA GeForce RTX 3050 Laptop GPU";

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

const totalRamGiB = gib(os.totalmem());
const freeRamGiB = gib(os.freemem());
const freeRamPercent =
  os.totalmem() > 0
    ? Math.round((os.freemem() / os.totalmem()) * 10_000) / 100
    : null;

const nvidiaSmi = runText("nvidia-smi", [
  "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,pci.bus_id",
  "--format=csv,noheader,nounits",
]);

const nvidiaRow = nvidiaSmi?.split(/\r?\n/)[0]?.trim() ?? null;
const nvidiaFields =
  nvidiaRow?.split(",").map((part) => part.trim()) ?? [];

const observedGpu = nvidiaFields[0] ?? null;
const driverVersion = nvidiaFields[1] ?? null;
const memoryTotalMiB = parseNumber(nvidiaFields[2]);
const memoryFreeMiB = parseNumber(nvidiaFields[3]);
const memoryUsedMiB = parseNumber(nvidiaFields[4]);
const pciBusId = nvidiaFields[5] ?? null;

const ollamaPsRaw = runText("ollama", ["ps"]);
const ollamaPsLines =
  ollamaPsRaw
    ?.split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean) ?? [];
const ollamaLoadedModelRows =
  ollamaPsLines.length > 1 ? ollamaPsLines.slice(1) : [];

const measurementComplete =
  observedGpu !== null &&
  memoryTotalMiB !== null &&
  memoryFreeMiB !== null &&
  memoryUsedMiB !== null;

const payload = {
  format:
    "OROTITAN_GATE18_PHASE_C_QWEN3_4B_MEMORY_PREFLIGHT_V0.1",
  gate: 18,
  phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
  stage: "C3_CAPABILITY_ESCALATION_PREFLIGHT",
  status: measurementComplete
    ? "MEASUREMENTS_COLLECTED_REVIEW_REQUIRED"
    : "BLOCKED_MEASUREMENT_INCOMPLETE",
  mode: "LOCAL_RESOURCE_MEASUREMENT_ONLY",
  target: {
    candidateId: TARGET_CANDIDATE_ID,
    modelName: TARGET_MODEL,
    quantization: "Q4_K_M",
    artifactClassGiBApprox: TARGET_ARTIFACT_CLASS_GIB,
    role: "CAPABILITY_ESCALATION_ONLY",
  },
  system: {
    totalRamGiB,
    freeRamGiBAtProbe: freeRamGiB,
    freeRamPercentAtProbe: freeRamPercent,
  },
  gpu: {
    expectedName: EXPECTED_GPU,
    observedName: observedGpu,
    exactGpuName: observedGpu === EXPECTED_GPU,
    driverVersion,
    memoryTotalMiB,
    memoryFreeMiB,
    memoryUsedMiB,
    pciBusId,
  },
  ollama: {
    psReachable: ollamaPsRaw !== null,
    loadedModelCount: ollamaLoadedModelRows.length,
    loadedModelRows: ollamaLoadedModelRows,
  },
  interpretationBoundary: {
    qwen3_1_7bCapabilityFailureEstablished: true,
    qwen3_4bResourceFitConcluded: false,
    qwen3_4bDownloadAuthorized: false,
    qwen3_4bInferenceAuthorized: false,
    humanReviewRequired: true,
    nextAction: measurementComplete
      ? "Review current free RAM, free VRAM and loaded-model state before any qwen3:4b-instruct download authorization."
      : "Resolve local resource measurement failure before any qwen3:4b-instruct download authorization.",
  },
  safety: {
    networkAccessRequested: false,
    modelDownloadExecuted: false,
    modelInferenceExecuted: false,
    modelSwitchExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
  },
};

console.log(JSON.stringify(payload, null, 2));

if (!measurementComplete) {
  process.exitCode = 1;
}
