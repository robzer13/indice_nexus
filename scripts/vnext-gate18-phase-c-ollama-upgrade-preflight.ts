import { execFileSync } from "node:child_process";

const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const TARGET_OLLAMA_VERSION = "0.34.3";
const MIN_NVIDIA_DRIVER_VERSION = "551.61";
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

function normalizeVersion(value: string): number[] {
  return value
    .trim()
    .replace(/^v/i, "")
    .split(".")
    .map((part) => {
      const match = part.match(/^\d+/);
      return match ? Number(match[0]) : 0;
    });
}

function compareVersions(
  left: string,
  right: string,
): number {
  const a = normalizeVersion(left);
  const b = normalizeVersion(right);
  const length = Math.max(a.length, b.length);

  for (let index = 0; index < length; index += 1) {
    const av = a[index] ?? 0;
    const bv = b[index] ?? 0;

    if (av > bv) return 1;
    if (av < bv) return -1;
  }

  return 0;
}

async function fetchJson<T>(path: string): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10_000);

  try {
    const response = await fetch(
      `${OLLAMA_ENDPOINT}${path}`,
      {
        method: "GET",
        signal: controller.signal,
      },
    );

    if (!response.ok) {
      throw new Error(
        `VNEXT_GATE18_PHASE_C_OLLAMA_UPGRADE_PREFLIGHT_HTTP_${response.status}`,
      );
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const cliRaw =
    runText("ollama", ["--version"]) ??
    runText("ollama", ["-v"]);

  const apiVersion = await fetchJson<{ version?: string }>(
    "/api/version",
  );
  const tags = await fetchJson<{ models?: OllamaTag[] }>(
    "/api/tags",
  );

  const installed = (tags.models ?? []).find(
    (item) => (item.name ?? item.model) === MODEL_NAME,
  );

  const driverRaw = runText("nvidia-smi", [
    "--query-gpu=driver_version",
    "--format=csv,noheader",
  ]);
  const driverVersion =
    driverRaw?.split(/\r?\n/)[0]?.trim() ?? null;

  const driverPass =
    driverVersion !== null &&
    compareVersions(
      driverVersion,
      MIN_NVIDIA_DRIVER_VERSION,
    ) >= 0;

  const modelDigestPass =
    installed?.digest === MODEL_DIGEST;

  const endpointPass =
    typeof apiVersion.version === "string" &&
    apiVersion.version.length > 0;

  const status =
    driverPass && modelDigestPass && endpointPass
      ? "PASS"
      : "BLOCKED";

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_OLLAMA_RUNTIME_UPGRADE_PREFLIGHT_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_RUNTIME_REMEDIATION",
    status,
    mode: "NO_INSTALL_NO_INFERENCE_PREFLIGHT",
    target: {
      ollamaVersion: TARGET_OLLAMA_VERSION,
      minimumNvidiaDriverVersion:
        MIN_NVIDIA_DRIVER_VERSION,
    },
    current: {
      cliVersionRaw: cliRaw,
      apiVersion: apiVersion.version ?? null,
      nvidiaDriverVersion: driverVersion,
    },
    model: {
      name: MODEL_NAME,
      expectedDigest: MODEL_DIGEST,
      observedDigest: installed?.digest ?? null,
      observedSizeBytes: installed?.size ?? null,
      exactDigestValid: modelDigestPass,
    },
    validation: {
      ollamaEndpointReachable: endpointPass,
      nvidiaDriverMeetsMinimum: driverPass,
      modelDigestValid: modelDigestPass,
    },
    safety: {
      installerExecuted: false,
      nvidiaDriverModified: false,
      modelInferenceExecuted: false,
      modelDownloadExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    nextAction:
      status === "PASS"
        ? "Pinned Ollama 0.34.3 upgrade may proceed under the existing authorization."
        : driverPass
          ? "Do not upgrade Ollama; resolve failed preflight condition first."
          : "Do not upgrade Ollama; NVIDIA driver is below the current Windows minimum or could not be verified.",
  };

  console.log(JSON.stringify(payload, null, 2));

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
          "OROTITAN_GATE18_PHASE_C_OLLAMA_RUNTIME_UPGRADE_PREFLIGHT_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C3_RUNTIME_REMEDIATION",
        status: "FAIL",
        mode: "NO_INSTALL_NO_INFERENCE_PREFLIGHT",
        installerExecuted: false,
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
