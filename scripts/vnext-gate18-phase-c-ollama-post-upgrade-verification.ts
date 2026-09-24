import { execFileSync } from "node:child_process";

const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const EXPECTED_OLLAMA_VERSION = "0.34.3";
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

function normalizeVersion(value: string): number[] {
  const match = value.match(/\d+(?:\.\d+)+/);
  if (!match) return [];
  return match[0].split(".").map((part) => Number(part));
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

async function tryFetchJson<T>(
  path: string,
): Promise<{ ok: true; value: T } | { ok: false; error: string }> {
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
      const body = (await response.text())
        .replace(/[\r\n\t]+/g, " ")
        .trim()
        .slice(0, 1000);
      return {
        ok: false,
        error: `HTTP_${response.status}:${body || "NO_RESPONSE_BODY"}`,
      };
    }

    return {
      ok: true,
      value: (await response.json()) as T,
    };
  } catch (error) {
    return {
      ok: false,
      error:
        error instanceof Error ? error.message : String(error),
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const cliVersionRaw =
    runText("ollama", ["--version"]) ??
    runText("ollama", ["-v"]);
  const ollamaPathsRaw = runText("where.exe", ["ollama"]);
  const driverRaw = runText("nvidia-smi", [
    "--query-gpu=driver_version,name",
    "--format=csv,noheader",
  ]);

  const apiVersionResult =
    await tryFetchJson<{ version?: string }>("/api/version");
  const tagsResult =
    await tryFetchJson<{ models?: OllamaTag[] }>("/api/tags");
  const psResult =
    await tryFetchJson<{ models?: OllamaPsModel[] }>("/api/ps");

  const apiVersion =
    apiVersionResult.ok
      ? apiVersionResult.value.version ?? null
      : null;

  const models =
    tagsResult.ok ? tagsResult.value.models ?? [] : [];
  const installed = models.find(
    (item) => (item.name ?? item.model) === MODEL_NAME,
  );

  const loaded =
    psResult.ok
      ? (psResult.value.models ?? []).filter(
          (item) =>
            (item.name ?? item.model) === MODEL_NAME,
        )
      : [];

  const driverLine =
    driverRaw?.split(/\r?\n/)[0]?.trim() ?? null;
  const driverVersion =
    driverLine?.split(",")[0]?.trim() ?? null;

  const cliVersionMatches =
    cliVersionRaw !== null &&
    compareVersions(
      cliVersionRaw,
      EXPECTED_OLLAMA_VERSION,
    ) === 0;

  const apiVersionMatches =
    apiVersion !== null &&
    compareVersions(
      apiVersion,
      EXPECTED_OLLAMA_VERSION,
    ) === 0;

  const driverMeetsMinimum =
    driverVersion !== null &&
    compareVersions(
      driverVersion,
      MIN_NVIDIA_DRIVER_VERSION,
    ) >= 0;

  const modelDigestValid =
    installed?.digest === MODEL_DIGEST;

  const binaryApiCoherent =
    cliVersionMatches && apiVersionMatches;

  const upgradeObserved =
    cliVersionMatches || apiVersionMatches;

  const status =
    binaryApiCoherent &&
    modelDigestValid &&
    driverMeetsMinimum
      ? "PASS"
      : binaryApiCoherent && modelDigestValid
        ? "BLOCKED_DRIVER"
        : upgradeObserved
          ? "BLOCKED_INCOHERENT_RUNTIME"
          : "FAIL_UPGRADE_NOT_OBSERVED";

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_OLLAMA_POST_UPGRADE_VERIFICATION_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_RUNTIME_REMEDIATION",
    status,
    mode: "NO_INSTALL_NO_INFERENCE_POSTCHECK",
    target: {
      ollamaVersion: EXPECTED_OLLAMA_VERSION,
      minimumNvidiaDriverVersion:
        MIN_NVIDIA_DRIVER_VERSION,
    },
    observed: {
      cliVersionRaw,
      apiVersion,
      ollamaPaths:
        ollamaPathsRaw?.split(/\r?\n/).filter(Boolean) ?? [],
      nvidiaDriverLine: driverLine,
      nvidiaDriverVersion: driverVersion,
    },
    model: {
      name: MODEL_NAME,
      expectedDigest: MODEL_DIGEST,
      observedDigest: installed?.digest ?? null,
      observedSizeBytes: installed?.size ?? null,
      exactDigestValid: modelDigestValid,
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
    validation: {
      cliVersionMatchesTarget: cliVersionMatches,
      apiVersionMatchesTarget: apiVersionMatches,
      binaryApiCoherent,
      runtimeUpgradeObserved: upgradeObserved,
      nvidiaDriverMeetsMinimum: driverMeetsMinimum,
      modelDigestValid,
      apiVersionEndpointReachable: apiVersionResult.ok,
      tagsEndpointReachable: tagsResult.ok,
      psEndpointReachable: psResult.ok,
    },
    endpointErrors: {
      apiVersion:
        apiVersionResult.ok ? null : apiVersionResult.error,
      tags: tagsResult.ok ? null : tagsResult.error,
      ps: psResult.ok ? null : psResult.error,
    },
    safety: {
      installerExecutedByThisScript: false,
      nvidiaDriverModified: false,
      modelInferenceExecuted: false,
      modelDownloadExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    interpretation: {
      c3InferenceAuthorized: false,
      structuredOutputSmokeAuthorized: false,
      driverUpgradeAuthorized: false,
      nextAction:
        status === "PASS"
          ? "Runtime upgrade verified. Proceed only to separately prepared no-inference metadata validation."
          : status === "BLOCKED_DRIVER"
            ? "Upgrade observed, but GPU qualification is blocked by the NVIDIA driver below the required minimum."
            : "Do not infer. Resolve runtime/version coherence before any further qualification.",
    },
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
          "OROTITAN_GATE18_PHASE_C_OLLAMA_POST_UPGRADE_VERIFICATION_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C3_RUNTIME_REMEDIATION",
        status: "FAIL",
        mode: "NO_INSTALL_NO_INFERENCE_POSTCHECK",
        installerExecutedByThisScript: false,
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
