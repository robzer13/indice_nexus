import { execFileSync } from "node:child_process";

interface OllamaTagModel {
  name?: string;
  model?: string;
  size?: number;
  digest?: string;
  modified_at?: string;
  details?: Record<string, unknown>;
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

async function getJson<T>(path: string): Promise<T | null> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5_000);

  try {
    const response = await fetch(`http://127.0.0.1:11434${path}`, {
      method: "GET",
      signal: controller.signal,
    });

    if (!response.ok) {
      return null;
    }

    return (await response.json()) as T;
  } catch {
    return null;
  } finally {
    clearTimeout(timeout);
  }
}

function redactModel(model: OllamaTagModel) {
  return {
    name: model.name ?? model.model ?? null,
    sizeBytes: typeof model.size === "number" ? model.size : null,
    digest: model.digest ?? null,
    modifiedAt: model.modified_at ?? null,
    details: model.details ?? null,
  };
}

function redactLoadedModel(model: OllamaPsModel) {
  return {
    name: model.name ?? model.model ?? null,
    sizeBytes: typeof model.size === "number" ? model.size : null,
    sizeVramBytes:
      typeof model.size_vram === "number" ? model.size_vram : null,
    expiresAt: model.expires_at ?? null,
    details: model.details ?? null,
  };
}

async function main() {
  const cliVersion = runText("ollama", ["--version"]);

  const apiVersion = await getJson<{ version?: string }>("/api/version");
  const tags = await getJson<{ models?: OllamaTagModel[] }>("/api/tags");
  const ps = await getJson<{ models?: OllamaPsModel[] }>("/api/ps");

  const endpointReachable = apiVersion !== null;
  const installedModels = (tags?.models ?? []).map(redactModel);
  const loadedModels = (ps?.models ?? []).map(redactLoadedModel);

  const payload = {
    format: "OROTITAN_GATE18_PHASE_C_C1_OLLAMA_RUNTIME_PREFLIGHT_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C1_LOCAL_RUNTIME_ADMISSION",
    mode: "LOCAL_RUNTIME_PREFLIGHT_ONLY",
    endpoint: "http://127.0.0.1:11434",
    endpointLoopbackOnly: true,
    externalNetworkAccessRequested: false,
    externalModelApiCall: false,
    modelInferenceExecuted: false,
    modelDownloadExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
    cliVersion,
    apiVersion: apiVersion?.version ?? null,
    endpointReachable,
    installedModels,
    loadedModels,
    admission: {
      runtimeCandidate: "OLLAMA",
      pass:
        cliVersion !== null &&
        endpointReachable &&
        apiVersion?.version !== undefined,
      noHiddenCloudFallbackAssertedByHarness: true,
      modelCandidateAdmitted: false,
      nextAction:
        "Human review before any model download or inference.",
    },
  };

  console.log(JSON.stringify(payload, null, 2));

  if (!payload.admission.pass) {
    process.exitCode = 1;
  }
}

void main();
