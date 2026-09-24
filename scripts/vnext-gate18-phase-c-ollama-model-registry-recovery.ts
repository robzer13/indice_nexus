import { execFileSync } from "node:child_process";

const OLLAMA_ENDPOINT = "http://127.0.0.1:11434";
const MODEL_NAME = "qwen3:1.7b";
const MODEL_DIGEST =
  "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7";
const TAGS_TIMEOUT_MS = 60_000;

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

function runText(
  command: string,
  args: readonly string[],
): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 30_000,
      maxBuffer: 4 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
}

async function fetchJson<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 10_000,
): Promise<{ ok: true; value: T; elapsedMs: number } | { ok: false; error: string; elapsedMs: number }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const started = Date.now();

  try {
    const response = await fetch(`${OLLAMA_ENDPOINT}${path}`, {
      ...init,
      signal: controller.signal,
    });

    if (!response.ok) {
      const body = (await response.text())
        .replace(/[\r\n\t]+/g, " ")
        .trim()
        .slice(0, 1000);

      return {
        ok: false,
        error: `HTTP_${response.status}:${body || "NO_RESPONSE_BODY"}`,
        elapsedMs: Date.now() - started,
      };
    }

    return {
      ok: true,
      value: (await response.json()) as T,
      elapsedMs: Date.now() - started,
    };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error),
      elapsedMs: Date.now() - started,
    };
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const cliVersionRaw =
    runText("ollama", ["--version"]) ??
    runText("ollama", ["-v"]);
  const cliListRaw = runText("ollama", ["list"]);

  const version = await fetchJson<{ version?: string }>(
    "/api/version",
  );
  const tags = await fetchJson<{ models?: OllamaTag[] }>(
    "/api/tags",
    { method: "GET" },
    TAGS_TIMEOUT_MS,
  );
  const show = await fetchJson<OllamaShow>(
    "/api/show",
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({ model: MODEL_NAME }),
    },
    30_000,
  );

  const tagModels = tags.ok ? tags.value.models ?? [] : [];
  const qwenTag = tagModels.find(
    (item) => (item.name ?? item.model) === MODEL_NAME,
  );

  const cliListsQwen =
    typeof cliListRaw === "string" &&
    cliListRaw
      .split(/\r?\n/)
      .some((line) => line.trim().startsWith(MODEL_NAME));

  const exactDigestValid =
    qwenTag?.digest === MODEL_DIGEST;

  const apiRegistryHealthy =
    tags.ok && exactDigestValid;

  const localPresenceCorroborated =
    cliListsQwen && show.ok;

  const status =
    apiRegistryHealthy && localPresenceCorroborated
      ? "PASS"
      : localPresenceCorroborated
        ? "BLOCKED_API_TAGS"
        : "BLOCKED_MODEL_REGISTRY";

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_OLLAMA_MODEL_REGISTRY_RECOVERY_CHECK_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_RUNTIME_REMEDIATION",
    status,
    mode: "LOCAL_METADATA_ONLY_NO_INFERENCE",
    runtime: {
      endpoint: OLLAMA_ENDPOINT,
      endpointLoopbackOnly: true,
      cliVersionRaw,
      apiVersion:
        version.ok ? version.value.version ?? null : null,
    },
    cliRegistry: {
      listCommandSucceeded: cliListRaw !== null,
      qwenListed: cliListsQwen,
      listOutput: cliListRaw,
    },
    apiRegistry: {
      tagsReachable: tags.ok,
      tagsElapsedMs: tags.elapsedMs,
      tagsError: tags.ok ? null : tags.error,
      qwenObservedDigest: qwenTag?.digest ?? null,
      expectedDigest: MODEL_DIGEST,
      exactDigestValid,
    },
    apiShow: {
      reachable: show.ok,
      elapsedMs: show.elapsedMs,
      error: show.ok ? null : show.error,
      capabilities: show.ok ? show.value.capabilities ?? [] : [],
      details: show.ok ? show.value.details ?? null : null,
    },
    validation: {
      localPresenceCorroborated,
      apiRegistryHealthy,
      exactDigestValid,
    },
    safety: {
      modelInferenceExecuted: false,
      modelDownloadExecuted: false,
      modelDeleted: false,
      driverModified: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    interpretation: {
      c3InferenceAuthorized: false,
      driverUpgradeAuthorized: false,
      nextAction:
        status === "PASS"
          ? "Model registry is verified. Driver remains a separate qualification blocker."
          : status === "BLOCKED_API_TAGS"
            ? "Model presence is corroborated locally, but exact API digest verification remains blocked by /api/tags."
            : "Do not infer or mutate models; resolve local registry visibility first.",
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
          "OROTITAN_GATE18_PHASE_C_OLLAMA_MODEL_REGISTRY_RECOVERY_CHECK_V0.1",
        status: "FAIL",
        mode: "LOCAL_METADATA_ONLY_NO_INFERENCE",
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
