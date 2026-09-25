import { execFileSync } from "node:child_process";

const MODEL = "phi4-mini:3.8b-q4_K_M";
const EXPECTED_DIGEST_PREFIX = "78fad5d182a7";
const EXPECTED_QUANTIZATION = "Q4_K_M";
const OLLAMA = "http://127.0.0.1:11434";

function run(command: string, args: readonly string[], timeout = 1_800_000): void {
  execFileSync(command, [...args], {
    stdio: "inherit",
    timeout,
  });
}

async function fetchJson(
  path: string,
  init?: RequestInit,
): Promise<Record<string, unknown>> {
  const response = await fetch(`${OLLAMA}${path}`, init);
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP_${response.status}:${text}`);
  }
  return JSON.parse(text) as Record<string, unknown>;
}

function asString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

async function main(): Promise<void> {
  run("ollama", ["pull", MODEL]);

  const tags = await fetchJson("/api/tags");
  const models = Array.isArray(tags.models) ? tags.models : [];
  const exact = models.find((item) => {
    if (!item || typeof item !== "object") return false;
    const row = item as Record<string, unknown>;
    return row.name === MODEL || row.model === MODEL;
  }) as Record<string, unknown> | undefined;

  if (!exact) {
    throw new Error("PHI4_MINI_EXACT_TAG_NOT_FOUND_AFTER_PULL");
  }

  const digest = asString(exact.digest);
  if (!digest || !digest.startsWith(EXPECTED_DIGEST_PREFIX)) {
    throw new Error(
      `PHI4_MINI_DIGEST_PREFIX_MISMATCH:${digest ?? "null"}`,
    );
  }

  const show = await fetchJson("/api/show", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL }),
  });

  const details =
    show.details && typeof show.details === "object"
      ? (show.details as Record<string, unknown>)
      : {};

  const quantization =
    asString(details.quantization_level) ??
    asString(details.quantization) ??
    null;

  if (quantization !== null && quantization !== EXPECTED_QUANTIZATION) {
    throw new Error(
      `PHI4_MINI_QUANTIZATION_MISMATCH:${quantization}`,
    );
  }

  const payload = {
    format: "OROTITAN_GATE18_PHASE_C_PHI4_MINI_DOWNLOAD_VERIFY_V0.1",
    status: "PASS_PINNED_DOWNLOAD_ONLY",
    authorizationId: "G18-PHASEC-PHI4-MINI-DOWNLOAD-AUTH-001",
    model: {
      name: MODEL,
      digest,
      sizeBytes: asNumber(exact.size),
      format: asString(details.format),
      family: asString(details.family),
      parameterSize: asString(details.parameter_size),
      quantization,
    },
    verification: {
      exactTagPresent: true,
      digestCaptured: true,
      expectedDigestPrefixMatched: true,
      apiShowReachable: true,
      expectedQuantizationMatched:
        quantization === null ? null : quantization === EXPECTED_QUANTIZATION,
    },
    safety: {
      downloadExecuted: true,
      otherModelDownloadExecuted: false,
      loadSmokeExecuted: false,
      promptProvided: false,
      modelInferenceExecuted: false,
      automaticRetryExecuted: false,
      automaticModelSwitchExecuted: false,
      paidExecutionExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
    nextAction:
      "PERSIST_PHI4_MINI_DOWNLOAD_RESULT_AND_PREPARE_SEPARATE_LOAD_ONLY_MEMORY_PREFLIGHT_AUTHORIZATION",
  };

  console.log(JSON.stringify(payload, null, 2));
}

main().catch((error) => {
  console.error(
    JSON.stringify(
      {
        format: "OROTITAN_GATE18_PHASE_C_PHI4_MINI_DOWNLOAD_VERIFY_V0.1",
        status: "BLOCKED",
        error: error instanceof Error ? error.message : String(error),
        safety: {
          modelInferenceExecuted: false,
          loadSmokeExecuted: false,
          automaticRetryExecuted: false,
          automaticModelSwitchExecuted: false,
          paidExecutionExecuted: false,
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
