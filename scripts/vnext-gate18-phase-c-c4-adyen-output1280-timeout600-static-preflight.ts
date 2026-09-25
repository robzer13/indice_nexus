import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
  GATE18_PHASE_B_V10_SYSTEM_PROMPT,
  buildGate18PhaseBV10ModelInput,
  buildVerifiedGate18V10MoatPacket,
} from "../runtime/vnext/model-calibration-pilot-v10";

const COMPANY = "Adyen";
const MODEL_NAME = "qwen3:4b-instruct";
const MODEL_DIGEST =
  "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
const CONTEXT_TOKENS = 16384;
const MAX_OUTPUT_TOKENS = 1280;
const CLIENT_TIMEOUT_MS = 600_000;
const EXPECTED_PACKET_SHA256 =
  "89e14b09d58b1305064d76170a15bb31e93a0768737d94f2bfc19d32c39a9b74";
const EXPECTED_PROMPT_SHA256 =
  "ef8f55478017eddfb26d68e652aa4855f9d474dc1a75114e6026a12e17666f0c";
const EXPECTED_PROMPT_BYTES = 11671;
const EXPECTED_EVIDENCE_COUNT = 16;
const EXPECTED_CONFLICT_COUNT = 3;

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

function parsePrivateRepoRoot(argv: readonly string[]): string {
  let root = "";

  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--private-repo-root") {
      root = argv[++index] ?? "";
      continue;
    }
    throw new Error(
      `VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_UNKNOWN_ARG:${argv[index]}`,
    );
  }

  if (!root.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  return root;
}

function artifactReader(
  privateRepoRoot: string,
): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = resolve(privateRepoRoot).replaceAll("\\", "/");
    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");
    const prefix = normalizedRoot.endsWith("/")
      ? normalizedRoot
      : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(prefix)) {
      throw new Error(
        "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_PRIVATE_PATH_ESCAPE",
      );
    }

    execFileSync(
      "git",
      [
        "-C",
        privateRepoRoot,
        "cat-file",
        "-e",
        `${pin.commit_sha}^{commit}`,
      ],
      { stdio: ["ignore", "ignore", "pipe"] },
    );

    return execFileSync(
      "git",
      [
        "-C",
        privateRepoRoot,
        "show",
        `${pin.commit_sha}:${pin.path.replaceAll("\\", "/")}`,
      ],
      {
        encoding: "buffer",
        maxBuffer: 20 * 1024 * 1024,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
  };
}

function findAdyen(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === COMPANY,
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_COMPANY_NOT_UNIQUE",
    );
  }

  return matches[0] as Gate18PilotCompany;
}

function main(): void {
  const privateRepoRoot = parsePrivateRepoRoot(
    process.argv.slice(2),
  );
  const company = findAdyen();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(privateRepoRoot),
  );
  const prompt = buildGate18PhaseBV10ModelInput(verified.packet);
  const promptSha256 = sha256(prompt);
  const promptBytes = byteLength(prompt);

  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error(
      "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_PACKET_IDENTITY_MISMATCH",
    );
  }
  if (promptSha256 !== EXPECTED_PROMPT_SHA256) {
    throw new Error(
      "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_PROMPT_IDENTITY_MISMATCH",
    );
  }
  if (promptBytes !== EXPECTED_PROMPT_BYTES) {
    throw new Error(
      "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_PROMPT_BYTES_MISMATCH",
    );
  }
  if (
    verified.packet.evidence_items.length !==
      EXPECTED_EVIDENCE_COUNT ||
    verified.packet.conflicts.length !== EXPECTED_CONFLICT_COUNT
  ) {
    throw new Error(
      "VNEXT_GATE18_C4_ADYEN_OUTPUT1280_PREFLIGHT_PACKET_COUNTS_MISMATCH",
    );
  }

  const requestBody = {
    model: MODEL_NAME,
    system: GATE18_PHASE_B_V10_SYSTEM_PROMPT,
    prompt,
    stream: false,
    format: GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
    keep_alive: "0s",
    options: {
      temperature: 0,
      num_ctx: CONTEXT_TOKENS,
      num_predict: MAX_OUTPUT_TOKENS,
    },
  };

  const requestJson = JSON.stringify(requestBody);

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_ADYEN_OUTPUT1280_TIMEOUT600_STATIC_REQUEST_PREFLIGHT_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C4_DIVERSIFIED_COMPANY_MODULE_MATRIX",
        status: "PASS_STATIC_REQUEST_MEASURED_NO_INFERENCE",
        mode: "NO_INFERENCE_NO_OLLAMA_REQUEST",
        company: {
          role: company.role,
          displayName: company.display_name,
          dataCutoff: company.data_cutoff,
          evidenceCount: verified.packet.evidence_items.length,
          conflictCount: verified.packet.conflicts.length,
        },
        model: {
          name: MODEL_NAME,
          digest: MODEL_DIGEST,
          contextTokens: CONTEXT_TOKENS,
          maxOutputTokens: MAX_OUTPUT_TOKENS,
          temperature: 0,
          clientTimeoutMs: CLIENT_TIMEOUT_MS,
          transport: "NODE_HTTP_REQUEST_LOOPBACK",
          keepAlive: "0s",
          sleepGuardRequired: true,
        },
        identities: {
          packetSha256: verified.packetSha256,
          promptSha256,
          promptBytes,
          requestSha256: sha256(requestJson),
          requestBytes: byteLength(requestJson),
        },
        runtimeBasis: {
          brookfieldNaturalStopEvalCount: 1096,
          brookfieldMaxOutputTokens: 1280,
          brookfieldWallClockMs: 495684,
          rationale:
            "Use validated remaining-cell headroom after Brookfield proved 1024 output tokens can truncate a complete C4 response.",
        },
        interpretationBoundary: {
          requestIdentityMeasured: true,
          inferenceFitConcluded: false,
          runtimeFitConcluded: false,
          semanticQualityAssessed: false,
          inferenceAuthorized: false,
          retryAuthorized: false,
          nextAction:
            "Review exact Adyen output1280 request identity before preparing any inference runner or authorization.",
        },
        safety: {
          externalNetworkAccessRequested: false,
          ollamaApiCalled: false,
          modelLoaded: false,
          modelInferenceExecuted: false,
          productionMutation: false,
          publicationAuthority: false,
        },
      },
      null,
      2,
    ),
  );
}

main();
