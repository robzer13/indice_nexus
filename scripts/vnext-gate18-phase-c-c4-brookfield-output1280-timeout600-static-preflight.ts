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

const COMPANY = "Brookfield Corporation";
const MODEL_NAME = "qwen3:4b-instruct";
const MODEL_DIGEST =
  "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
const CONTEXT_TOKENS = 16384;
const MAX_OUTPUT_TOKENS = 1280;
const CLIENT_TIMEOUT_MS = 600_000;
const EXPECTED_PACKET_SHA256 =
  "eb2d779b95fa5f207e91fd3485ece39b2bd101c2589eaaf8c48481f750ef75b3";
const EXPECTED_PROMPT_SHA256 =
  "516496bf4dc9bcdce47897c056282eb2bda63b014cd63c1b76e3992b1c0cdad2";

function sha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

function parsePrivateRepoRoot(argv: readonly string[]): string {
  let root = "";
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--private-repo-root") {
      root = argv[++i] ?? "";
      continue;
    }
    throw new Error(
      `VNEXT_GATE18_C4_BROOKFIELD_OUTPUT1280_PREFLIGHT_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (!root.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_OUTPUT1280_PREFLIGHT_PRIVATE_REPO_ROOT_REQUIRED",
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
        "VNEXT_GATE18_C4_BROOKFIELD_OUTPUT1280_PREFLIGHT_PRIVATE_PATH_ESCAPE",
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

function findBrookfield(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === COMPANY,
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_OUTPUT1280_PREFLIGHT_COMPANY_NOT_UNIQUE",
    );
  }

  return matches[0] as Gate18PilotCompany;
}

function main(): void {
  const privateRepoRoot = parsePrivateRepoRoot(
    process.argv.slice(2),
  );
  const company = findBrookfield();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(privateRepoRoot),
  );
  const prompt = buildGate18PhaseBV10ModelInput(verified.packet);
  const promptSha256 = sha256(prompt);

  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_OUTPUT1280_PREFLIGHT_PACKET_IDENTITY_MISMATCH",
    );
  }
  if (promptSha256 !== EXPECTED_PROMPT_SHA256) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_OUTPUT1280_PREFLIGHT_PROMPT_IDENTITY_MISMATCH",
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
          "OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_OUTPUT1280_TIMEOUT600_STATIC_REQUEST_PREFLIGHT_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C4_OUTPUT_BUDGET_REMEDIATION",
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
          promptBytes: byteLength(prompt),
          requestSha256: sha256(requestJson),
          requestBytes: byteLength(requestJson),
        },
        remediationDelta: {
          maxOutputTokensFrom: 1024,
          maxOutputTokensTo: 1280,
          maxOutputDelta: 256,
          clientTimeoutMsFrom: 480000,
          clientTimeoutMsTo: CLIENT_TIMEOUT_MS,
          clientTimeoutDeltaMs: CLIENT_TIMEOUT_MS - 480000,
          modelChange: false,
          modelDigestChange: false,
          promptChange: false,
          schemaChange: false,
          packetChange: false,
          contextChange: false,
          temperatureChange: false,
          transportChange: false,
          sleepGuardChange: false,
        },
        diagnosticBasis: {
          priorEvalCount: 1024,
          priorDoneReason: "length",
          priorWallClockMs: 416162,
          output1280LinearUpperBoundWallClockMs: 520203,
          proposedTimeoutHeadroomMs: CLIENT_TIMEOUT_MS - 520203,
        },
        interpretationBoundary: {
          requestIdentityMeasured: true,
          inferenceFitConcluded: false,
          runtimeFitConcluded: false,
          semanticQualityAssessed: false,
          inferenceAuthorized: false,
          retryAuthorized: false,
          nextAction:
            "Human review of exact output1280 request identity before preparing any inference runner or authorization.",
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
