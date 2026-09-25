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

const MODEL_NAME = "qwen3:4b-instruct";
const MODEL_DIGEST =
  "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
const CONTEXT_TOKENS = 16384;
const MAX_OUTPUT_TOKENS = 1024;
const CLIENT_TIMEOUT_MS = 420_000;
const TRANSPORT = "NODE_HTTP_REQUEST_LOOPBACK";
const COMPLETED_COMPANY = "STMicroelectronics";

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
      `VNEXT_GATE18_C4_REMAINING_PREFLIGHT_UNKNOWN_ARG:${argv[i]}`,
    );
  }
  if (!root.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_REMAINING_PREFLIGHT_PRIVATE_REPO_ROOT_REQUIRED",
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
        "VNEXT_GATE18_C4_REMAINING_PREFLIGHT_PRIVATE_PATH_ESCAPE",
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

function requestMetrics(
  company: Gate18PilotCompany,
  privateRepoRoot: string,
) {
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(privateRepoRoot),
  );
  const prompt = buildGate18PhaseBV10ModelInput(verified.packet);
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

  return {
    role: company.role,
    company: company.display_name,
    dataCutoff: company.data_cutoff,
    evidenceCount: verified.packet.evidence_items.length,
    conflictCount: verified.packet.conflicts.length,
    promptChars: prompt.length,
    promptBytes: byteLength(prompt),
    requestChars: requestJson.length,
    requestBytes: byteLength(requestJson),
    promptSha256: sha256(prompt),
    requestSha256: sha256(requestJson),
    packetSha256: verified.packetSha256,
    matrixStatus:
      company.display_name === COMPLETED_COMPANY
        ? "COMPLETED_WITH_HUMAN_QUALITY_CARRY"
        : "PENDING",
  };
}

function main(): void {
  const privateRepoRoot = parsePrivateRepoRoot(
    process.argv.slice(2),
  );
  const rows = pilotJson.companies.map((company) =>
    requestMetrics(
      company as Gate18PilotCompany,
      privateRepoRoot,
    ),
  );
  const remaining = rows.filter(
    (row) => row.company !== COMPLETED_COMPANY,
  );

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_REMAINING_MATRIX_STATIC_PREFLIGHT_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C4_DIVERSIFIED_COMPANY_MODULE_MATRIX",
        status: "PASS_STATIC_REQUESTS_MEASURED_NO_INFERENCE",
        mode: "NO_INFERENCE_NO_OLLAMA_REQUEST",
        model: {
          name: MODEL_NAME,
          digest: MODEL_DIGEST,
          contextTokens: CONTEXT_TOKENS,
          maxOutputTokens: MAX_OUTPUT_TOKENS,
          temperature: 0,
          clientTimeoutMs: CLIENT_TIMEOUT_MS,
          transport: TRANSPORT,
          sleepGuardRequired: true,
        },
        completedReferenceCell: {
          company: COMPLETED_COMPANY,
          disposition:
            "COMPLETED_WITH_HUMAN_QUALITY_CARRY",
          result:
            "G18-PHASEC-C4-STM-HUMAN-ADJUDICATION-RESULT-001",
        },
        rows,
        remainingRows: remaining,
        summary: {
          totalRowCount: rows.length,
          completedRowCount: 1,
          remainingRowCount: remaining.length,
          minRemainingPromptBytes: Math.min(
            ...remaining.map((row) => row.promptBytes),
          ),
          maxRemainingPromptBytes: Math.max(
            ...remaining.map((row) => row.promptBytes),
          ),
          minRemainingRequestBytes: Math.min(
            ...remaining.map((row) => row.requestBytes),
          ),
          maxRemainingRequestBytes: Math.max(
            ...remaining.map((row) => row.requestBytes),
          ),
        },
        interpretationBoundary: {
          promptIdentityMeasured: true,
          packetIdentityMeasured: true,
          inferenceFitConcluded: false,
          runtimeFitForRemainingCellsConcluded: false,
          semanticQualityAssessed: false,
          c4InferenceAuthorized: false,
          automaticRetryAuthorized: false,
          nextAction:
            "Human review of exact remaining four-company request identities before preparing the next single-cell inference.",
        },
        safety: {
          networkAccessRequested: false,
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
