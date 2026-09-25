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

const MODEL_NAME = "phi4-mini:3.8b-q4_K_M";
const MODEL_DIGEST =
  "78fad5d182a7c33065e153a5f8ba210754207ba9d91973f57dffa7f487363753";
const CONTEXT_TOKENS = 4096;
const MAX_OUTPUT_TOKENS = 768;

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
    throw new Error(`VNEXT_GATE18_C4_PHI4_PREFLIGHT_UNKNOWN_ARG:${argv[i]}`);
  }
  if (!root.trim()) {
    throw new Error("VNEXT_GATE18_C4_PHI4_PREFLIGHT_PRIVATE_REPO_ROOT_REQUIRED");
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
    const prefix = normalizedRoot.endsWith("/") ? normalizedRoot : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(prefix)) {
      throw new Error("VNEXT_GATE18_C4_PHI4_PREFLIGHT_PRIVATE_PATH_ESCAPE");
    }

    execFileSync(
      "git",
      ["-C", privateRepoRoot, "cat-file", "-e", `${pin.commit_sha}^{commit}`],
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

function requestMetrics(company: Gate18PilotCompany, privateRepoRoot: string) {
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
  };
}

function main(): void {
  const privateRepoRoot = parsePrivateRepoRoot(process.argv.slice(2));
  const rows = pilotJson.companies.map((company) =>
    requestMetrics(company as Gate18PilotCompany, privateRepoRoot),
  );

  console.log(JSON.stringify({
    format: "OROTITAN_GATE18_PHASE_C_C4_PHI4_MINI_STATIC_REQUEST_PREFLIGHT_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C4_DIVERSIFIED_COMPANY_MODULE_MATRIX",
    status: "PASS_STATIC_REQUESTS_MEASURED",
    mode: "NO_INFERENCE_NO_OLLAMA_REQUEST",
    model: {
      candidateId: "PHI4_MINI_3_8B_OLLAMA_Q4_K_M",
      name: MODEL_NAME,
      digest: MODEL_DIGEST,
      contextTokens: CONTEXT_TOKENS,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      temperature: 0,
    },
    module: {
      id: "MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3",
      promptTemplateId: "GATE18_MOAT_EVIDENCE_AUDIT_V0_8",
      generationSchemaId: "GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6",
    },
    rows,
    summary: {
      rowCount: rows.length,
      minPromptBytes: Math.min(...rows.map((row) => row.promptBytes)),
      maxPromptBytes: Math.max(...rows.map((row) => row.promptBytes)),
      minRequestBytes: Math.min(...rows.map((row) => row.requestBytes)),
      maxRequestBytes: Math.max(...rows.map((row) => row.requestBytes)),
    },
    calibrationBasis: {
      compactBrookfieldPromptBytes: 4492,
      compactBrookfieldPromptEvalCount: 2020,
      compactAdyenPromptBytes: 7742,
      compactAdyenPromptEvalCount: 2872,
      exactTokenizer: false,
      nextUse: "Derive a Phi-4-specific context-risk screen only after these five static requests are measured.",
    },
    interpretationBoundary: {
      contextFitConcluded: false,
      inferenceFitConcluded: false,
      semanticQualityAssessed: false,
      c4InferenceAuthorized: false,
      nextAction:
        "Review five-company request sizes and derive Phi-4-specific context-risk screen before any C4 inference.",
    },
    safety: {
      networkAccessRequested: false,
      ollamaApiCalled: false,
      modelLoaded: false,
      modelInferenceExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
  }, null, 2));
}

main();
