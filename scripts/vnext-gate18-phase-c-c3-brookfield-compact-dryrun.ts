import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import type { Gate18V02EvidencePacket } from "../runtime/vnext/model-calibration-pilot-v02";
import {
  GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10GenerationSchemaSha256,
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT,
  buildGate18V10BrookfieldTargetedProbeInput,
  gate18V10BrookfieldTargetedProbePromptSha256,
} from "../runtime/vnext/model-calibration-targeted-brookfield-v10";

const MODEL_NAME = "qwen3:1.7b";
const MODEL_DIGEST =
  "8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7";
const LOCAL_CONTEXT_TOKENS = 4096;
const LOCAL_MAX_OUTPUT_TOKENS = 1024;
const REQUIRED_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-039",
  "E-042",
] as const;

interface CliOptions {
  privateRepoRoot: string;
}

interface OllamaTag {
  name?: string;
  model?: string;
  digest?: string;
  size?: number;
  details?: {
    format?: string;
    family?: string;
    parameter_size?: string;
    quantization_level?: string;
  };
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateRepoRoot = "";

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--private-repo-root") {
      privateRepoRoot = argv[++index] ?? "";
      continue;
    }

    throw new Error(
      `VNEXT_GATE18_PHASE_C_C3_DRYRUN_UNKNOWN_ARG:${arg}`,
    );
  }

  if (privateRepoRoot.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_DRYRUN_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  return { privateRepoRoot };
}

function findBrookfield(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) =>
      company.display_name === "Brookfield Corporation",
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_PHASE_C_C3_DRYRUN_BROOKFIELD_NOT_UNIQUE",
    );
  }

  return matches[0] as Gate18PilotCompany;
}

function artifactReader(
  privateRepoRoot: string,
): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = resolve(privateRepoRoot)
      .replaceAll("\\", "/");
    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");
    const normalizedPrefix = normalizedRoot.endsWith("/")
      ? normalizedRoot
      : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(normalizedPrefix)) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C3_DRYRUN_PRIVATE_PATH_ESCAPE",
      );
    }

    try {
      execFileSync(
        "git",
        [
          "-C",
          privateRepoRoot,
          "cat-file",
          "-e",
          `${pin.commit_sha}^{commit}`,
        ],
        {
          stdio: ["ignore", "ignore", "pipe"],
        },
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
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
              .replace(/[\r\n\t]+/g, " ")
              .slice(0, 300)
          : "UNKNOWN_GIT_READ_ERROR";

      throw new Error(
        `VNEXT_GATE18_PHASE_C_C3_DRYRUN_PINNED_GIT_READ_FAILED:${message}`,
      );
    }
  };
}

async function getInstalledModel(): Promise<OllamaTag> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5_000);

  try {
    const response = await fetch(
      "http://127.0.0.1:11434/api/tags",
      {
        method: "GET",
        signal: controller.signal,
      },
    );

    if (!response.ok) {
      throw new Error(
        `VNEXT_GATE18_PHASE_C_C3_DRYRUN_OLLAMA_HTTP_${response.status}`,
      );
    }

    const payload = (await response.json()) as {
      models?: OllamaTag[];
    };

    const model = (payload.models ?? []).find(
      (item) => (item.name ?? item.model) === MODEL_NAME,
    );

    if (!model) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C3_DRYRUN_MODEL_NOT_INSTALLED",
      );
    }

    if (model.digest !== MODEL_DIGEST) {
      throw new Error(
        "VNEXT_GATE18_PHASE_C_C3_DRYRUN_MODEL_DIGEST_MISMATCH",
      );
    }

    return model;
  } finally {
    clearTimeout(timeout);
  }
}

function buildCompactPacket(
  source: Gate18V02EvidencePacket,
): Gate18V02EvidencePacket {
  const evidenceItems = source.evidence_items.filter((item) =>
    REQUIRED_EVIDENCE_IDS.includes(
      item.evidence_id as (typeof REQUIRED_EVIDENCE_IDS)[number],
    ),
  );

  const observedIds = evidenceItems.map(
    (item) => item.evidence_id,
  );

  for (const id of REQUIRED_EVIDENCE_IDS) {
    if (!observedIds.includes(id)) {
      throw new Error(
        `VNEXT_GATE18_PHASE_C_C3_DRYRUN_EVIDENCE_MISSING:${id}`,
      );
    }
  }

  return {
    ...source,
    evidence_items: evidenceItems,
    conflicts: [],
  };
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const company = findBrookfield();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(options.privateRepoRoot),
  );
  const compactPacket = buildCompactPacket(verified.packet);
  const modelInput =
    buildGate18V10BrookfieldTargetedProbeInput(compactPacket);
  const installedModel = await getInstalledModel();

  const systemChars =
    GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT.length;
  const promptChars = modelInput.length;
  const combinedPromptChars = systemChars + promptChars;

  // Deliberately conservative guard for English + JSON.
  // This is not a tokenizer replacement.
  const conservativeInputTokenCeiling = Math.ceil(
    combinedPromptChars / 2,
  );
  const plannedTotalTokenCeiling =
    conservativeInputTokenCeiling + LOCAL_MAX_OUTPUT_TOKENS;

  const compactPacketJson = JSON.stringify(compactPacket);
  const outputSchemaJson = JSON.stringify(
    GATE18_PHASE_B_V10_GENERATION_SCHEMA_SPEC,
  );

  const status =
    plannedTotalTokenCeiling <= LOCAL_CONTEXT_TOKENS
      ? "PASS"
      : "BLOCKED";

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_COMPACT_DRYRUN_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_HISTORICAL_SEMANTIC_REGRESSION_SUITE",
    status,
    mode: "DRY_RUN_NO_INFERENCE",
    externalModelApiCall: false,
    modelInferenceExecuted: false,
    modelDownloadExecuted: false,
    productionMutation: false,
    publicationAuthority: false,
    probe: {
      semanticProbeId:
        GATE18_V10_BROOKFIELD_TARGETED_PROBE_ID,
      phaseCExecutionId:
        "BROOKFIELD_PEER_ROLE_CORE_001_LOCAL_COMPACT",
      comparisonAdmissible: false,
      modelRankingAuthority: false,
      routingAuthority: false,
    },
    case: {
      caseId: compactPacket.case_id,
      displayName: compactPacket.display_name,
      role: compactPacket.role,
      sourceRunId: compactPacket.source_run_id,
      dataCutoff: compactPacket.data_cutoff,
    },
    sourceIntegrity: {
      fullPacketSha256: verified.packetSha256,
      evidenceLedgerSha256:
        verified.evidenceLedgerSha256,
      conflictLedgerSha256:
        verified.conflictLedgerSha256,
    },
    compactPacket: {
      evidenceIds: compactPacket.evidence_items.map(
        (item) => item.evidence_id,
      ),
      conflictIds: [],
      evidenceItems: compactPacket.evidence_items.length,
      conflicts: compactPacket.conflicts.length,
      serializedChars: compactPacketJson.length,
      sha256: sha256Hex(compactPacketJson),
    },
    prompt: {
      promptTemplateId:
        GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_ID,
      promptTemplateVersion:
        GATE18_V10_BROOKFIELD_TARGETED_PROBE_PROMPT_TEMPLATE_VERSION,
      promptTemplateSha256:
        gate18V10BrookfieldTargetedProbePromptSha256(),
      generationSchemaSha256:
        gate18PhaseBV10GenerationSchemaSha256(),
      systemChars,
      promptChars,
      combinedPromptChars,
      generationSchemaChars: outputSchemaJson.length,
    },
    model: {
      name: MODEL_NAME,
      digest: MODEL_DIGEST,
      sizeBytes: installedModel.size ?? null,
      format: installedModel.details?.format ?? null,
      family: installedModel.details?.family ?? null,
      parameterSize:
        installedModel.details?.parameter_size ?? null,
      quantization:
        installedModel.details?.quantization_level ?? null,
    },
    executionPlan: {
      endpoint: "http://127.0.0.1:11434",
      endpointLoopbackOnly: true,
      temperature: 0,
      contextTokens: LOCAL_CONTEXT_TOKENS,
      maxOutputTokens: LOCAL_MAX_OUTPUT_TOKENS,
      keepAlive: "0s",
      conservativeInputTokenCeiling,
      plannedTotalTokenCeiling,
      inferenceAuthorized: false,
    },
    expectedSemantics: {
      findingCount: 2,
      finding1: {
        supportState: "SUPPORTED",
        evidenceIds: ["E-036", "E-037", "E-039"],
        conflictIds: [],
        counterevidenceIds: [],
      },
      finding2: {
        supportState: "SUPPORTED",
        evidenceIds: ["E-042"],
        conflictIds: [],
        counterevidenceIds: [],
      },
    },
    nextAction:
      status === "PASS"
        ? "Human review and separate local-inference authorization."
        : "Do not infer; reduce prompt footprint or revise context plan.",
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
          "OROTITAN_GATE18_PHASE_C_C3_BROOKFIELD_COMPACT_DRYRUN_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C3_HISTORICAL_SEMANTIC_REGRESSION_SUITE",
        status: "FAIL",
        mode: "DRY_RUN_NO_INFERENCE",
        externalModelApiCall: false,
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
