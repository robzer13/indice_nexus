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
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  GATE18_V10_TARGETED_PROBE_SYSTEM_PROMPT as ADYEN_SYSTEM_PROMPT,
  buildGate18V10TargetedProbeInput,
} from "../runtime/vnext/model-calibration-targeted-regression-v10";
import {
  GATE18_V10_BROOKFIELD_TARGETED_PROBE_SYSTEM_PROMPT as BROOKFIELD_SYSTEM_PROMPT,
  buildGate18V10BrookfieldTargetedProbeInput,
} from "../runtime/vnext/model-calibration-targeted-brookfield-v10";

const MODEL_NAME = "qwen3:4b-instruct";
const CONTEXT_TOKENS = 4096;
const MAX_OUTPUT_TOKENS = 768;

const ADYEN_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-040",
  "E-041",
  "E-042",
  "E-043",
  "E-055",
  "E-056",
] as const;

const ADYEN_CONFLICT_IDS = ["C-005", "C-010"] as const;

const BROOKFIELD_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-039",
  "E-042",
] as const;

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
      `VNEXT_GATE18_C3_ADYEN_STATIC_DIAGNOSTIC_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (root.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_C3_ADYEN_STATIC_DIAGNOSTIC_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  return root;
}

function findCompany(displayName: string): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === displayName,
  );

  if (matches.length !== 1) {
    throw new Error(
      `VNEXT_GATE18_C3_STATIC_DIAGNOSTIC_COMPANY_NOT_UNIQUE:${displayName}`,
    );
  }

  return matches[0] as Gate18PilotCompany;
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
        "VNEXT_GATE18_C3_STATIC_DIAGNOSTIC_PRIVATE_PATH_ESCAPE",
      );
    }

    execFileSync(
      "git",
      ["-C", privateRepoRoot, "cat-file", "-e", `${pin.commit_sha}^{commit}`],
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
  };
}

function compactAdyen(
  source: Gate18V02EvidencePacket,
): Gate18V02EvidencePacket {
  const evidenceItems = source.evidence_items.filter((item) =>
    ADYEN_EVIDENCE_IDS.includes(
      item.evidence_id as (typeof ADYEN_EVIDENCE_IDS)[number],
    ),
  );
  const conflicts = source.conflicts.filter((item) =>
    ADYEN_CONFLICT_IDS.includes(
      item.conflict_id as (typeof ADYEN_CONFLICT_IDS)[number],
    ),
  );

  for (const id of ADYEN_EVIDENCE_IDS) {
    if (!evidenceItems.some((item) => item.evidence_id === id)) {
      throw new Error(
        `VNEXT_GATE18_C3_STATIC_DIAGNOSTIC_ADYEN_EVIDENCE_MISSING:${id}`,
      );
    }
  }

  for (const id of ADYEN_CONFLICT_IDS) {
    if (!conflicts.some((item) => item.conflict_id === id)) {
      throw new Error(
        `VNEXT_GATE18_C3_STATIC_DIAGNOSTIC_ADYEN_CONFLICT_MISSING:${id}`,
      );
    }
  }

  return {
    ...source,
    evidence_items: evidenceItems,
    conflicts,
  };
}

function compactBrookfield(
  source: Gate18V02EvidencePacket,
): Gate18V02EvidencePacket {
  const evidenceItems = source.evidence_items.filter((item) =>
    BROOKFIELD_EVIDENCE_IDS.includes(
      item.evidence_id as (typeof BROOKFIELD_EVIDENCE_IDS)[number],
    ),
  );

  for (const id of BROOKFIELD_EVIDENCE_IDS) {
    if (!evidenceItems.some((item) => item.evidence_id === id)) {
      throw new Error(
        `VNEXT_GATE18_C3_STATIC_DIAGNOSTIC_BROOKFIELD_EVIDENCE_MISSING:${id}`,
      );
    }
  }

  return {
    ...source,
    evidence_items: evidenceItems,
    conflicts: [],
  };
}

function requestMetrics(
  system: string,
  prompt: string,
): {
  systemChars: number;
  systemBytes: number;
  promptChars: number;
  promptBytes: number;
  requestChars: number;
  requestBytes: number;
  promptSha256: string;
  requestSha256: string;
} {
  const requestBody = {
    model: MODEL_NAME,
    system,
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
    systemChars: system.length,
    systemBytes: byteLength(system),
    promptChars: prompt.length,
    promptBytes: byteLength(prompt),
    requestChars: requestJson.length,
    requestBytes: byteLength(requestJson),
    promptSha256: sha256(prompt),
    requestSha256: sha256(requestJson),
  };
}

function ratio(numerator: number, denominator: number): number {
  return Math.round((numerator / denominator) * 1000) / 1000;
}

function main(): void {
  const privateRepoRoot = parsePrivateRepoRoot(process.argv.slice(2));
  const reader = artifactReader(privateRepoRoot);

  const adyenVerified = buildVerifiedGate18V10MoatPacket(
    findCompany("Adyen"),
    reader,
  );
  const brookfieldVerified = buildVerifiedGate18V10MoatPacket(
    findCompany("Brookfield Corporation"),
    reader,
  );

  const adyenPacket = compactAdyen(adyenVerified.packet);
  const brookfieldPacket = compactBrookfield(brookfieldVerified.packet);

  const adyenPrompt = buildGate18V10TargetedProbeInput(adyenPacket);
  const brookfieldPrompt =
    buildGate18V10BrookfieldTargetedProbeInput(brookfieldPacket);

  const adyen = requestMetrics(ADYEN_SYSTEM_PROMPT, adyenPrompt);
  const brookfield = requestMetrics(
    BROOKFIELD_SYSTEM_PROMPT,
    brookfieldPrompt,
  );

  const payload = {
    format:
      "OROTITAN_GATE18_PHASE_C_C3_ADYEN_TIMEOUT_STATIC_DIAGNOSTIC_V0.1",
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C3_RUNTIME_FAILURE_DIAGNOSTIC",
    status: "PASS_STATIC_REQUEST_MEASURED",
    mode: "NO_INFERENCE_NO_OLLAMA_REQUEST",
    pinned_generation_config: {
      model: MODEL_NAME,
      contextTokens: CONTEXT_TOKENS,
      maxOutputTokens: MAX_OUTPUT_TOKENS,
      temperature: 0,
    },
    adyen: {
      requiredPriorityFindings: 3,
      evidenceCount: adyenPacket.evidence_items.length,
      conflictCount: adyenPacket.conflicts.length,
      ...adyen,
    },
    brookfield_baseline: {
      requiredPriorityFindings: 2,
      evidenceCount: brookfieldPacket.evidence_items.length,
      conflictCount: brookfieldPacket.conflicts.length,
      observedWallClockMs: 101077,
      observedPromptEvalCount: 2162,
      observedEvalCount: 405,
      ...brookfield,
    },
    relative_size: {
      promptBytesAdyenVsBrookfield: ratio(
        adyen.promptBytes,
        brookfield.promptBytes,
      ),
      requestBytesAdyenVsBrookfield: ratio(
        adyen.requestBytes,
        brookfield.requestBytes,
      ),
      requiredFindingCountAdyenVsBrookfield: 1.5,
      evidenceCountAdyenVsBrookfield: ratio(
        adyenPacket.evidence_items.length,
        brookfieldPacket.evidence_items.length,
      ),
    },
    interpretation_boundary: {
      timeoutRootCauseProven: false,
      semanticQualityAssessed: false,
      modelInferenceExecuted: false,
      ollamaRequestExecuted: false,
      timeoutChangeAuthorized: false,
      retryAuthorized: false,
      nextAction:
        "Human review of exact request-size delta and Brookfield runtime baseline before proposing any timeout-only remediation.",
    },
    safety: {
      networkAccessRequested: false,
      ollamaApiCalled: false,
      modelLoaded: false,
      modelInferenceExecuted: false,
      productionMutation: false,
      publicationAuthority: false,
    },
  };

  console.log(JSON.stringify(payload, null, 2));
}

main();
