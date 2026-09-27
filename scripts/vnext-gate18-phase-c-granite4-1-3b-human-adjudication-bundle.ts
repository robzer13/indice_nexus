import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  buildVerifiedGate18V10MoatPacket,
  buildGate18PhaseBV10ModelInput,
} from "../runtime/vnext/model-calibration-pilot-v10";

const EXPECTED_MODEL = "granite4.1:3b-q4_K_M";
const EXPECTED_DIGEST =
  "6fd349357287c7ffc9e38189a93b48ea175d24fc566b38f09cfc564fb7f303eb";
const EXPECTED_PACKET_SHA256 =
  "9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8";
const EXPECTED_PROMPT_SHA256 =
  "0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8";
const EXPECTED_SOURCE_BASENAME =
  "2026-09-27T214331230Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_GRANITE4_1_3B_V1_1_CONTEXT16384_001.json";

interface CliOptions {
  privateRepoRoot: string;
  sourceRunPath: string;
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateRepoRoot = "";
  let sourceRunPath = "";

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--private-repo-root") {
      privateRepoRoot = argv[++i] ?? "";
      continue;
    }
    if (arg === "--source-run") {
      sourceRunPath = argv[++i] ?? "";
      continue;
    }
    throw new Error(`VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_UNKNOWN_ARG:${arg}`);
  }

  if (!privateRepoRoot.trim()) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_PRIVATE_REPO_ROOT_REQUIRED");
  }
  if (!sourceRunPath.trim()) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_SOURCE_RUN_REQUIRED");
  }

  return { privateRepoRoot, sourceRunPath };
}

function findConstellationSoftware(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === "Constellation Software",
  );
  if (matches.length !== 1) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_CONSTELLATION_NOT_UNIQUE");
  }
  return matches[0] as Gate18PilotCompany;
}

function artifactReader(
  privateRepoRoot: string,
): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = resolve(privateRepoRoot).replaceAll("\\", "/");
    const normalizedPrefix = normalizedRoot.endsWith("/")
      ? normalizedRoot
      : `${normalizedRoot}/`;

    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");

    if (!normalizedAbsolute.startsWith(normalizedPrefix)) {
      throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_PRIVATE_PATH_ESCAPE");
    }

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

function sourceBasename(path: string): string {
  return path.replaceAll("\\", "/").split("/").at(-1) ?? "";
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const sourcePath = resolve(process.cwd(), options.sourceRunPath);

  if (!existsSync(sourcePath)) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_SOURCE_RUN_MISSING");
  }
  if (sourceBasename(sourcePath) !== EXPECTED_SOURCE_BASENAME) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_SOURCE_RUN_IDENTITY_MISMATCH");
  }

  const source = JSON.parse(readFileSync(sourcePath, "utf8")) as Record<string, any>;

  if (
    source.invocation?.model !== EXPECTED_MODEL ||
    source.invocation?.modelDigest !== EXPECTED_DIGEST ||
    source.invocation?.fullPacketSha256 !== EXPECTED_PACKET_SHA256 ||
    source.invocation?.promptSha256 !== EXPECTED_PROMPT_SHA256
  ) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_SOURCE_METADATA_MISMATCH");
  }
  if (
    source.execution?.runtimeError !== null ||
    source.execution?.schemaValid !== true ||
    source.execution?.semanticValid !== true
  ) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_SOURCE_NOT_ENGINEERING_PASS");
  }

  const company = findConstellationSoftware();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(options.privateRepoRoot),
  );
  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_PACKET_SHA_MISMATCH");
  }

  const modelInput = buildGate18PhaseBV10ModelInput(verified.packet);
  if (sha256Hex(modelInput) !== EXPECTED_PROMPT_SHA256) {
    throw new Error("VNEXT_GATE18_GRANITE4_1_3B_HUMAN_BUNDLE_PROMPT_SHA_MISMATCH");
  }

  const bundle = {
    format: "OROTITAN_GATE18_PHASE_C_C4_GRANITE4_1_3B_PRIVATE_HUMAN_ADJUDICATION_BUNDLE_V0.1",
    privateArtifact: true,
    publication: false,
    sourceRunPath: options.sourceRunPath,
    company: "Constellation Software",
    model: {
      name: EXPECTED_MODEL,
      digest: EXPECTED_DIGEST,
    },
    engineering: {
      status: "PASS",
      runtimeError: source.execution.runtimeError,
      schemaValid: source.execution.schemaValid,
      semanticValid: source.execution.semanticValid,
      doneReason: source.execution.doneReason,
      promptEvalCount: source.execution.promptEvalCount,
      evalCount: source.execution.evalCount,
      validationV11: source.validationV11,
    },
    response: source.response,
    packet: verified.packet,
    reviewChecklist: {
      exactEvidenceGrounding: true,
      claimAtomicity: true,
      claimTargetAlignment: true,
      supportCounterevidencePolarity: true,
      qualificationOrthogonality: true,
      conflictHandling: true,
      weakLinkUsefulness: true,
      unresolvedPointUsefulness: true,
      judgmentBoundaryCompliance: true,
      prioritySelectionUsefulness: true,
      samePacketRegressionChecks: [
        "E-045 RFP/replacement-process overstatement",
        "E-029 acquisition-criteria universalization",
        "E-063 already-answered retention question",
        "near-duplicate recurring-revenue priority slots",
        "omission of E-066/E-067 direct serial-acquirer inputs",
        "suppression or mishandling of C-005",
      ],
    },
    safety: {
      inferenceExecuted: false,
      sourceArtifactMutated: false,
      privatePacketPublished: false,
      rawOutputPublished: false,
      productionMutation: false,
    },
  };

  const outputDir = resolve(process.cwd(), "calibration/vnext/private-runs");
  mkdirSync(outputDir, { recursive: true });
  const outputPath = resolve(
    outputDir,
    "OROTITAN_GATE18_PHASE_C_C4_GRANITE4_1_3B_PRIVATE_HUMAN_ADJUDICATION_BUNDLE_001.json",
  );
  writeFileSync(outputPath, JSON.stringify(bundle, null, 2) + "\n", "utf8");

  console.log(
    JSON.stringify(
      {
        format: "OROTITAN_GATE18_PHASE_C_C4_GRANITE4_1_3B_HUMAN_ADJUDICATION_BUNDLE_SUMMARY_V0.1",
        status: "PASS_PRIVATE_BUNDLE_READY",
        outputPath,
        sourceRunPath: options.sourceRunPath,
        packetSha256: verified.packetSha256,
        promptSha256: EXPECTED_PROMPT_SHA256,
        inferenceExecuted: false,
        productionMutation: false,
        publicationAuthority: false,
        nextAction: "UPLOAD_PRIVATE_BUNDLE_FOR_HUMAN_ADJUDICATION",
      },
      null,
      2,
    ),
  );
}

try {
  main();
} catch (error) {
  console.error(
    JSON.stringify(
      {
        format: "OROTITAN_GATE18_PHASE_C_C4_GRANITE4_1_3B_HUMAN_ADJUDICATION_BUNDLE_SUMMARY_V0.1",
        status: "BLOCKED",
        error: error instanceof Error ? error.message : String(error),
        inferenceExecuted: false,
        productionMutation: false,
        publicationAuthority: false,
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
}
