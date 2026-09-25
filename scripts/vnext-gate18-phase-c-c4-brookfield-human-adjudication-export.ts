import { createHash } from "node:crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  buildGate18PhaseBV10ModelInput,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v10";

const EXPECTED_ATTEMPT_ID =
  "C4_BROOKFIELD_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_MATRIX_CELL_ID =
  "C4_BROOKFIELD_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001";
const EXPECTED_MODEL = "qwen3:4b-instruct";
const EXPECTED_MODEL_DIGEST =
  "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0";
const EXPECTED_PROMPT_SHA256 =
  "516496bf4dc9bcdce47897c056282eb2bda63b014cd63c1b76e3992b1c0cdad2";
const EXPECTED_PACKET_SHA256 =
  "eb2d779b95fa5f207e91fd3485ece39b2bd101c2589eaaf8c48481f750ef75b3";
const EXPECTED_REQUEST_SHA256 =
  "cc51153ec6bdc8041f22800aea2b6aa6418a27837589547b19db88745bc0ff8f";
const EXPECTED_SEMANTIC_ERROR =
  "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS";

interface CliOptions {
  privateRepoRoot: string;
  privateOutputPath: string;
  outputPath: string | null;
}

interface PrivateRunArtifact {
  status?: string;
  invocation?: {
    matrixCellId?: string;
    attemptId?: string;
    company?: string;
    model?: string;
    modelDigest?: string;
    promptSha256?: string;
    fullPacketSha256?: string;
    requestSha256?: string;
  };
  localRuntime?: {
    transport?: string;
    globalFetchUsed?: boolean;
    hiddenUndiciHeadersTimeoutConfigured?: boolean;
    generation?: {
      contextTokens?: number;
      maxOutputTokens?: number;
      temperature?: number;
      clientTimeoutMs?: number;
    };
  };
  execution?: {
    wallClockMs?: number;
    doneReason?: string | null;
    promptEvalCount?: number | null;
    evalCount?: number | null;
    runtimeError?: string | null;
    schemaValid?: boolean;
    schemaError?: string | null;
    semanticValid?: boolean;
    semanticError?: string | null;
  };
  response?: {
    parsedJson?: unknown;
  };
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function parseArgs(argv: readonly string[]): CliOptions {
  let privateRepoRoot = "";
  let privateOutputPath = "";
  let outputPath: string | null = null;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--private-repo-root") {
      privateRepoRoot = argv[++index] ?? "";
      continue;
    }

    if (arg === "--private-output-path") {
      privateOutputPath = argv[++index] ?? "";
      continue;
    }

    if (arg === "--output-path") {
      outputPath = argv[++index] ?? null;
      continue;
    }

    throw new Error(
      `VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_UNKNOWN_ARG:${arg}`,
    );
  }

  if (privateRepoRoot.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_PRIVATE_REPO_REQUIRED",
    );
  }

  if (privateOutputPath.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_PRIVATE_RUN_REQUIRED",
    );
  }

  return { privateRepoRoot, privateOutputPath, outputPath };
}

function findBrookfieldCorporation(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === "Brookfield Corporation",
  );

  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_COMPANY_NOT_UNIQUE",
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
        "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_PRIVATE_PATH_ESCAPE",
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

function deriveOutputPath(privateOutputPath: string): string {
  const absolute = resolve(privateOutputPath);
  const suffix = ".json";
  const stem = absolute.endsWith(suffix)
    ? absolute.slice(0, -suffix.length)
    : absolute;

  return `${stem}__HUMAN_ADJUDICATION_PACKET.json`;
}

function main(): void {
  const options = parseArgs(process.argv.slice(2));
  const privateOutputAbsolute = resolve(options.privateOutputPath);

  if (!existsSync(privateOutputAbsolute)) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_PRIVATE_RUN_NOT_FOUND",
    );
  }

  const run = JSON.parse(
    readFileSync(privateOutputAbsolute, "utf8"),
  ) as PrivateRunArtifact;

  if (
    run.status !== "FAIL" ||
    run.invocation?.matrixCellId !== EXPECTED_MATRIX_CELL_ID ||
    run.invocation?.attemptId !== EXPECTED_ATTEMPT_ID ||
    run.invocation?.company !== "Brookfield Corporation" ||
    run.invocation?.model !== EXPECTED_MODEL ||
    run.invocation?.modelDigest !== EXPECTED_MODEL_DIGEST ||
    run.invocation?.promptSha256 !== EXPECTED_PROMPT_SHA256 ||
    run.invocation?.requestSha256 !== EXPECTED_REQUEST_SHA256 ||
    run.localRuntime?.transport !== "NODE_HTTP_REQUEST_LOOPBACK" ||
    run.execution?.runtimeError !== null ||
    run.execution?.schemaValid !== true ||
    run.execution?.semanticValid !== false ||
    run.execution?.semanticError !== EXPECTED_SEMANTIC_ERROR
  ) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_PRIVATE_RUN_IDENTITY_OR_ENGINEERING_FAIL_MISMATCH",
    );
  }

  const parsedOutput = gate18PhaseBV10OutputSchema.parse(
    run.response?.parsedJson,
  );

  const company = findBrookfieldCorporation();
  const verified = buildVerifiedGate18V10MoatPacket(
    company,
    artifactReader(options.privateRepoRoot),
  );
  const modelInput = buildGate18PhaseBV10ModelInput(
    verified.packet,
  );
  const rebuiltPromptSha256 = sha256Hex(modelInput);

  if (
    verified.packetSha256 !== EXPECTED_PACKET_SHA256 ||
    verified.packetSha256 !== run.invocation?.fullPacketSha256 ||
    rebuiltPromptSha256 !== run.invocation?.promptSha256
  ) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_ADJUDICATION_EXPORT_REBUILT_PACKET_IDENTITY_MISMATCH",
    );
  }

  const outputPath = resolve(
    options.outputPath ?? deriveOutputPath(privateOutputAbsolute),
  );
  mkdirSync(dirname(outputPath), { recursive: true });

  const reviewPacket = {
    format:
      "OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_HUMAN_ADJUDICATION_PACKET_V0.1",
    privateArtifact: true,
    publication: false,
    productionMutation: false,
    gate: 18,
    phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
    stage: "C4_HUMAN_QUALITY_ADJUDICATION",
    status: "READY_FOR_HUMAN_ADJUDICATION_ENGINEERING_FAIL",
    sourceRun: {
      matrixCellId: EXPECTED_MATRIX_CELL_ID,
      attemptId: EXPECTED_ATTEMPT_ID,
      company: "Brookfield Corporation",
      model: EXPECTED_MODEL,
      modelDigest: EXPECTED_MODEL_DIGEST,
      fullPacketSha256: verified.packetSha256,
      promptSha256: rebuiltPromptSha256,
      execution: run.execution,
      transport: run.localRuntime,
    },
    engineeringObservations: {
      runtimePass: true,
      schemaValid: true,
      semanticValidatorPass: false,
      deterministicSemanticFailure: true,
      semanticError: run.execution?.semanticError ?? null,
      deterministicDefectClasses: [
        "COUNTEREVIDENCE_LINK_WITHOUT_IDS",
        "FINDING_CONFLICT_ID_NOT_GROUNDED_IN_FINDING_EVIDENCE_REFS",
        "NON_ATOMIC_CONTRASTIVE_CLAIM",
      ],
      deterministicDefectSetExhaustedByForensics: true,
      outputBudgetFullyConsumed:
        run.execution?.doneReason === "length" &&
        run.execution?.evalCount ===
          run.localRuntime?.generation?.maxOutputTokens,
      structuralTruncationObserved: false,
      semanticTruncationObserved: false,
    },
    humanQualityCriteria: [
      "exact_evidence_grounding",
      "claim_atomicity",
      "claim_target_alignment",
      "support_counterevidence_polarity",
      "qualification_orthogonality",
      "conflict_handling",
      "weak_link_usefulness",
      "unresolved_point_usefulness",
      "judgment_boundary_compliance",
      "priority_selection_usefulness",
    ],
    evidencePacket: verified.packet,
    modelOutput: parsedOutput,
    adjudicationPolicy: {
      originalOutputMustBeAdjudicatedAsEmitted: true,
      diagnosticNormalizationsMustNotBeAppliedToReviewPacket: true,
      autoRepairForbidden: true,
      finalMoatConclusionForbidden: true,
      generatedContentMustRemainPrivate: true,
      publicRepoPersistenceForbidden: true,
    },
  };

  writeFileSync(
    outputPath,
    JSON.stringify(reviewPacket, null, 2) + "\n",
    "utf8",
  );

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_HUMAN_ADJUDICATION_EXPORT_SUMMARY_V0.1",
        status: "PASS_REVIEW_PACKET_READY_ENGINEERING_FAIL",
        reviewPacketPath: outputPath,
        evidenceCount: verified.packet.evidence_items.length,
        conflictCount: verified.packet.conflicts.length,
        priorityFindingCount:
          parsedOutput.priority_findings.length,
        materialConflictCount:
          parsedOutput.material_conflicts.length,
        weakLinkCount:
          parsedOutput.weak_link_candidates.length,
        unresolvedPointCount:
          parsedOutput.unresolved_points.length,
        packetIdentityVerified: true,
        promptIdentityVerified: true,
        noInferenceExecuted: true,
        publicRepoPersistence: false,
        publicationAuthority: false,
      },
      null,
      2,
    ),
  );
}

main();
