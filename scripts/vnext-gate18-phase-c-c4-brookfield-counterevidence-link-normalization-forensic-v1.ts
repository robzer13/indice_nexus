import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  assertGate18PhaseBV10Semantics,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v10";

const EXPECTED_MATRIX_CELL_ID =
  "C4_BROOKFIELD_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_BROOKFIELD_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT600_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "Brookfield Corporation";
const EXPECTED_PROMPT_SHA256 =
  "516496bf4dc9bcdce47897c056282eb2bda63b014cd63c1b76e3992b1c0cdad2";
const EXPECTED_PACKET_SHA256 =
  "eb2d779b95fa5f207e91fd3485ece39b2bd101c2589eaaf8c48481f750ef75b3";
const EXPECTED_REQUEST_SHA256 =
  "cc51153ec6bdc8041f22800aea2b6aa6418a27837589547b19db88745bc0ff8f";
const EXPECTED_SEMANTIC_ERROR =
  "VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS";

function parseArgs(argv: readonly string[]): {
  privateRepoRoot: string;
  privateArtifact: string;
} {
  let privateRepoRoot = "";
  let privateArtifact = "";

  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--private-repo-root") {
      privateRepoRoot = argv[++i] ?? "";
      continue;
    }
    if (argv[i] === "--private-artifact") {
      privateArtifact = argv[++i] ?? "";
      continue;
    }
    throw new Error(
      `VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }
  if (!privateArtifact.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_PRIVATE_ARTIFACT_REQUIRED",
    );
  }

  return { privateRepoRoot, privateArtifact };
}

function findBrookfield(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === EXPECTED_COMPANY,
  );
  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_COMPANY_NOT_UNIQUE",
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
        "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_PRIVATE_PATH_ESCAPE",
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

function main(): void {
  const { privateRepoRoot, privateArtifact } = parseArgs(
    process.argv.slice(2),
  );

  const raw = JSON.parse(
    readFileSync(resolve(privateArtifact), "utf8"),
  ) as {
    status?: string;
    invocation?: {
      matrixCellId?: string;
      attemptId?: string;
      company?: string;
      promptSha256?: string;
      fullPacketSha256?: string;
      requestSha256?: string;
    };
    execution?: {
      doneReason?: string | null;
      evalCount?: number | null;
      schemaValid?: boolean;
      schemaError?: string | null;
      semanticValid?: boolean;
      semanticError?: string | null;
    };
    response?: {
      parsedJson?: unknown;
    };
  };

  if (
    raw.status !== "FAIL" ||
    raw.invocation?.matrixCellId !== EXPECTED_MATRIX_CELL_ID ||
    raw.invocation?.attemptId !== EXPECTED_ATTEMPT_ID ||
    raw.invocation?.company !== EXPECTED_COMPANY ||
    raw.invocation?.promptSha256 !== EXPECTED_PROMPT_SHA256 ||
    raw.invocation?.fullPacketSha256 !== EXPECTED_PACKET_SHA256 ||
    raw.invocation?.requestSha256 !== EXPECTED_REQUEST_SHA256 ||
    raw.execution?.doneReason !== "stop" ||
    raw.execution?.schemaValid !== true ||
    raw.execution?.schemaError !== null ||
    raw.execution?.semanticValid !== false ||
    raw.execution?.semanticError !== EXPECTED_SEMANTIC_ERROR
  ) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_RUN_IDENTITY_MISMATCH",
    );
  }

  const parsed = gate18PhaseBV10OutputSchema.safeParse(
    raw.response?.parsedJson,
  );
  if (!parsed.success) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_SCHEMA_REPARSE_FAILED",
    );
  }

  const verified = buildVerifiedGate18V10MoatPacket(
    findBrookfield(),
    artifactReader(privateRepoRoot),
  );

  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error(
      "VNEXT_GATE18_C4_BROOKFIELD_COUNTEREVIDENCE_FORENSIC_PACKET_IDENTITY_MISMATCH",
    );
  }

  const findingSummaries = parsed.data.priority_findings.map(
    (finding, index) => ({
      index: index + 1,
      supportState: finding.support_state,
      evidenceIds: finding.evidence_ids,
      conflictIds: finding.conflict_ids,
      counterevidenceIds: finding.counterevidence_ids,
      counterevidenceLinkPresent:
        finding.counterevidence_link !== null,
      counterevidenceLinkLength:
        finding.counterevidence_link?.trim().length ?? 0,
      qualificationEvidenceIds:
        finding.evidence_qualifications.map(
          (item) => item.evidence_id,
        ),
    }),
  );

  const violatingFindingIndexes = findingSummaries
    .filter(
      (finding) =>
        finding.counterevidenceIds.length === 0 &&
        finding.counterevidenceLinkPresent,
    )
    .map((finding) => finding.index);

  const normalized = structuredClone(parsed.data);

  for (const finding of normalized.priority_findings) {
    if (
      finding.counterevidence_ids.length === 0 &&
      finding.counterevidence_link !== null
    ) {
      finding.counterevidence_link = null;
    }
  }

  let downstreamValidationPass = false;
  let downstreamValidationError: string | null = null;

  try {
    assertGate18PhaseBV10Semantics(
      verified.packet,
      normalized,
    );
    downstreamValidationPass = true;
  } catch (error) {
    downstreamValidationError =
      error instanceof Error ? error.message : String(error);
  }

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_BROOKFIELD_COUNTEREVIDENCE_LINK_NORMALIZATION_FORENSIC_V0.1",
        gate: 18,
        phase: "C_LOCAL_FIRST_MODEL_QUALIFICATION",
        stage: "C4_DETERMINISTIC_SEMANTIC_FORENSICS",
        status: "FORENSIC_COMPLETE",
        mode:
          "IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE",
        sourceRun: {
          matrixCellId: EXPECTED_MATRIX_CELL_ID,
          attemptId: EXPECTED_ATTEMPT_ID,
          doneReason: raw.execution?.doneReason ?? null,
          evalCount: raw.execution?.evalCount ?? null,
          schemaValid: raw.execution?.schemaValid ?? null,
          semanticValid: raw.execution?.semanticValid ?? null,
          semanticError: raw.execution?.semanticError ?? null,
        },
        original: {
          findingSummaries,
          violatingFindingIndexes,
          violatingFindingCount:
            violatingFindingIndexes.length,
        },
        diagnosticNormalization: {
          rule:
            "Set counterevidence_link to null only when counterevidence_ids is empty.",
          normalizedFindingIndexes:
            violatingFindingIndexes,
          sourceArtifactMutated: false,
          generatedContentPublished: false,
        },
        downstreamValidation: {
          fullValidatorExecutedOnInMemoryCopy: true,
          pass: downstreamValidationPass,
          error: downstreamValidationError,
        },
        interpretationBoundary: {
          retroactivePassAllowed: false,
          originalRunStatusChanged: false,
          inferenceExecuted: false,
          autoRepairExecuted: false,
          soleDefectConcluded:
            downstreamValidationPass &&
            violatingFindingIndexes.length > 0,
          modelCapabilityFailureConcluded: false,
          retryAuthorized: false,
          purpose:
            "Determine whether any downstream deterministic semantic defect exists after the first counterevidence-link contract failure.",
        },
        safety: {
          externalNetworkAccessRequested: false,
          ollamaApiCalled: false,
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
