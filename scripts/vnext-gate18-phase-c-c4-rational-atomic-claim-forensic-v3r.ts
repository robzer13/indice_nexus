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
  "C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_001";
const EXPECTED_ATTEMPT_ID =
  "C4_RATIONAL_MOAT_EVIDENCE_AUDIT_QWEN4B_CONTEXT16384_OUTPUT1280_TIMEOUT2700_LOOPBACK_GUARDED_001";
const EXPECTED_COMPANY = "RATIONAL AG";
const EXPECTED_PROMPT_SHA256 =
  "70265015372600e619010150a72e72dad5df32973f61c01a79658994ea02c0a5";
const EXPECTED_PACKET_SHA256 =
  "3dc89f69ff39bee857595624cfe65b7772a3591fda538de05af0c757f83ba879";
const EXPECTED_REQUEST_SHA256 =
  "0132c86d5372c512b6a7628cd6f06bd8ea79e5348ee1a992b9467cea6c65fe6d";
const EXPECTED_ORIGINAL_SEMANTIC_ERROR =
  "VNEXT_GATE18_V10_CONFLICT_NOT_GROUNDED_IN_FINDING_REFS";

const NON_ATOMIC_PATTERNS = [
  { label: "but", pattern: /\bbut\b/i },
  { label: "despite", pattern: /\bdespite\b/i },
  { label: "although", pattern: /\balthough\b/i },
  { label: "though", pattern: /\bthough\b/i },
  { label: "while", pattern: /\bwhile\b/i },
  { label: "whereas", pattern: /\bwhereas\b/i },
  { label: "yet", pattern: /\byet\b/i },
  { label: "however", pattern: /\bhowever\b/i },
  { label: "nevertheless", pattern: /\bnevertheless\b/i },
  { label: "nonetheless", pattern: /\bnonetheless\b/i },
  { label: "coexist", pattern: /\bcoexist(?:s|ed|ing)?\b/i },
] as const;

function assertPatternParitySelfCheck(): void {
  const probes: Array<{ label: string; probe: string }> = [
    { label: "but", probe: "A but B." },
    { label: "despite", probe: "Despite evidence, A." },
    { label: "although", probe: "Although A, B." },
    { label: "though", probe: "Though A, B." },
    { label: "while", probe: "While A, B." },
    { label: "whereas", probe: "A whereas B." },
    { label: "yet", probe: "A yet B." },
    { label: "however", probe: "However, A." },
    { label: "nevertheless", probe: "Nevertheless, A." },
    { label: "nonetheless", probe: "Nonetheless, A." },
    { label: "coexist", probe: "A and B coexist." },
  ];

  for (const { label, pattern } of NON_ATOMIC_PATTERNS) {
    const probe = probes.find((item) => item.label === label)?.probe;
    if (!probe || !pattern.test(probe)) {
      throw new Error(
        `VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_PATTERN_SELF_CHECK_FAILED:${label}`,
      );
    }
  }
}

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
      `VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }
  if (!privateArtifact.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_PRIVATE_ARTIFACT_REQUIRED",
    );
  }

  return { privateRepoRoot, privateArtifact };
}

function findRational(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === EXPECTED_COMPANY,
  );
  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_COMPANY_NOT_UNIQUE",
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
        "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_PRIVATE_PATH_ESCAPE",
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
  assertPatternParitySelfCheck();

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
    raw.execution?.semanticError !== EXPECTED_ORIGINAL_SEMANTIC_ERROR
  ) {
    throw new Error(
      "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_RUN_IDENTITY_MISMATCH",
    );
  }

  const parsed = gate18PhaseBV10OutputSchema.safeParse(
    raw.response?.parsedJson,
  );
  if (!parsed.success) {
    throw new Error(
      "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_SCHEMA_REPARSE_FAILED",
    );
  }

  const verified = buildVerifiedGate18V10MoatPacket(
    findRational(),
    artifactReader(privateRepoRoot),
  );

  if (verified.packetSha256 !== EXPECTED_PACKET_SHA256) {
    throw new Error(
      "VNEXT_GATE18_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_PACKET_IDENTITY_MISMATCH",
    );
  }

  const conflictsById = new Map(
    verified.packet.conflicts.map((conflict) => [
      conflict.conflict_id,
      conflict,
    ]),
  );

  const originalFindingConflictGrounding =
    parsed.data.priority_findings.map((finding, index) => {
      const findingRefs = new Set([
        ...finding.evidence_ids,
        ...finding.counterevidence_ids,
      ]);

      return {
        findingIndex: index + 1,
        evidenceIds: finding.evidence_ids,
        counterevidenceIds: finding.counterevidence_ids,
        conflictGrounding: finding.conflict_ids.map((conflictId) => {
          const conflict = conflictsById.get(conflictId);
          const conflictEvidenceRefs = conflict?.evidence_refs ?? [];
          const overlap = conflictEvidenceRefs.filter((id) =>
            findingRefs.has(id),
          );
          return {
            conflictId,
            conflictEvidenceRefs,
            overlapWithFindingRefs: overlap,
            grounded:
              conflictEvidenceRefs.length === 0 ||
              overlap.length > 0,
          };
        }),
      };
    });

  const originalDirectionRoleOverlaps =
    parsed.data.priority_findings
      .map((finding, index) => {
        const supportIds = new Set(finding.evidence_ids);
        const overlap = finding.counterevidence_ids.filter((id) =>
          supportIds.has(id),
        );
        return {
          findingIndex: index + 1,
          overlapEvidenceIds: overlap,
        };
      })
      .filter((item) => item.overlapEvidenceIds.length > 0);

  const normalized = structuredClone(parsed.data);
  const removedConflictRefs: Array<{
    findingIndex: number;
    conflictId: string;
    conflictEvidenceRefs: string[];
    findingEvidenceRefs: string[];
  }> = [];

  normalized.priority_findings.forEach((finding, findingIndex) => {
    const findingRefs = new Set([
      ...finding.evidence_ids,
      ...finding.counterevidence_ids,
    ]);

    finding.conflict_ids = finding.conflict_ids.filter(
      (conflictId) => {
        const conflict = conflictsById.get(conflictId);
        if (!conflict || conflict.evidence_refs.length === 0) {
          return true;
        }

        const grounded = conflict.evidence_refs.some((id) =>
          findingRefs.has(id),
        );

        if (!grounded) {
          removedConflictRefs.push({
            findingIndex: findingIndex + 1,
            conflictId,
            conflictEvidenceRefs: [...conflict.evidence_refs],
            findingEvidenceRefs: [...findingRefs],
          });
        }

        return grounded;
      },
    );
  });

  const removedSupportOverlapRefs: Array<{
    findingIndex: number;
    evidenceId: string;
    retainedRole: "counterevidence";
  }> = [];

  normalized.priority_findings.forEach((finding, findingIndex) => {
    const counterIds = new Set(finding.counterevidence_ids);
    finding.evidence_ids = finding.evidence_ids.filter((id) => {
      if (!counterIds.has(id)) {
        return true;
      }

      removedSupportOverlapRefs.push({
        findingIndex: findingIndex + 1,
        evidenceId: id,
        retainedRole: "counterevidence",
      });
      return false;
    });
  });

  const nonAtomicFindings: Array<{
    findingIndex: number;
    matchedPatterns: string[];
  }> = [];

  normalized.priority_findings.forEach((finding, findingIndex) => {
    const matchedPatterns = NON_ATOMIC_PATTERNS
      .filter(({ pattern }) => pattern.test(finding.claim))
      .map(({ label }) => label);

    if (matchedPatterns.length > 0) {
      nonAtomicFindings.push({
        findingIndex: findingIndex + 1,
        matchedPatterns,
      });
      finding.claim = "Diagnostic atomic claim.";
    }
  });

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
          "OROTITAN_GATE18_PHASE_C_C4_RATIONAL_ATOMIC_CLAIM_FORENSIC_V3R_V0.1",
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
          findingConflictGrounding: originalFindingConflictGrounding,
          directionRoleOverlaps: originalDirectionRoleOverlaps,
        },
        diagnosticNormalization: {
          conflictGroundingRule:
            "Remove a finding conflict_id only when the packet conflict has evidence_refs and none overlap the finding support/counterevidence refs.",
          removedConflictRefs,
          removedConflictRefCount: removedConflictRefs.length,
          directionRoleOverlapRule:
            "When an evidence_id appears in both evidence_ids and counterevidence_ids, remove it only from evidence_ids and retain the explicit counterevidence role on the in-memory diagnostic copy.",
          removedSupportOverlapRefs,
          removedSupportOverlapRefCount: removedSupportOverlapRefs.length,
          nonAtomicClaimRule:
            "Replace only claims matching the frozen contrastive non-atomic patterns with a short atomic diagnostic placeholder on the in-memory copy.",
          nonAtomicFindings,
          atomicClaimReplacement: "Diagnostic atomic claim.",
          rawClaimTextIncludedInConsole: false,
          rawNarrativeTextIncludedInConsole: false,
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
          allKnownDeterministicDefectsExhausted:
            downstreamValidationPass,
          modelCapabilityFailureConcluded: false,
          retryAuthorized: false,
          purpose:
            "Reveal whether any downstream deterministic semantic defect remains after cumulative diagnostic normalization of finding-conflict grounding, direction-role overlap, and non-atomic contrastive claims.",
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
