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
      `VNEXT_GATE18_C4_STMICRO_PHI4_FINDING_CLAIM_FORENSIC_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_STMICRO_PHI4_FINDING_CLAIM_FORENSIC_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }
  if (!privateArtifact.trim()) {
    throw new Error(
      "VNEXT_GATE18_C4_STMICRO_PHI4_FINDING_CLAIM_FORENSIC_PRIVATE_ARTIFACT_REQUIRED",
    );
  }

  return { privateRepoRoot, privateArtifact };
}

function findSTMicroelectronics(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === "STMicroelectronics",
  );
  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C4_STMICRO_PHI4_FINDING_CLAIM_FORENSIC_STMICRO_NOT_UNIQUE",
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
        "VNEXT_GATE18_C4_STMICRO_PHI4_FINDING_CLAIM_FORENSIC_PRIVATE_PATH_ESCAPE",
      );
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

function terminalPunctuation(value: string): boolean {
  return /[.!?]$/.test(value.trim());
}

function main(): void {
  const { privateRepoRoot, privateArtifact } = parseArgs(
    process.argv.slice(2),
  );

  const raw = JSON.parse(readFileSync(privateArtifact, "utf8")) as {
    response?: { parsedJson?: unknown };
  };

  const parsed = gate18PhaseBV10OutputSchema.safeParse(
    raw.response?.parsedJson,
  );
  if (!parsed.success) {
    throw new Error(
      "VNEXT_GATE18_C4_STMICRO_PHI4_FINDING_CLAIM_FORENSIC_PRIVATE_OUTPUT_SCHEMA_INVALID",
    );
  }

  const verified = buildVerifiedGate18V10MoatPacket(
    findSTMicroelectronics(),
    artifactReader(privateRepoRoot),
  );

  const originalFindingSummary = parsed.data.priority_findings.map(
    (item, index) => ({
      findingIndex: index + 1,
      claimLength: item.claim.trim().length,
      endsWithTerminalPunctuation: terminalPunctuation(item.claim),
    }),
  );

  const normalized = structuredClone(parsed.data);
  const normalizedFindingIndexes: number[] = [];

  normalized.priority_findings.forEach((finding, index) => {
    const trimmed = finding.claim.trim();
    if (!terminalPunctuation(trimmed)) {
      finding.claim = `${trimmed}.`;
      normalizedFindingIndexes.push(index + 1);
    }
  });

  const normalizedSchema = gate18PhaseBV10OutputSchema.safeParse(normalized);
  const normalizedSchemaValid = normalizedSchema.success;
  const normalizedSchemaError = normalizedSchema.success
    ? null
    : normalizedSchema.error.message;

  let downstreamValidationPass = false;
  let downstreamValidationError: string | null = null;

  if (normalizedSchema.success) {
    try {
      assertGate18PhaseBV10Semantics(
        verified.packet,
        normalizedSchema.data,
      );
      downstreamValidationPass = true;
    } catch (error) {
      downstreamValidationError =
        error instanceof Error ? error.message : String(error);
    }
  }

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C4_STMICRO_PHI4_MINI_FINDING_CLAIM_PUNCTUATION_FORENSIC_V0.1",
        status: "FORENSIC_COMPLETE",
        mode: "IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE",
        original: {
          findingSummary: originalFindingSummary,
        },
        diagnosticNormalization: {
          normalizedFindingIndexes,
          normalizationRule:
            "Append one period only when priority_findings[*].claim terminal punctuation is absent.",
          sourceArtifactMutated: false,
          generatedContentPublished: false,
        },
        downstreamValidation: {
          fullSchemaValidatorExecutedOnInMemoryCopy: true,
          schemaPass: normalizedSchemaValid,
          schemaError: normalizedSchemaError,
          fullSemanticValidatorExecutedOnInMemoryCopy: normalizedSchemaValid,
          semanticPass: downstreamValidationPass,
          semanticError: downstreamValidationError,
        },
        interpretationBoundary: {
          retroactivePassAllowed: false,
          originalRunStatusChanged: false,
          inferenceExecuted: false,
          modelCapabilityConclusionMade: false,
          purpose:
            "Reveal whether any downstream semantic or output-contract defect exists after finding-claim punctuation-only normalization.",
        },
      },
      null,
      2,
    ),
  );
}

main();
