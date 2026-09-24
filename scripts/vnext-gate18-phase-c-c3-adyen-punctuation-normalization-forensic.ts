import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import type { Gate18V02EvidencePacket } from "../runtime/vnext/model-calibration-pilot-v02";
import {
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v10";
import {
  assertGate18V10TargetedProbeSemantics,
} from "../runtime/vnext/model-calibration-targeted-regression-v10";

const REQUIRED_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-040",
  "E-041",
  "E-042",
  "E-043",
  "E-055",
  "E-056",
] as const;

const REQUIRED_CONFLICT_IDS = ["C-005", "C-010"] as const;

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
      `VNEXT_GATE18_C3_ADYEN_FORENSIC_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_C3_ADYEN_FORENSIC_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }
  if (!privateArtifact.trim()) {
    throw new Error(
      "VNEXT_GATE18_C3_ADYEN_FORENSIC_PRIVATE_ARTIFACT_REQUIRED",
    );
  }

  return { privateRepoRoot, privateArtifact };
}

function findAdyen(): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === "Adyen",
  );
  if (matches.length !== 1) {
    throw new Error(
      "VNEXT_GATE18_C3_ADYEN_FORENSIC_ADYEN_NOT_UNIQUE",
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
        "VNEXT_GATE18_C3_ADYEN_FORENSIC_PRIVATE_PATH_ESCAPE",
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

function compactAdyen(
  source: Gate18V02EvidencePacket,
): Gate18V02EvidencePacket {
  const evidenceItems = source.evidence_items.filter((item) =>
    REQUIRED_EVIDENCE_IDS.includes(
      item.evidence_id as (typeof REQUIRED_EVIDENCE_IDS)[number],
    ),
  );
  const conflicts = source.conflicts.filter((item) =>
    REQUIRED_CONFLICT_IDS.includes(
      item.conflict_id as (typeof REQUIRED_CONFLICT_IDS)[number],
    ),
  );

  for (const id of REQUIRED_EVIDENCE_IDS) {
    if (!evidenceItems.some((item) => item.evidence_id === id)) {
      throw new Error(
        `VNEXT_GATE18_C3_ADYEN_FORENSIC_EVIDENCE_MISSING:${id}`,
      );
    }
  }
  for (const id of REQUIRED_CONFLICT_IDS) {
    if (!conflicts.some((item) => item.conflict_id === id)) {
      throw new Error(
        `VNEXT_GATE18_C3_ADYEN_FORENSIC_CONFLICT_MISSING:${id}`,
      );
    }
  }

  return {
    ...source,
    evidence_items: evidenceItems,
    conflicts,
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
      "VNEXT_GATE18_C3_ADYEN_FORENSIC_PRIVATE_OUTPUT_SCHEMA_INVALID",
    );
  }

  const verified = buildVerifiedGate18V10MoatPacket(
    findAdyen(),
    artifactReader(privateRepoRoot),
  );
  const packet = compactAdyen(verified.packet);

  const originalConflictSummary = parsed.data.material_conflicts.map(
    (item) => ({
      conflictId: item.conflict_id,
      resolutionState: item.resolution_state,
      implicationLength: item.implication.trim().length,
      endsWithTerminalPunctuation: terminalPunctuation(
        item.implication,
      ),
    }),
  );

  const normalized = structuredClone(parsed.data);
  const normalizedConflictIds: string[] = [];

  for (const conflict of normalized.material_conflicts) {
    const trimmed = conflict.implication.trim();
    if (!terminalPunctuation(trimmed)) {
      conflict.implication = `${trimmed}.`;
      normalizedConflictIds.push(conflict.conflict_id);
    }
  }

  let downstreamValidationPass = false;
  let downstreamValidationError: string | null = null;

  try {
    assertGate18V10TargetedProbeSemantics(packet, normalized);
    downstreamValidationPass = true;
  } catch (error) {
    downstreamValidationError =
      error instanceof Error ? error.message : String(error);
  }

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHASE_C_C3_ADYEN_PUNCTUATION_NORMALIZATION_FORENSIC_V0.1",
        status: "FORENSIC_COMPLETE",
        mode: "IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE",
        original: {
          conflictSummary: originalConflictSummary,
        },
        diagnosticNormalization: {
          normalizedConflictIds,
          normalizationRule:
            "Append one period only when terminal punctuation is absent.",
          sourceArtifactMutated: false,
          generatedContentPublished: false,
        },
        downstreamValidation: {
          fullTargetedValidatorExecutedOnInMemoryCopy: true,
          pass: downstreamValidationPass,
          error: downstreamValidationError,
        },
        interpretationBoundary: {
          retroactivePassAllowed: false,
          originalRunStatusChanged: false,
          inferenceExecuted: false,
          modelCapabilityConclusionMade: false,
          purpose:
            "Reveal whether any downstream semantic or output-contract defect exists after the first punctuation-only failure.",
        },
      },
      null,
      2,
    ),
  );
}

main();
