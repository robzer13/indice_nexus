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

type NarrativeField = {
  path: string;
  value: string;
  set: (value: string) => void;
};

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
      `VNEXT_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_UNKNOWN_ARG:${argv[i]}`,
    );
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }
  if (!privateArtifact.trim()) {
    throw new Error(
      "VNEXT_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_PRIVATE_ARTIFACT_REQUIRED",
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
      "VNEXT_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_STMICRO_NOT_UNIQUE",
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
        "VNEXT_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_PRIVATE_PATH_ESCAPE",
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

function collectNarrativeFields(
  output: ReturnType<typeof structuredClone>,
): NarrativeField[] {
  const fields: NarrativeField[] = [];

  output.priority_findings.forEach((finding: any, findingIndex: number) => {
    fields.push({
      path: `priority_findings[${findingIndex + 1}].claim`,
      value: finding.claim,
      set: (value) => {
        finding.claim = value;
      },
    });
    fields.push({
      path: `priority_findings[${findingIndex + 1}].causal_link`,
      value: finding.causal_link,
      set: (value) => {
        finding.causal_link = value;
      },
    });

    finding.evidence_qualifications.forEach(
      (qualification: any, qualificationIndex: number) => {
        fields.push({
          path:
            `priority_findings[${findingIndex + 1}].evidence_qualifications[${qualificationIndex + 1}].qualification`,
          value: qualification.qualification,
          set: (value) => {
            qualification.qualification = value;
          },
        });
      },
    );

    if (finding.counterevidence_link !== null) {
      fields.push({
        path: `priority_findings[${findingIndex + 1}].counterevidence_link`,
        value: finding.counterevidence_link,
        set: (value) => {
          finding.counterevidence_link = value;
        },
      });
    }
  });

  output.material_conflicts.forEach((conflict: any, index: number) => {
    fields.push({
      path: `material_conflicts[${index + 1}].implication`,
      value: conflict.implication,
      set: (value) => {
        conflict.implication = value;
      },
    });
  });

  output.weak_link_candidates.forEach((candidate: any, index: number) => {
    fields.push({
      path: `weak_link_candidates[${index + 1}].why_uncertain`,
      value: candidate.why_uncertain,
      set: (value) => {
        candidate.why_uncertain = value;
      },
    });
  });

  output.unresolved_points.forEach((point: any, index: number) => {
    fields.push({
      path: `unresolved_points[${index + 1}].question`,
      value: point.question,
      set: (value) => {
        point.question = value;
      },
    });
  });

  return fields;
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
      "VNEXT_GATE18_PHI4_GLOBAL_NARRATIVE_FORENSIC_PRIVATE_OUTPUT_SCHEMA_INVALID",
    );
  }

  const verified = buildVerifiedGate18V10MoatPacket(
    findSTMicroelectronics(),
    artifactReader(privateRepoRoot),
  );

  const normalized = structuredClone(parsed.data);
  const fields = collectNarrativeFields(normalized);

  const inventory = fields.map((field) => {
    const trimmed = field.value.trim();
    return {
      path: field.path,
      length: trimmed.length,
      endsWithTerminalPunctuation: terminalPunctuation(trimmed),
      saturationBoundaryHit: trimmed.length >= 178,
    };
  });

  const normalizedPaths: string[] = [];
  const blockedBySchemaMaxPaths: string[] = [];

  for (const field of fields) {
    const trimmed = field.value.trim();
    if (terminalPunctuation(trimmed)) {
      continue;
    }
    if (trimmed.length >= 180) {
      blockedBySchemaMaxPaths.push(field.path);
      continue;
    }
    field.set(`${trimmed}.`);
    normalizedPaths.push(field.path);
  }

  const normalizedSchema = gate18PhaseBV10OutputSchema.safeParse(normalized);
  const schemaPass = normalizedSchema.success;
  const schemaError = normalizedSchema.success
    ? null
    : normalizedSchema.error.message;

  let semanticPass = false;
  let semanticError: string | null = null;

  if (normalizedSchema.success) {
    try {
      assertGate18PhaseBV10Semantics(
        verified.packet,
        normalizedSchema.data,
      );
      semanticPass = true;
    } catch (error) {
      semanticError =
        error instanceof Error ? error.message : String(error);
    }
  }

  const missingPunctuationCount = inventory.filter(
    (item) => !item.endsWithTerminalPunctuation,
  ).length;
  const saturationBoundaryCount = inventory.filter(
    (item) => item.saturationBoundaryHit,
  ).length;

  console.log(
    JSON.stringify(
      {
        format:
          "OROTITAN_GATE18_PHI4_GLOBAL_NARRATIVE_COMPLETENESS_FORENSIC_V0.1",
        status: "FORENSIC_COMPLETE",
        mode: "IN_MEMORY_DIAGNOSTIC_ONLY_NO_ARTIFACT_MUTATION_NO_INFERENCE",
        inventory: {
          fieldCount: inventory.length,
          missingPunctuationCount,
          saturationBoundaryCount,
          fields: inventory,
        },
        diagnosticNormalization: {
          normalizedPaths,
          normalizedCount: normalizedPaths.length,
          blockedBySchemaMaxPaths,
          rule:
            "Append one period only to assertCompleteNarrative-governed fields missing terminal punctuation when length < 180.",
          sourceArtifactMutated: false,
          generatedContentPublished: false,
        },
        downstreamValidation: {
          fullSchemaValidatorExecutedOnInMemoryCopy: true,
          schemaPass,
          schemaError,
          fullSemanticValidatorExecutedOnInMemoryCopy: schemaPass,
          semanticPass,
          semanticError,
        },
        strategicDiscriminator: {
          contractArchitectureReviewSignal:
            semanticPass || saturationBoundaryCount > 0,
          modelPivotSignal:
            !semanticPass &&
            semanticError !== null &&
            !/_INCOMPLETE(?:_SATURATED)?$/.test(semanticError),
          secondPhi4C4CellAuthorized: false,
        },
        interpretationBoundary: {
          retroactivePassAllowed: false,
          originalRunStatusChanged: false,
          inferenceExecuted: false,
          modelCapabilityConclusionMade: false,
          purpose:
            "Discriminate cosmetic/contract narrative-completeness failures from deeper semantic failures before any second Phi-4 C4 cell.",
        },
      },
      null,
      2,
    ),
  );
}

main();
