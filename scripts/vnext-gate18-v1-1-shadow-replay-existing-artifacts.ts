import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { execFileSync } from "node:child_process";

import manifestJson from "../calibration/vnext/OROTITAN_GATE18_V1_1_SHADOW_REPLAY_MANIFEST_001.json";
import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import type {
  Gate18ArtifactPin,
  Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import type { Gate18V02EvidencePacket } from "../runtime/vnext/model-calibration-pilot-v02";
import {
  assertGate18PhaseBV10Semantics,
  buildVerifiedGate18V10MoatPacket,
  gate18PhaseBV10OutputSchema,
} from "../runtime/vnext/model-calibration-pilot-v10";
import { assertGate18V10TargetedProbeSemantics } from "../runtime/vnext/model-calibration-targeted-regression-v10";

const SAFE_NARRATIVE_MAX_EXCLUSIVE = 178;

const ADYEN_REQUIRED_EVIDENCE_IDS = [
  "E-036",
  "E-037",
  "E-040",
  "E-041",
  "E-042",
  "E-043",
  "E-055",
  "E-056",
] as const;

const ADYEN_REQUIRED_CONFLICT_IDS = ["C-005", "C-010"] as const;

type ManifestCase = (typeof manifestJson.cases)[number];

type NarrativeField = {
  path: string;
  value: string;
  set: (value: string) => void;
};

function parseArgs(argv: readonly string[]): { privateRepoRoot: string } {
  let privateRepoRoot = "";

  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === "--private-repo-root") {
      privateRepoRoot = argv[++i] ?? "";
      continue;
    }
    throw new Error(`VNEXT_GATE18_V1_1_SHADOW_REPLAY_UNKNOWN_ARG:${argv[i]}`);
  }

  if (!privateRepoRoot.trim()) {
    throw new Error(
      "VNEXT_GATE18_V1_1_SHADOW_REPLAY_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }

  return { privateRepoRoot };
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
      throw new Error("VNEXT_GATE18_V1_1_SHADOW_REPLAY_PRIVATE_PATH_ESCAPE");
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

function findCompany(displayName: string): Gate18PilotCompany {
  const matches = pilotJson.companies.filter(
    (company) => company.display_name === displayName,
  );
  if (matches.length !== 1) {
    throw new Error(
      `VNEXT_GATE18_V1_1_SHADOW_REPLAY_COMPANY_NOT_UNIQUE:${displayName}`,
    );
  }
  return matches[0] as Gate18PilotCompany;
}

function compactAdyen(
  source: Gate18V02EvidencePacket,
): Gate18V02EvidencePacket {
  const evidenceItems = source.evidence_items.filter((item) =>
    ADYEN_REQUIRED_EVIDENCE_IDS.includes(
      item.evidence_id as (typeof ADYEN_REQUIRED_EVIDENCE_IDS)[number],
    ),
  );
  const conflicts = source.conflicts.filter((item) =>
    ADYEN_REQUIRED_CONFLICT_IDS.includes(
      item.conflict_id as (typeof ADYEN_REQUIRED_CONFLICT_IDS)[number],
    ),
  );

  for (const id of ADYEN_REQUIRED_EVIDENCE_IDS) {
    if (!evidenceItems.some((item) => item.evidence_id === id)) {
      throw new Error(
        `VNEXT_GATE18_V1_1_SHADOW_REPLAY_ADYEN_EVIDENCE_MISSING:${id}`,
      );
    }
  }
  for (const id of ADYEN_REQUIRED_CONFLICT_IDS) {
    if (!conflicts.some((item) => item.conflict_id === id)) {
      throw new Error(
        `VNEXT_GATE18_V1_1_SHADOW_REPLAY_ADYEN_CONFLICT_MISSING:${id}`,
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

function collectNarrativeFields(output: any): NarrativeField[] {
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

function validateCase(
  manifestCase: ManifestCase,
  packet: Gate18V02EvidencePacket,
  output: any,
): { pass: boolean; error: string | null } {
  try {
    if (manifestCase.validator_mode === "TARGETED_ADYEN_COMPACT") {
      assertGate18V10TargetedProbeSemantics(packet, output);
    } else {
      assertGate18PhaseBV10Semantics(packet, output);
    }
    return { pass: true, error: null };
  } catch (error) {
    return {
      pass: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function main(): void {
  const { privateRepoRoot } = parseArgs(process.argv.slice(2));
  const projectRoot = process.cwd();

  const results = manifestJson.cases.map((manifestCase) => {
    const artifactPath = resolve(projectRoot, manifestCase.artifact_path);
    if (!existsSync(artifactPath)) {
      throw new Error(
        `VNEXT_GATE18_V1_1_SHADOW_REPLAY_ARTIFACT_MISSING:${manifestCase.case_id}`,
      );
    }

    const rawArtifact = JSON.parse(readFileSync(artifactPath, "utf8")) as {
      response?: { parsedJson?: unknown };
    };

    const parsed = gate18PhaseBV10OutputSchema.safeParse(
      rawArtifact.response?.parsedJson,
    );
    if (!parsed.success) {
      return {
        caseId: manifestCase.case_id,
        model: manifestCase.model,
        company: manifestCase.company,
        validatorMode: manifestCase.validator_mode,
        recordedV10Status: manifestCase.recorded_status,
        recordedV10SemanticError: manifestCase.recorded_semantic_error,
        rawSchemaPass: false,
        rawSchemaError: parsed.error.message,
        rawPresentationCompliance: null,
        rawValidatorReplay: null,
        normalizedCopy: null,
        dispositionDelta: "NOT_COMPARABLE_SCHEMA_INVALID",
      };
    }

    const verified = buildVerifiedGate18V10MoatPacket(
      findCompany(manifestCase.company),
      artifactReader(privateRepoRoot),
    );
    const packet =
      manifestCase.validator_mode === "TARGETED_ADYEN_COMPACT"
        ? compactAdyen(verified.packet)
        : verified.packet;

    const rawFields = collectNarrativeFields(structuredClone(parsed.data));
    const rawInventory = rawFields.map((field) => {
      const trimmed = field.value.trim();
      return {
        path: field.path,
        length: trimmed.length,
        terminalPunctuation: terminalPunctuation(trimmed),
        saturationBoundaryHit:
          trimmed.length >= SAFE_NARRATIVE_MAX_EXCLUSIVE,
      };
    });

    const rawMissingPunctuationCount = rawInventory.filter(
      (item) => !item.terminalPunctuation,
    ).length;
    const rawSaturationBoundaryCount = rawInventory.filter(
      (item) => item.saturationBoundaryHit,
    ).length;

    const rawValidatorReplay = validateCase(
      manifestCase,
      packet,
      parsed.data,
    );

    const normalized = structuredClone(parsed.data);
    const normalizedFields = collectNarrativeFields(normalized);
    const normalizedPaths: string[] = [];
    const blockedBySaturationPaths: string[] = [];

    for (const field of normalizedFields) {
      const trimmed = field.value.trim();
      if (terminalPunctuation(trimmed)) {
        continue;
      }
      if (trimmed.length >= SAFE_NARRATIVE_MAX_EXCLUSIVE) {
        blockedBySaturationPaths.push(field.path);
        continue;
      }
      field.set(`${trimmed}.`);
      normalizedPaths.push(field.path);
    }

    const normalizedSchema = gate18PhaseBV10OutputSchema.safeParse(normalized);
    const normalizedSemantic = normalizedSchema.success
      ? validateCase(manifestCase, packet, normalizedSchema.data)
      : { pass: false, error: "NORMALIZED_COPY_SCHEMA_INVALID" };

    let dispositionDelta = "NO_CHANGE";
    if (
      !rawValidatorReplay.pass &&
      normalizedSemantic.pass &&
      blockedBySaturationPaths.length === 0
    ) {
      dispositionDelta =
        "RAW_FAIL_TO_SHADOW_SEMANTIC_PASS_AFTER_PRESENTATION_ONLY_NORMALIZATION";
    } else if (
      !rawValidatorReplay.pass &&
      !normalizedSemantic.pass &&
      rawValidatorReplay.error !== normalizedSemantic.error
    ) {
      dispositionDelta =
        "RAW_PRESENTATION_FAILURE_MASKED_DOWNSTREAM_SUBSTANTIVE_FAILURE";
    } else if (rawValidatorReplay.pass && normalizedSemantic.pass) {
      dispositionDelta = "CONTROL_PASS_STABLE";
    }

    return {
      caseId: manifestCase.case_id,
      model: manifestCase.model,
      company: manifestCase.company,
      validatorMode: manifestCase.validator_mode,
      recordedV10Status: manifestCase.recorded_status,
      recordedV10SemanticError: manifestCase.recorded_semantic_error,
      rawSchemaPass: true,
      rawSchemaError: null,
      rawPresentationCompliance: {
        fieldCount: rawInventory.length,
        missingTerminalPunctuationCount: rawMissingPunctuationCount,
        saturationBoundaryCount: rawSaturationBoundaryCount,
        compliant:
          rawMissingPunctuationCount === 0 &&
          rawSaturationBoundaryCount === 0,
      },
      rawValidatorReplay,
      normalizedCopy: {
        normalizationPolicy:
          "APPEND_PERIOD_ONLY_NO_LEXICAL_CHANGE_BELOW_FROZEN_178_CHAR_COMPLETENESS_BOUNDARY",
        normalizedPathCount: normalizedPaths.length,
        normalizedPaths,
        blockedBySaturationPaths,
        schemaPass: normalizedSchema.success,
        schemaError: normalizedSchema.success
          ? null
          : normalizedSchema.error.message,
        semanticPass: normalizedSemantic.pass,
        semanticError: normalizedSemantic.error,
      },
      dispositionDelta,
    };
  });

  const summary = {
    caseCount: results.length,
    rawPresentationNoncompliantCount: results.filter(
      (item: any) =>
        item.rawPresentationCompliance &&
        item.rawPresentationCompliance.compliant === false,
    ).length,
    rawFailToShadowSemanticPassCount: results.filter(
      (item: any) =>
        item.dispositionDelta ===
        "RAW_FAIL_TO_SHADOW_SEMANTIC_PASS_AFTER_PRESENTATION_ONLY_NORMALIZATION",
    ).length,
    downstreamSubstantiveFailureRevealedCount: results.filter(
      (item: any) =>
        item.dispositionDelta ===
        "RAW_PRESENTATION_FAILURE_MASKED_DOWNSTREAM_SUBSTANTIVE_FAILURE",
    ).length,
    stableControlPassCount: results.filter(
      (item: any) => item.dispositionDelta === "CONTROL_PASS_STABLE",
    ).length,
  };

  console.log(
    JSON.stringify(
      {
        format: "OROTITAN_GATE18_V1_1_SHADOW_REPLAY_RESULT_V0.1",
        status: "SHADOW_REPLAY_COMPLETE",
        mode:
          "EXISTING_PRIVATE_ARTIFACT_REPLAY_NO_INFERENCE_NO_SOURCE_MUTATION",
        architectureHypothesis:
          "A_TWO_LAYER_VALIDATION_WITH_SAFE_NORMALIZATION",
        manifestId: manifestJson.manifest_id,
        safeNarrativeBoundaryExclusive: SAFE_NARRATIVE_MAX_EXCLUSIVE,
        cases: results,
        summary,
        interpretationBoundary: {
          historicalV10ResultsChanged: false,
          retroactivePassAllowed: false,
          sourceArtifactsMutated: false,
          inferenceExecuted: false,
          v11ContractImplemented: false,
          productionMutation: false,
          humanQualityReassessed: false,
          purpose:
            "Measure whether v1.0 candidate dispositions are confounded by presentation-compliance checks before any versioned contract implementation or further model testing.",
        },
      },
      null,
      2,
    ),
  );
}

main();
