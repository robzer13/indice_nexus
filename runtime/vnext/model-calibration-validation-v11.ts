import { createHash } from "node:crypto";

import type { Gate18V02EvidencePacket } from "./model-calibration-pilot-v02";
import {
  assertGate18PhaseBV10Semantics,
  gate18PhaseBV10OutputSchema,
  type Gate18PhaseBV10Output,
} from "./model-calibration-pilot-v10";
import {
  assertGate18V10TargetedProbeSemantics,
} from "./model-calibration-targeted-regression-v10";
import {
  assertGate18V10BrookfieldTargetedProbeSemantics,
} from "./model-calibration-targeted-brookfield-v10";

export const GATE18_V11_VALIDATION_CONTRACT_ID =
  "GATE18_MOAT_EVIDENCE_AUDIT_VALIDATION_V1_1" as const;

export const GATE18_V11_VALIDATION_CONTRACT_VERSION =
  "1.1" as const;

export const GATE18_V11_SAFE_NARRATIVE_BOUNDARY_EXCLUSIVE =
  178 as const;

export type Gate18V11SemanticMode =
  | "FULL"
  | "TARGETED_ADYEN"
  | "TARGETED_BROOKFIELD";

export type Gate18V11PresentationIssueKind =
  | "MISSING_TERMINAL_PUNCTUATION"
  | "NARRATIVE_BOUNDARY_SATURATION";

export interface Gate18V11PresentationIssue {
  path: string;
  kind: Gate18V11PresentationIssueKind;
  trimmedLength: number;
  safelyNormalizable: boolean;
  blockedReason: string | null;
}

export interface Gate18V11PresentationReport {
  fieldCount: number;
  issueCount: number;
  missingTerminalPunctuationCount: number;
  saturationBoundaryCount: number;
  compliant: boolean;
  issues: Gate18V11PresentationIssue[];
}

export type Gate18V11SubstantiveStatus =
  | "PASS"
  | "FAIL"
  | "NOT_EVALUATED_SCHEMA_FAILURE"
  | "NOT_EVALUATED_PRESENTATION_BLOCKER";

export interface Gate18V11ValidationResult {
  contractId: typeof GATE18_V11_VALIDATION_CONTRACT_ID;
  contractVersion: typeof GATE18_V11_VALIDATION_CONTRACT_VERSION;
  semanticMode: Gate18V11SemanticMode;
  rawSchemaPass: boolean;
  rawSchemaError: string | null;
  rawPresentationCompliance: Gate18V11PresentationReport | null;
  normalization: {
    policy:
      "APPEND_PERIOD_ONLY_PRESERVE_ALL_EXISTING_CHARACTERS_BELOW_178_BOUNDARY";
    normalizedPathCount: number;
    normalizedPaths: string[];
    blockedPathCount: number;
    blockedPaths: string[];
    rawOutputMutated: false;
  };
  normalizedSchemaPass: boolean | null;
  normalizedSchemaError: string | null;
  normalizedPresentationCompliance: Gate18V11PresentationReport | null;
  substantiveValidation: {
    status: Gate18V11SubstantiveStatus;
    pass: boolean | null;
    error: string | null;
  };
  humanQualityReassessed: false;
  historicalV10ResultChanged: false;
}

export const GATE18_V11_VALIDATION_CONTRACT_SPEC = {
  contract_id: GATE18_V11_VALIDATION_CONTRACT_ID,
  contract_version: GATE18_V11_VALIDATION_CONTRACT_VERSION,
  generation_contract: "UNCHANGED_V1_0",
  layers: [
    "RAW_OUTPUT_SCHEMA_VALIDATION",
    "RAW_PRESENTATION_COMPLIANCE_VALIDATION",
    "STRICT_SAFE_PRESENTATION_NORMALIZER",
    "SUBSTANTIVE_DETERMINISTIC_SEMANTIC_VALIDATION_ON_NORMALIZED_COPY",
    "HUMAN_QUALITY_ADJUDICATION_SEPARATE",
  ],
  presentation: {
    terminal_punctuation: "[.!?]",
    safe_narrative_boundary_exclusive:
      GATE18_V11_SAFE_NARRATIVE_BOUNDARY_EXCLUSIVE,
  },
  normalization: {
    operation: "APPEND_PERIOD_ONLY",
    lexical_change_allowed: false,
    removal_allowed: false,
    replacement_allowed: false,
    reorder_allowed: false,
    raw_output_mutation_allowed: false,
    only_when_normalized_trimmed_length_remains_below_boundary: true,
  },
  semantic_evaluation: {
    reuses_frozen_v1_0_semantic_validators: true,
    only_after_normalized_copy_is_presentation_compliant: true,
    presentation_blocker_is_not_semantic_failure: true,
  },
  history: {
    historical_v1_0_results_immutable: true,
    retroactive_pass_allowed: false,
  },
} as const;

interface NarrativeField {
  path: string;
  get: () => string;
  set: (value: string) => void;
}

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export function gate18V11ValidationContractSha256(): string {
  return sha256Hex(JSON.stringify(GATE18_V11_VALIDATION_CONTRACT_SPEC));
}

function narrativeFields(
  output: Gate18PhaseBV10Output,
): NarrativeField[] {
  const fields: NarrativeField[] = [];

  output.priority_findings.forEach((finding, findingIndex) => {
    fields.push({
      path: `priority_findings[${findingIndex}].claim`,
      get: () => finding.claim,
      set: (value) => {
        finding.claim = value;
      },
    });
    fields.push({
      path: `priority_findings[${findingIndex}].causal_link`,
      get: () => finding.causal_link,
      set: (value) => {
        finding.causal_link = value;
      },
    });

    finding.evidence_qualifications.forEach(
      (qualification, qualificationIndex) => {
        fields.push({
          path:
            `priority_findings[${findingIndex}].evidence_qualifications[${qualificationIndex}].qualification`,
          get: () => qualification.qualification,
          set: (value) => {
            qualification.qualification = value;
          },
        });
      },
    );

    if (finding.counterevidence_link !== null) {
      fields.push({
        path: `priority_findings[${findingIndex}].counterevidence_link`,
        get: () => finding.counterevidence_link ?? "",
        set: (value) => {
          finding.counterevidence_link = value;
        },
      });
    }
  });

  output.material_conflicts.forEach((conflict, conflictIndex) => {
    fields.push({
      path: `material_conflicts[${conflictIndex}].implication`,
      get: () => conflict.implication,
      set: (value) => {
        conflict.implication = value;
      },
    });
  });

  output.weak_link_candidates.forEach((candidate, candidateIndex) => {
    fields.push({
      path: `weak_link_candidates[${candidateIndex}].why_uncertain`,
      get: () => candidate.why_uncertain,
      set: (value) => {
        candidate.why_uncertain = value;
      },
    });
  });

  output.unresolved_points.forEach((point, pointIndex) => {
    fields.push({
      path: `unresolved_points[${pointIndex}].question`,
      get: () => point.question,
      set: (value) => {
        point.question = value;
      },
    });
  });

  return fields;
}

function cloneOutput(
  output: Gate18PhaseBV10Output,
): Gate18PhaseBV10Output {
  return {
    ...output,
    priority_findings: output.priority_findings.map((finding) => ({
      ...finding,
      evidence_ids: [...finding.evidence_ids],
      conflict_ids: [...finding.conflict_ids],
      evidence_qualifications:
        finding.evidence_qualifications.map((item) => ({
          ...item,
        })),
      counterevidence_ids: [...finding.counterevidence_ids],
    })),
    material_conflicts: output.material_conflicts.map((item) => ({
      ...item,
    })),
    weak_link_candidates: output.weak_link_candidates.map((item) => ({
      ...item,
      evidence_ids: [...item.evidence_ids],
      conflict_ids: [...item.conflict_ids],
    })),
    unresolved_points: output.unresolved_points.map((item) => ({
      ...item,
      evidence_ids: [...item.evidence_ids],
      conflict_ids: [...item.conflict_ids],
    })),
  };
}

function appendPeriodPreservingCharacters(value: string): string {
  const trailingWhitespace = value.match(/\s*$/)?.[0] ?? "";
  const insertionIndex = value.length - trailingWhitespace.length;

  return [
    value.slice(0, insertionIndex),
    ".",
    trailingWhitespace,
  ].join("");
}

export function inspectGate18V11Presentation(
  output: Gate18PhaseBV10Output,
): Gate18V11PresentationReport {
  const fields = narrativeFields(output);
  const issues: Gate18V11PresentationIssue[] = [];

  for (const field of fields) {
    const value = field.get();
    const trimmed = value.trim();
    const trimmedLength = trimmed.length;
    const missingTerminalPunctuation = !/[.!?]$/.test(trimmed);
    const saturated =
      trimmedLength >= GATE18_V11_SAFE_NARRATIVE_BOUNDARY_EXCLUSIVE;

    if (missingTerminalPunctuation) {
      const normalizedLength = trimmedLength + 1;
      const safelyNormalizable =
        trimmedLength > 0 &&
        normalizedLength <
          GATE18_V11_SAFE_NARRATIVE_BOUNDARY_EXCLUSIVE;

      issues.push({
        path: field.path,
        kind: "MISSING_TERMINAL_PUNCTUATION",
        trimmedLength,
        safelyNormalizable,
        blockedReason: safelyNormalizable
          ? null
          : "NORMALIZATION_WOULD_REACH_OR_EXCEED_178_BOUNDARY",
      });
    }

    if (saturated) {
      issues.push({
        path: field.path,
        kind: "NARRATIVE_BOUNDARY_SATURATION",
        trimmedLength,
        safelyNormalizable: false,
        blockedReason: "RAW_FIELD_AT_OR_ABOVE_178_BOUNDARY",
      });
    }
  }

  return {
    fieldCount: fields.length,
    issueCount: issues.length,
    missingTerminalPunctuationCount: issues.filter(
      (issue) =>
        issue.kind === "MISSING_TERMINAL_PUNCTUATION",
    ).length,
    saturationBoundaryCount: issues.filter(
      (issue) =>
        issue.kind === "NARRATIVE_BOUNDARY_SATURATION",
    ).length,
    compliant: issues.length === 0,
    issues,
  };
}

export function normalizeGate18V11Presentation(
  output: Gate18PhaseBV10Output,
): {
  normalized: Gate18PhaseBV10Output;
  normalizedPaths: string[];
  blockedPaths: string[];
} {
  const normalized = cloneOutput(output);
  const report = inspectGate18V11Presentation(normalized);
  const normalizablePaths = new Set(
    report.issues
      .filter(
        (issue) =>
          issue.kind === "MISSING_TERMINAL_PUNCTUATION" &&
          issue.safelyNormalizable,
      )
      .map((issue) => issue.path),
  );
  const blockedPaths = [
    ...new Set(
      report.issues
        .filter((issue) => !issue.safelyNormalizable)
        .map((issue) => issue.path),
    ),
  ];

  const normalizedPaths: string[] = [];
  for (const field of narrativeFields(normalized)) {
    if (!normalizablePaths.has(field.path)) {
      continue;
    }

    field.set(appendPeriodPreservingCharacters(field.get()));
    normalizedPaths.push(field.path);
  }

  return {
    normalized,
    normalizedPaths,
    blockedPaths,
  };
}

function semanticValidator(
  mode: Gate18V11SemanticMode,
): (
  packet: Gate18V02EvidencePacket,
  output: Gate18PhaseBV10Output,
) => void {
  switch (mode) {
    case "FULL":
      return assertGate18PhaseBV10Semantics;
    case "TARGETED_ADYEN":
      return assertGate18V10TargetedProbeSemantics;
    case "TARGETED_BROOKFIELD":
      return assertGate18V10BrookfieldTargetedProbeSemantics;
  }
}

function safeError(error: unknown): string {
  return error instanceof Error ? error.message : "UNKNOWN_ERROR";
}

export function evaluateGate18V11Validation(
  packet: Gate18V02EvidencePacket,
  rawOutput: unknown,
  mode: Gate18V11SemanticMode = "FULL",
): Gate18V11ValidationResult {
  const parsed = gate18PhaseBV10OutputSchema.safeParse(rawOutput);

  if (!parsed.success) {
    return {
      contractId: GATE18_V11_VALIDATION_CONTRACT_ID,
      contractVersion: GATE18_V11_VALIDATION_CONTRACT_VERSION,
      semanticMode: mode,
      rawSchemaPass: false,
      rawSchemaError: "VNEXT_GATE18_V11_RAW_SCHEMA_INVALID",
      rawPresentationCompliance: null,
      normalization: {
        policy:
          "APPEND_PERIOD_ONLY_PRESERVE_ALL_EXISTING_CHARACTERS_BELOW_178_BOUNDARY",
        normalizedPathCount: 0,
        normalizedPaths: [],
        blockedPathCount: 0,
        blockedPaths: [],
        rawOutputMutated: false,
      },
      normalizedSchemaPass: null,
      normalizedSchemaError: null,
      normalizedPresentationCompliance: null,
      substantiveValidation: {
        status: "NOT_EVALUATED_SCHEMA_FAILURE",
        pass: null,
        error: null,
      },
      humanQualityReassessed: false,
      historicalV10ResultChanged: false,
    };
  }

  const raw = parsed.data;
  const rawPresentationCompliance =
    inspectGate18V11Presentation(raw);
  const normalization = normalizeGate18V11Presentation(raw);
  const normalizedParse =
    gate18PhaseBV10OutputSchema.safeParse(normalization.normalized);

  if (!normalizedParse.success) {
    return {
      contractId: GATE18_V11_VALIDATION_CONTRACT_ID,
      contractVersion: GATE18_V11_VALIDATION_CONTRACT_VERSION,
      semanticMode: mode,
      rawSchemaPass: true,
      rawSchemaError: null,
      rawPresentationCompliance,
      normalization: {
        policy:
          "APPEND_PERIOD_ONLY_PRESERVE_ALL_EXISTING_CHARACTERS_BELOW_178_BOUNDARY",
        normalizedPathCount: normalization.normalizedPaths.length,
        normalizedPaths: normalization.normalizedPaths,
        blockedPathCount: normalization.blockedPaths.length,
        blockedPaths: normalization.blockedPaths,
        rawOutputMutated: false,
      },
      normalizedSchemaPass: false,
      normalizedSchemaError:
        "VNEXT_GATE18_V11_NORMALIZED_SCHEMA_INVALID",
      normalizedPresentationCompliance: null,
      substantiveValidation: {
        status: "NOT_EVALUATED_SCHEMA_FAILURE",
        pass: null,
        error: null,
      },
      humanQualityReassessed: false,
      historicalV10ResultChanged: false,
    };
  }

  const normalized = normalizedParse.data;
  const normalizedPresentationCompliance =
    inspectGate18V11Presentation(normalized);

  if (!normalizedPresentationCompliance.compliant) {
    return {
      contractId: GATE18_V11_VALIDATION_CONTRACT_ID,
      contractVersion: GATE18_V11_VALIDATION_CONTRACT_VERSION,
      semanticMode: mode,
      rawSchemaPass: true,
      rawSchemaError: null,
      rawPresentationCompliance,
      normalization: {
        policy:
          "APPEND_PERIOD_ONLY_PRESERVE_ALL_EXISTING_CHARACTERS_BELOW_178_BOUNDARY",
        normalizedPathCount: normalization.normalizedPaths.length,
        normalizedPaths: normalization.normalizedPaths,
        blockedPathCount: normalization.blockedPaths.length,
        blockedPaths: normalization.blockedPaths,
        rawOutputMutated: false,
      },
      normalizedSchemaPass: true,
      normalizedSchemaError: null,
      normalizedPresentationCompliance,
      substantiveValidation: {
        status: "NOT_EVALUATED_PRESENTATION_BLOCKER",
        pass: null,
        error: null,
      },
      humanQualityReassessed: false,
      historicalV10ResultChanged: false,
    };
  }

  try {
    semanticValidator(mode)(packet, normalized);
  } catch (error) {
    const message = safeError(error);

    if (
      message.endsWith("_INCOMPLETE") ||
      message === "VNEXT_GATE18_V10_NARRATIVE_BOUNDARY_SATURATION"
    ) {
      throw new Error(
        `VNEXT_GATE18_V11_PRESENTATION_LAYER_LEAK:${message}`,
      );
    }

    return {
      contractId: GATE18_V11_VALIDATION_CONTRACT_ID,
      contractVersion: GATE18_V11_VALIDATION_CONTRACT_VERSION,
      semanticMode: mode,
      rawSchemaPass: true,
      rawSchemaError: null,
      rawPresentationCompliance,
      normalization: {
        policy:
          "APPEND_PERIOD_ONLY_PRESERVE_ALL_EXISTING_CHARACTERS_BELOW_178_BOUNDARY",
        normalizedPathCount: normalization.normalizedPaths.length,
        normalizedPaths: normalization.normalizedPaths,
        blockedPathCount: normalization.blockedPaths.length,
        blockedPaths: normalization.blockedPaths,
        rawOutputMutated: false,
      },
      normalizedSchemaPass: true,
      normalizedSchemaError: null,
      normalizedPresentationCompliance,
      substantiveValidation: {
        status: "FAIL",
        pass: false,
        error: message,
      },
      humanQualityReassessed: false,
      historicalV10ResultChanged: false,
    };
  }

  return {
    contractId: GATE18_V11_VALIDATION_CONTRACT_ID,
    contractVersion: GATE18_V11_VALIDATION_CONTRACT_VERSION,
    semanticMode: mode,
    rawSchemaPass: true,
    rawSchemaError: null,
    rawPresentationCompliance,
    normalization: {
      policy:
        "APPEND_PERIOD_ONLY_PRESERVE_ALL_EXISTING_CHARACTERS_BELOW_178_BOUNDARY",
      normalizedPathCount: normalization.normalizedPaths.length,
      normalizedPaths: normalization.normalizedPaths,
      blockedPathCount: normalization.blockedPaths.length,
      blockedPaths: normalization.blockedPaths,
      rawOutputMutated: false,
    },
    normalizedSchemaPass: true,
    normalizedSchemaError: null,
    normalizedPresentationCompliance,
    substantiveValidation: {
      status: "PASS",
      pass: true,
      error: null,
    },
    humanQualityReassessed: false,
    historicalV10ResultChanged: false,
  };
}
