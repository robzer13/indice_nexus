export const WEAK_LINK_TRI_STATES = ["YES", "NO", "UNKNOWN"] as const;

export type WeakLinkTriState = (typeof WEAK_LINK_TRI_STATES)[number];

export type ExecutionConfidence = "HIGH" | "MEDIUM" | "LOW";

export interface WeakLinkCriterionAssessment {
  state: WeakLinkTriState;
  rationale: string;
  evidenceIds: readonly string[];
  contradictingEvidenceIds: readonly string[];
  assumptionIds: readonly string[];
}

export interface WeakLinkTaxonomyInput {
  weakLinkId: string;
  title: string;
  sourceBlockIds: readonly string[];
  materiality: WeakLinkCriterionAssessment;
  causality: WeakLinkCriterionAssessment;
  unresolvedness: WeakLinkCriterionAssessment;
  executionConfidence: ExecutionConfidence;
  overallRationale: string;
}

export type WeakLinkTerminalSignal =
  | "OROTITAN_STATUS_NO_REQUIRED"
  | "NO_TERMINAL_SIGNAL"
  | "UNRESOLVED";

export interface WeakLinkTaxonomyOutput {
  weakLinkId: string;
  title: string;
  sourceBlockIds: readonly string[];
  materiality: WeakLinkCriterionAssessment;
  causality: WeakLinkCriterionAssessment;
  unresolvedness: WeakLinkCriterionAssessment;
  materialWeakLink: WeakLinkTriState;
  terminalSignal: WeakLinkTerminalSignal;
  executionConfidence: ExecutionConfidence;
  supportingEvidenceIds: readonly string[];
  contradictingEvidenceIds: readonly string[];
  assumptionIds: readonly string[];
  overallRationale: string;
}

function assertNonBlank(value: string, code: string): void {
  if (value.trim().length === 0) throw new Error(code);
}

function uniqueSorted(values: readonly string[]): string[] {
  return [...new Set(values)].sort();
}

function assertAssessment(
  label: string,
  assessment: WeakLinkCriterionAssessment,
): void {
  if (!WEAK_LINK_TRI_STATES.includes(assessment.state)) {
    throw new Error(`VNEXT_WEAK_LINK_INVALID_${label}_STATE`);
  }

  assertNonBlank(
    assessment.rationale,
    `VNEXT_WEAK_LINK_${label}_RATIONALE_REQUIRED`,
  );

  const supporting = new Set(assessment.evidenceIds);
  for (const id of assessment.contradictingEvidenceIds) {
    if (supporting.has(id)) {
      throw new Error(
        `VNEXT_WEAK_LINK_${label}_EVIDENCE_CONTRADICTION_OVERLAP`,
      );
    }
  }

  if (
    assessment.state === "YES" &&
    assessment.evidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_WEAK_LINK_${label}_YES_REQUIRES_EVIDENCE`,
    );
  }

  if (
    assessment.state === "UNKNOWN" &&
    assessment.evidenceIds.length === 0 &&
    assessment.contradictingEvidenceIds.length === 0
  ) {
    throw new Error(
      `VNEXT_WEAK_LINK_${label}_UNKNOWN_REQUIRES_TRACEABLE_BASIS`,
    );
  }
}

/**
 * Frozen rule:
 * MATERIAL_WEAK_LINK = YES only when MATERIALITY + CAUSALITY +
 * UNRESOLVEDNESS all hold.
 *
 * The three criterion assessments are analytical judgments.
 * This function only applies the frozen conjunction deterministically.
 */
export function deriveMaterialWeakLink(
  materiality: WeakLinkTriState,
  causality: WeakLinkTriState,
  unresolvedness: WeakLinkTriState,
): WeakLinkTriState {
  const states = [materiality, causality, unresolvedness];

  if (states.includes("NO")) return "NO";
  if (states.every((state) => state === "YES")) return "YES";
  return "UNKNOWN";
}

export function evaluateWeakLinkTaxonomy(
  input: WeakLinkTaxonomyInput,
): WeakLinkTaxonomyOutput {
  assertNonBlank(input.weakLinkId, "VNEXT_WEAK_LINK_ID_REQUIRED");
  assertNonBlank(input.title, "VNEXT_WEAK_LINK_TITLE_REQUIRED");
  assertNonBlank(
    input.overallRationale,
    "VNEXT_WEAK_LINK_OVERALL_RATIONALE_REQUIRED",
  );

  if (input.sourceBlockIds.length === 0) {
    throw new Error("VNEXT_WEAK_LINK_SOURCE_BLOCK_REQUIRED");
  }

  assertAssessment("MATERIALITY", input.materiality);
  assertAssessment("CAUSALITY", input.causality);
  assertAssessment("UNRESOLVEDNESS", input.unresolvedness);

  const materialWeakLink = deriveMaterialWeakLink(
    input.materiality.state,
    input.causality.state,
    input.unresolvedness.state,
  );

  const terminalSignal: WeakLinkTerminalSignal =
    materialWeakLink === "YES"
      ? "OROTITAN_STATUS_NO_REQUIRED"
      : materialWeakLink === "UNKNOWN"
        ? "UNRESOLVED"
        : "NO_TERMINAL_SIGNAL";

  return {
    weakLinkId: input.weakLinkId,
    title: input.title,
    sourceBlockIds: uniqueSorted(input.sourceBlockIds),
    materiality: input.materiality,
    causality: input.causality,
    unresolvedness: input.unresolvedness,
    materialWeakLink,
    terminalSignal,
    executionConfidence: input.executionConfidence,
    supportingEvidenceIds: uniqueSorted([
      ...input.materiality.evidenceIds,
      ...input.causality.evidenceIds,
      ...input.unresolvedness.evidenceIds,
    ]),
    contradictingEvidenceIds: uniqueSorted([
      ...input.materiality.contradictingEvidenceIds,
      ...input.causality.contradictingEvidenceIds,
      ...input.unresolvedness.contradictingEvidenceIds,
    ]),
    assumptionIds: uniqueSorted([
      ...input.materiality.assumptionIds,
      ...input.causality.assumptionIds,
      ...input.unresolvedness.assumptionIds,
    ]),
    overallRationale: input.overallRationale,
  };
}
