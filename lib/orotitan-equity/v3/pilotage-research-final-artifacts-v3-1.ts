/**
 * Exact Research FINAL artifact vocabulary from the pinned
 * OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0 (section 8).
 *
 * Both the D2 immutable-byte verifier and the D1 admission planner must
 * consume the same list; the Registry does not acquire new artifact aliases.
 * A Research CHECKPOINT or an Excel report never satisfies a FINAL handoff.
 */
export const RESEARCH_FINAL_REQUIRED_ARTIFACT_TYPES = [
  "RESEARCH_SOURCE_MANIFEST",
  "EVIDENCE_LEDGER",
  "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER",
  "RESEARCH_GAP_REGISTER",
  "INPUT_SUFFICIENCY_ASSESSMENT",
  "CLASSIFICATION_INPUT_RECORD",
] as const;


/**
 * Schema-only projection for the frozen *V1* stage manifest validator.
 *
 * V1 requires two deprecated Research artifact type names. V2 Research,
 * pinned by this V3.1 run, requires the corresponding genuine V2 outputs.
 * This copy is used ONLY when checking the legacy V1 schema/contract. The
 * unmodified manifest and seven REAL, SHA-verified V2 output registrations
 * are used for lineage, admission, handoff, and persistence.
 *
 * Physical Registry SQL still contains V1 requirements. This bridge does NOT
 * claim that a V2 FINAL can be persisted to production until SQL authority is
 * made version-aware under separately authorized migration.
 */
export function projectV31ResearchFinalForV1Schema(
  input: Record<string, unknown>,
): Record<string, unknown> {
  if (input.manifest_kind !== "FINAL" || input.stage !== "RESEARCH" ||
      input.process_version !== "3.1" || input.pilotage_contract_version !== "3.1")
    return input;
  const contract = input.stage_contract;
  if (contract === null || typeof contract !== "object" || Array.isArray(contract))
    return input;
  const stageContract = contract as Record<string, unknown>;
  if (stageContract.name !== "OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0" ||
      stageContract.version !== "2.0") return input;
  if (!Array.isArray(input.output_artifacts)) return input;
  const outputs = input.output_artifacts as unknown[];
  // No invented artifact rows: every alias is a schema-only projection from
  // an existing, identically-referenced source artifact.
  const projected = outputs.map(item => {
    if (item === null || typeof item !== "object" || Array.isArray(item))
      return item;
    const ref = item as Record<string, unknown>;
    if (ref.artifact_type === "INPUT_SUFFICIENCY_ASSESSMENT")
      return {...ref, artifact_type: "DD_INPUT_SUFFICIENCY_RECORD"};
    if (ref.artifact_type === "CLASSIFICATION_INPUT_RECORD")
      return {...ref, artifact_type: "ANALYSIS_INPUT_LOCK"};
    return item;
  });
  return {...input, output_artifacts: projected};
}
