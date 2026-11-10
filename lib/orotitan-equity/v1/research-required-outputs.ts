/**
 * Contract-specific Research FINAL output identities.
 *
 * Only the exact frozen Research V2 contract uses the two V2 logical names.
 * Older contracts remain strict and are never silently relabeled or aliased.
 * PRIVATE_GITHUB storage, evidence hashes and stage authority are checked elsewhere.
 */
export const LEGACY_RESEARCH_FINAL_OUTPUTS = [
  "RESEARCH_SOURCE_MANIFEST",
  "EVIDENCE_LEDGER",
  "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER",
  "RESEARCH_GAP_REGISTER",
  "DD_INPUT_SUFFICIENCY_RECORD",
  "ANALYSIS_INPUT_LOCK",
] as const;

export const RESEARCH_V2_FINAL_OUTPUTS = [
  "RESEARCH_SOURCE_MANIFEST",
  "EVIDENCE_LEDGER",
  "CONFLICT_LEDGER",
  "MATERIAL_RESEARCH_HYPOTHESIS_REGISTER",
  "RESEARCH_GAP_REGISTER",
  "INPUT_SUFFICIENCY_ASSESSMENT",
  "CLASSIFICATION_INPUT_RECORD",
] as const;

export function requiredResearchFinalOutputs(
  stageContractName: string,
  stageContractVersion: string,
): readonly string[] {
  return stageContractName === "OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0"
    && stageContractVersion === "2.0"
    ? RESEARCH_V2_FINAL_OUTPUTS
    : LEGACY_RESEARCH_FINAL_OUTPUTS;
}
