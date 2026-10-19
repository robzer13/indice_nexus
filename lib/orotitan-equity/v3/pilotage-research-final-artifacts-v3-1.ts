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
