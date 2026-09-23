import {
  GATE18_MODEL_CANDIDATES,
} from "./model-calibration";

/**
 * Gate 18 Phase B company-level execution profiles.
 *
 * Physical model identity remains unchanged from the Gate 18 candidate set.
 * Reasoning effort is a model-specific request field explicitly permitted by
 * the same-request invariant.
 *
 * SOL/high exhausted the complete output budget twice (1536 and 4096 tokens)
 * with zero visible output on the bounded company-level MOAT audit. Phase B
 * therefore calibrates SOL at medium reasoning before any further SOL/high
 * spend. This is an execution-profile correction, not a model substitution or
 * routing promotion.
 */
export const GATE18_PHASE_B_MODEL_CANDIDATES = [
  {
    ...GATE18_MODEL_CANDIDATES[0],
    reasoning: "minimal",
  },
  {
    ...GATE18_MODEL_CANDIDATES[1],
    reasoning: "medium",
  },
  {
    ...GATE18_MODEL_CANDIDATES[2],
    reasoning: "medium",
  },
  {
    ...GATE18_MODEL_CANDIDATES[3],
    reasoning: "high",
  },
] as const;

export type Gate18PhaseBModelCandidate =
  (typeof GATE18_PHASE_B_MODEL_CANDIDATES)[number];
