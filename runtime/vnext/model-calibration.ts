export const GATE18_MODEL_CANDIDATES = [
  {
    label: "LUNA",
    modelId: "openai/gpt-5.6-luna",
    intendedTier: "T1_STRUCTURED_EXTRACTION",
    reasoning: "minimal",
  },
  {
    label: "TERRA",
    modelId: "openai/gpt-5.6-terra",
    intendedTier: "T2_STANDARD_ANALYSIS",
    reasoning: "medium",
  },
  {
    label: "SOL",
    modelId: "openai/gpt-5.6-sol",
    intendedTier: "T3_PREMIUM_REASONING",
    reasoning: "high",
  },
  {
    label: "ASTRA",
    modelId: "openai/gpt-6-astra",
    intendedTier: "T4_FRONTIER_ESCALATION",
    reasoning: "high",
  },
] as const;

export type Gate18ModelCandidate =
  (typeof GATE18_MODEL_CANDIDATES)[number];

export interface Gate18SmokeReceipt {
  label: Gate18ModelCandidate["label"];
  modelId: string;
  intendedTier: Gate18ModelCandidate["intendedTier"];
  requestedReasoning: Gate18ModelCandidate["reasoning"];
  schemaValid: boolean;
  latencyMs: number;
  inputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  totalTokens: number | null;
  finishReason: string | null;
  providerMetadata: unknown;
}

export function assertGate18SmokeReceipt(
  receipt: Gate18SmokeReceipt,
): void {
  if (!GATE18_MODEL_CANDIDATES.some(
    (candidate) =>
      candidate.label === receipt.label &&
      candidate.modelId === receipt.modelId,
  )) {
    throw new Error("VNEXT_GATE18_UNKNOWN_MODEL_CANDIDATE");
  }

  if (!receipt.schemaValid) {
    throw new Error("VNEXT_GATE18_SCHEMA_INVALID");
  }

  if (!Number.isFinite(receipt.latencyMs) || receipt.latencyMs < 0) {
    throw new Error("VNEXT_GATE18_LATENCY_INVALID");
  }

  for (const tokenCount of [
    receipt.inputTokens,
    receipt.outputTokens,
    receipt.reasoningTokens,
    receipt.totalTokens,
  ]) {
    if (
      tokenCount !== null &&
      (!Number.isInteger(tokenCount) || tokenCount < 0)
    ) {
      throw new Error("VNEXT_GATE18_TOKEN_USAGE_INVALID");
    }
  }
}
