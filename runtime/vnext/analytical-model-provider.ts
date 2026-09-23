export type JsonSchema = Record<string, unknown>;

export interface StructuredModelRequest {
  operationId: string;
  systemPrompt: string;
  input: string;
  schemaName: string;
  schema: JsonSchema;
  maxOutputTokens: number;
  metadata?: Readonly<Record<string, string>>;
}

export interface ModelTokenUsage {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  totalTokens: number;
}

export interface ModelRateLimitSnapshot {
  limitRequests: number | null;
  limitTokens: number | null;
  remainingRequests: number | null;
  remainingTokens: number | null;
  resetRequests: string | null;
  resetTokens: string | null;
  retryAfterMs: number | null;
}

export interface ModelPriceSchedule {
  currency: "USD";
  source: string;
  effectiveDate: string;
  inputUsdPerMillionTokens: number;
  cachedInputUsdPerMillionTokens: number;
  outputUsdPerMillionTokens: number;
}

export interface ModelCostReceipt {
  currency: "USD";
  pricingSource: string;
  pricingEffectiveDate: string;
  inputTokens: number;
  cachedInputTokens: number;
  uncachedInputTokens: number;
  outputTokens: number;
  inputUsdPerMillionTokens: number;
  cachedInputUsdPerMillionTokens: number;
  outputUsdPerMillionTokens: number;
  uncachedInputCostUsd: number;
  cachedInputCostUsd: number;
  outputCostUsd: number;
  totalEstimatedCostUsd: number;
  usageSource: "PROVIDER_RESPONSE";
  pricingMode: "PINNED_RETAIL_ESTIMATE";
}

export interface StructuredModelResponse<T> {
  providerId: string;
  deployment: string;
  providerRequestId: string;
  modelReported: string | null;
  output: T;
  rawOutputText: string;
  usage: ModelTokenUsage;
  rateLimits: ModelRateLimitSnapshot;
  cost: ModelCostReceipt;
}

export interface AnalyticalModelProvider {
  readonly providerId: string;

  invokeStructured<T>(
    request: StructuredModelRequest,
  ): Promise<StructuredModelResponse<T>>;
}

function assertNonNegativeFinite(name: string, value: number): void {
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`VNEXT_MODEL_PRICE_INVALID:${name}`);
  }
}

export function validateModelPriceSchedule(
  schedule: ModelPriceSchedule,
): void {
  if (schedule.currency !== "USD") {
    throw new Error("VNEXT_MODEL_PRICE_CURRENCY_UNSUPPORTED");
  }
  if (schedule.source.trim().length === 0) {
    throw new Error("VNEXT_MODEL_PRICE_SOURCE_REQUIRED");
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(schedule.effectiveDate)) {
    throw new Error("VNEXT_MODEL_PRICE_EFFECTIVE_DATE_INVALID");
  }

  assertNonNegativeFinite(
    "inputUsdPerMillionTokens",
    schedule.inputUsdPerMillionTokens,
  );
  assertNonNegativeFinite(
    "cachedInputUsdPerMillionTokens",
    schedule.cachedInputUsdPerMillionTokens,
  );
  assertNonNegativeFinite(
    "outputUsdPerMillionTokens",
    schedule.outputUsdPerMillionTokens,
  );
}

function roundUsd(value: number): number {
  return Number(value.toFixed(12));
}

export function buildModelCostReceipt(
  usage: ModelTokenUsage,
  schedule: ModelPriceSchedule,
): ModelCostReceipt {
  validateModelPriceSchedule(schedule);

  const uncachedInputTokens = Math.max(
    0,
    usage.inputTokens - usage.cachedInputTokens,
  );

  const uncachedInputCostUsd =
    (uncachedInputTokens / 1_000_000) *
    schedule.inputUsdPerMillionTokens;
  const cachedInputCostUsd =
    (usage.cachedInputTokens / 1_000_000) *
    schedule.cachedInputUsdPerMillionTokens;
  const outputCostUsd =
    (usage.outputTokens / 1_000_000) *
    schedule.outputUsdPerMillionTokens;

  return {
    currency: "USD",
    pricingSource: schedule.source,
    pricingEffectiveDate: schedule.effectiveDate,
    inputTokens: usage.inputTokens,
    cachedInputTokens: usage.cachedInputTokens,
    uncachedInputTokens,
    outputTokens: usage.outputTokens,
    inputUsdPerMillionTokens: schedule.inputUsdPerMillionTokens,
    cachedInputUsdPerMillionTokens:
      schedule.cachedInputUsdPerMillionTokens,
    outputUsdPerMillionTokens: schedule.outputUsdPerMillionTokens,
    uncachedInputCostUsd: roundUsd(uncachedInputCostUsd),
    cachedInputCostUsd: roundUsd(cachedInputCostUsd),
    outputCostUsd: roundUsd(outputCostUsd),
    totalEstimatedCostUsd: roundUsd(
      uncachedInputCostUsd + cachedInputCostUsd + outputCostUsd,
    ),
    usageSource: "PROVIDER_RESPONSE",
    pricingMode: "PINNED_RETAIL_ESTIMATE",
  };
}
