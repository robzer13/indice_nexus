import Ajv2020 from "ajv/dist/2020";

import {
  buildModelCostReceipt,
  validateModelPriceSchedule,
  type AnalyticalModelProvider,
  type ModelPriceSchedule,
  type ModelRateLimitSnapshot,
  type ModelTokenUsage,
  type StructuredModelRequest,
  type StructuredModelResponse,
} from "./analytical-model-provider";

type FetchLike = typeof fetch;

export interface AzureProviderConfig {
  endpoint: string;
  deployment: string;
  apiKey: string;
  priceSchedule: ModelPriceSchedule;
  timeoutMs?: number;
}

type AzureResponsesUsage = {
  input_tokens?: unknown;
  output_tokens?: unknown;
  total_tokens?: unknown;
  input_tokens_details?: {
    cached_tokens?: unknown;
  } | null;
};

type AzureResponsesBody = {
  id?: unknown;
  model?: unknown;
  output_text?: unknown;
  output?: unknown;
  usage?: AzureResponsesUsage | null;
};

const AZURE_PROVIDER_ID = "AZURE_OPENAI";
const DEFAULT_TIMEOUT_MS = 60_000;

function normalizeAzureEndpoint(value: string): string {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error("VNEXT_AZURE_ENDPOINT_INVALID");
  }

  if (url.protocol !== "https:") {
    throw new Error("VNEXT_AZURE_ENDPOINT_REQUIRES_HTTPS");
  }

  const hostname = url.hostname.toLowerCase();
  const allowedHost =
    hostname.endsWith(".openai.azure.com") ||
    hostname.endsWith(".services.ai.azure.com");

  if (!allowedHost) {
    throw new Error("VNEXT_AZURE_ENDPOINT_HOST_FORBIDDEN");
  }

  const normalizedPath = url.pathname.replace(/\/+$/, "");
  if (
    normalizedPath !== "" &&
    normalizedPath !== "/openai/v1"
  ) {
    throw new Error("VNEXT_AZURE_ENDPOINT_PATH_INVALID");
  }

  url.pathname = "/openai/v1/";
  url.search = "";
  url.hash = "";
  return url.toString();
}

function assertPositiveInteger(name: string, value: number): void {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`VNEXT_AZURE_REQUEST_INVALID:${name}`);
  }
}

function parseTokenCount(
  name: string,
  value: unknown,
  allowZero = true,
): number {
  if (
    typeof value !== "number" ||
    !Number.isInteger(value) ||
    value < (allowZero ? 0 : 1)
  ) {
    throw new Error(`VNEXT_AZURE_USAGE_INVALID:${name}`);
  }
  return value;
}

function parseNullableIntegerHeader(
  headers: Headers,
  name: string,
): number | null {
  const raw = headers.get(name);
  if (raw === null) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && Number.isInteger(parsed)
    ? parsed
    : null;
}

function readRateLimits(headers: Headers): ModelRateLimitSnapshot {
  return {
    limitRequests: parseNullableIntegerHeader(
      headers,
      "x-ratelimit-limit-requests",
    ),
    limitTokens: parseNullableIntegerHeader(
      headers,
      "x-ratelimit-limit-tokens",
    ),
    remainingRequests: parseNullableIntegerHeader(
      headers,
      "x-ratelimit-remaining-requests",
    ),
    remainingTokens: parseNullableIntegerHeader(
      headers,
      "x-ratelimit-remaining-tokens",
    ),
    resetRequests: headers.get("x-ratelimit-reset-requests"),
    resetTokens: headers.get("x-ratelimit-reset-tokens"),
    retryAfterMs: parseNullableIntegerHeader(headers, "retry-after-ms"),
  };
}

function extractOutputText(body: AzureResponsesBody): string {
  if (
    typeof body.output_text === "string" &&
    body.output_text.length > 0
  ) {
    return body.output_text;
  }

  if (!Array.isArray(body.output)) {
    throw new Error("VNEXT_AZURE_OUTPUT_TEXT_MISSING");
  }

  const texts: string[] = [];

  for (const item of body.output) {
    if (
      typeof item !== "object" ||
      item === null ||
      !("content" in item) ||
      !Array.isArray(item.content)
    ) {
      continue;
    }

    for (const content of item.content) {
      if (
        typeof content === "object" &&
        content !== null &&
        "type" in content &&
        content.type === "output_text" &&
        "text" in content &&
        typeof content.text === "string"
      ) {
        texts.push(content.text);
      }
    }
  }

  if (texts.length === 0) {
    throw new Error("VNEXT_AZURE_OUTPUT_TEXT_MISSING");
  }

  return texts.join("");
}

function parseUsage(body: AzureResponsesBody): ModelTokenUsage {
  if (body.usage === null || typeof body.usage !== "object") {
    throw new Error("VNEXT_AZURE_USAGE_MISSING");
  }

  const inputTokens = parseTokenCount(
    "input_tokens",
    body.usage.input_tokens,
  );
  const outputTokens = parseTokenCount(
    "output_tokens",
    body.usage.output_tokens,
  );
  const cachedInputTokens =
    body.usage.input_tokens_details === null ||
    typeof body.usage.input_tokens_details !== "object" ||
    body.usage.input_tokens_details.cached_tokens === undefined
      ? 0
      : parseTokenCount(
          "cached_tokens",
          body.usage.input_tokens_details.cached_tokens,
        );

  if (cachedInputTokens > inputTokens) {
    throw new Error("VNEXT_AZURE_USAGE_INVALID:cached_gt_input");
  }

  const providerTotal =
    body.usage.total_tokens === undefined
      ? inputTokens + outputTokens
      : parseTokenCount("total_tokens", body.usage.total_tokens);

  if (providerTotal !== inputTokens + outputTokens) {
    throw new Error("VNEXT_AZURE_USAGE_INVALID:total_tokens");
  }

  return {
    inputTokens,
    cachedInputTokens,
    outputTokens,
    totalTokens: providerTotal,
  };
}

function assertStructuredOutput<T>(
  rawOutputText: string,
  schema: Record<string, unknown>,
): T {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawOutputText);
  } catch {
    throw new Error("VNEXT_AZURE_STRUCTURED_OUTPUT_NOT_JSON");
  }

  const ajv = new Ajv2020({
    allErrors: true,
    strict: false,
  });
  const validate = ajv.compile(schema);

  if (!validate(parsed)) {
    const details = (validate.errors ?? [])
      .map((error) => `${error.instancePath || "/"}:${error.keyword}`)
      .join(",");
    throw new Error(
      "VNEXT_AZURE_STRUCTURED_OUTPUT_SCHEMA_FAIL:" + details,
    );
  }

  return parsed as T;
}

function sanitizedHttpFailure(
  status: number,
  statusText: string,
  responseText: string,
): Error {
  const compact = responseText
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 600);

  return new Error(
    `VNEXT_AZURE_HTTP_ERROR:${status}:${statusText}` +
      (compact.length > 0 ? `:${compact}` : ""),
  );
}

export class AzureProvider implements AnalyticalModelProvider {
  readonly providerId = AZURE_PROVIDER_ID;

  private readonly endpoint: string;
  private readonly deployment: string;
  private readonly apiKey: string;
  private readonly priceSchedule: ModelPriceSchedule;
  private readonly timeoutMs: number;
  private readonly fetchImpl: FetchLike;

  constructor(
    config: AzureProviderConfig,
    fetchImpl: FetchLike = fetch,
  ) {
    this.endpoint = normalizeAzureEndpoint(config.endpoint);

    if (config.deployment.trim().length === 0) {
      throw new Error("VNEXT_AZURE_DEPLOYMENT_REQUIRED");
    }
    if (config.apiKey.trim().length === 0) {
      throw new Error("VNEXT_AZURE_API_KEY_REQUIRED");
    }

    validateModelPriceSchedule(config.priceSchedule);

    const timeoutMs = config.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    assertPositiveInteger("timeoutMs", timeoutMs);

    this.deployment = config.deployment;
    this.apiKey = config.apiKey;
    this.priceSchedule = structuredClone(config.priceSchedule);
    this.timeoutMs = timeoutMs;
    this.fetchImpl = fetchImpl;
  }

  async invokeStructured<T>(
    request: StructuredModelRequest,
  ): Promise<StructuredModelResponse<T>> {
    if (request.operationId.trim().length === 0) {
      throw new Error("VNEXT_AZURE_OPERATION_ID_REQUIRED");
    }
    if (request.systemPrompt.trim().length === 0) {
      throw new Error("VNEXT_AZURE_SYSTEM_PROMPT_REQUIRED");
    }
    if (request.input.trim().length === 0) {
      throw new Error("VNEXT_AZURE_INPUT_REQUIRED");
    }
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(request.schemaName)) {
      throw new Error("VNEXT_AZURE_SCHEMA_NAME_INVALID");
    }
    assertPositiveInteger(
      "maxOutputTokens",
      request.maxOutputTokens,
    );

    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(),
      this.timeoutMs,
    );

    try {
      const metadata = {
        ...(request.metadata ?? {}),
        orotitan_operation_id: request.operationId,
      };

      const response = await this.fetchImpl(
        new URL("responses", this.endpoint),
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "api-key": this.apiKey,
          },
          signal: controller.signal,
          body: JSON.stringify({
            model: this.deployment,
            instructions: request.systemPrompt,
            input: request.input,
            max_output_tokens: request.maxOutputTokens,
            store: false,
            metadata,
            text: {
              format: {
                type: "json_schema",
                name: request.schemaName,
                schema: request.schema,
                strict: true,
              },
            },
          }),
        },
      );

      const responseText = await response.text();

      if (!response.ok) {
        throw sanitizedHttpFailure(
          response.status,
          response.statusText,
          responseText,
        );
      }

      let body: AzureResponsesBody;
      try {
        body = JSON.parse(responseText) as AzureResponsesBody;
      } catch {
        throw new Error("VNEXT_AZURE_RESPONSE_NOT_JSON");
      }

      if (typeof body.id !== "string" || body.id.length === 0) {
        throw new Error("VNEXT_AZURE_PROVIDER_REQUEST_ID_MISSING");
      }

      const rawOutputText = extractOutputText(body);
      const output = assertStructuredOutput<T>(
        rawOutputText,
        request.schema,
      );
      const usage = parseUsage(body);

      return {
        providerId: this.providerId,
        deployment: this.deployment,
        providerRequestId: body.id,
        modelReported:
          typeof body.model === "string" ? body.model : null,
        output,
        rawOutputText,
        usage,
        rateLimits: readRateLimits(response.headers),
        cost: buildModelCostReceipt(
          usage,
          this.priceSchedule,
        ),
      };
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name === "AbortError"
      ) {
        throw new Error("VNEXT_AZURE_TIMEOUT");
      }
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }
}

export function createAzureProviderFromEnv(
  env: Readonly<Record<string, string | undefined>> = process.env,
  fetchImpl: FetchLike = fetch,
): AzureProvider {
  const required = (name: string): string => {
    const value = env[name]?.trim();
    if (!value) {
      throw new Error(`VNEXT_AZURE_ENV_MISSING:${name}`);
    }
    return value;
  };

  const numeric = (name: string): number => {
    const raw = required(name);
    const value = Number(raw);
    if (!Number.isFinite(value) || value < 0) {
      throw new Error(`VNEXT_AZURE_ENV_INVALID:${name}`);
    }
    return value;
  };

  return new AzureProvider(
    {
      endpoint: required("AZURE_OPENAI_ENDPOINT"),
      deployment: required("AZURE_OPENAI_DEPLOYMENT"),
      apiKey: required("AZURE_OPENAI_API_KEY"),
      priceSchedule: {
        currency: "USD",
        source: required("AZURE_OPENAI_PRICING_SOURCE"),
        effectiveDate: required(
          "AZURE_OPENAI_PRICING_EFFECTIVE_DATE",
        ),
        inputUsdPerMillionTokens: numeric(
          "AZURE_OPENAI_INPUT_USD_PER_1M",
        ),
        cachedInputUsdPerMillionTokens: numeric(
          "AZURE_OPENAI_CACHED_INPUT_USD_PER_1M",
        ),
        outputUsdPerMillionTokens: numeric(
          "AZURE_OPENAI_OUTPUT_USD_PER_1M",
        ),
      },
    },
    fetchImpl,
  );
}
