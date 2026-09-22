import assert from "node:assert/strict";
import test from "node:test";

import {
  buildModelCostReceipt,
  type AnalyticalModelProvider,
  type StructuredModelRequest,
} from "../runtime/vnext/analytical-model-provider";
import {
  AzureProvider,
  createAzureProviderFromEnv,
} from "../runtime/vnext/azure-provider";

const schema = {
  type: "object",
  additionalProperties: false,
  properties: {
    answer: { type: "string" },
    confidence: {
      enum: ["HIGH", "MEDIUM", "LOW"],
    },
  },
  required: ["answer", "confidence"],
} as const;

function request(): StructuredModelRequest {
  return {
    operationId: "gate13-smoke-001",
    systemPrompt:
      "Return only the requested analytical JSON object.",
    input: "Classify this synthetic fixture.",
    schemaName: "Gate13Smoke",
    schema,
    maxOutputTokens: 200,
    metadata: {
      run_mode: "VNEXT_SHADOW",
    },
  };
}

function config() {
  return {
    endpoint: "https://orotitan-student.openai.azure.com/",
    deployment: "orotitan-gate13",
    apiKey: "synthetic-test-key",
    priceSchedule: {
      currency: "USD" as const,
      source: "fixture://azure-retail-price",
      effectiveDate: "2026-09-22",
      inputUsdPerMillionTokens: 2,
      cachedInputUsdPerMillionTokens: 0.5,
      outputUsdPerMillionTokens: 8,
    },
  };
}

function successfulResponse(
  outputText = JSON.stringify({
    answer: "PASS",
    confidence: "HIGH",
  }),
): Response {
  return new Response(
    JSON.stringify({
      id: "resp_gate13_fixture",
      model: "gpt-fixture",
      output: [
        {
          type: "message",
          content: [
            {
              type: "output_text",
              text: outputText,
            },
          ],
        },
      ],
      usage: {
        input_tokens: 1000,
        output_tokens: 100,
        total_tokens: 1100,
        input_tokens_details: {
          cached_tokens: 250,
        },
      },
    }),
    {
      status: 200,
      headers: {
        "content-type": "application/json",
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-limit-tokens": "150000",
        "x-ratelimit-remaining-requests": "59",
        "x-ratelimit-remaining-tokens": "148900",
        "x-ratelimit-reset-requests": "10",
        "x-ratelimit-reset-tokens": "300",
      },
    },
  );
}

test("Gate 13 Azure adapter requests strict JSON Schema output", async () => {
  let capturedUrl = "";
  let capturedInit: RequestInit | undefined;

  const provider = new AzureProvider(
    config(),
    async (input, init) => {
      capturedUrl = String(input);
      capturedInit = init;
      return successfulResponse();
    },
  );

  const result = await provider.invokeStructured<{
    answer: string;
    confidence: string;
  }>(request());

  assert.equal(
    capturedUrl,
    "https://orotitan-student.openai.azure.com/openai/v1/responses",
  );

  const body = JSON.parse(String(capturedInit?.body)) as {
    model: string;
    text: {
      format: {
        type: string;
        name: string;
        strict: boolean;
        schema: unknown;
      };
    };
    metadata: Record<string, string>;
  };

  assert.equal(body.model, "orotitan-gate13");
  assert.equal(body.text.format.type, "json_schema");
  assert.equal(body.text.format.name, "Gate13Smoke");
  assert.equal(body.text.format.strict, true);
  assert.deepEqual(body.text.format.schema, schema);
  assert.equal(
    body.metadata.orotitan_operation_id,
    "gate13-smoke-001",
  );

  assert.deepEqual(result.output, {
    answer: "PASS",
    confidence: "HIGH",
  });
  assert.equal(result.providerId, "AZURE_OPENAI");
  assert.equal(result.providerRequestId, "resp_gate13_fixture");
});

test("Gate 13 independently validates provider output against JSON Schema", async () => {
  const provider = new AzureProvider(
    config(),
    async () =>
      successfulResponse(
        JSON.stringify({
          answer: "PASS",
          confidence: "INVALID",
        }),
      ),
  );

  await assert.rejects(
    provider.invokeStructured(request()),
    /VNEXT_AZURE_STRUCTURED_OUTPUT_SCHEMA_FAIL/,
  );
});

test("Gate 13 rejects non-JSON structured output", async () => {
  const provider = new AzureProvider(
    config(),
    async () => successfulResponse("not-json"),
  );

  await assert.rejects(
    provider.invokeStructured(request()),
    /VNEXT_AZURE_STRUCTURED_OUTPUT_NOT_JSON/,
  );
});

test("Gate 13 traces provider token usage, quotas and pinned estimated cost", async () => {
  const provider = new AzureProvider(
    config(),
    async () => successfulResponse(),
  );

  const result = await provider.invokeStructured(request());

  assert.deepEqual(result.usage, {
    inputTokens: 1000,
    cachedInputTokens: 250,
    outputTokens: 100,
    totalTokens: 1100,
  });

  assert.deepEqual(result.rateLimits, {
    limitRequests: 60,
    limitTokens: 150000,
    remainingRequests: 59,
    remainingTokens: 148900,
    resetRequests: "10",
    resetTokens: "300",
    retryAfterMs: null,
  });

  assert.equal(result.cost.uncachedInputTokens, 750);
  assert.equal(result.cost.uncachedInputCostUsd, 0.0015);
  assert.equal(result.cost.cachedInputCostUsd, 0.000125);
  assert.equal(result.cost.outputCostUsd, 0.0008);
  assert.equal(result.cost.totalEstimatedCostUsd, 0.002425);
  assert.equal(
    result.cost.pricingSource,
    "fixture://azure-retail-price",
  );
  assert.equal(result.cost.usageSource, "PROVIDER_RESPONSE");
  assert.equal(
    result.cost.pricingMode,
    "PINNED_RETAIL_ESTIMATE",
  );
});

test("Gate 13 cost estimator separates cached and uncached input", () => {
  const cost = buildModelCostReceipt(
    {
      inputTokens: 2_000_000,
      cachedInputTokens: 500_000,
      outputTokens: 250_000,
      totalTokens: 2_250_000,
    },
    config().priceSchedule,
  );

  assert.equal(cost.uncachedInputCostUsd, 3);
  assert.equal(cost.cachedInputCostUsd, 0.25);
  assert.equal(cost.outputCostUsd, 2);
  assert.equal(cost.totalEstimatedCostUsd, 5.25);
});

test("Gate 13 refuses endpoints outside the Azure inference boundary", () => {
  assert.throws(
    () =>
      new AzureProvider({
        ...config(),
        endpoint: "https://api.example.com/",
      }),
    /VNEXT_AZURE_ENDPOINT_HOST_FORBIDDEN/,
  );

  assert.throws(
    () =>
      new AzureProvider({
        ...config(),
        endpoint: "http://orotitan.openai.azure.com/",
      }),
    /VNEXT_AZURE_ENDPOINT_REQUIRES_HTTPS/,
  );
});

test("Gate 13 fails closed when Azure usage is absent", async () => {
  const provider = new AzureProvider(
    config(),
    async () =>
      new Response(
        JSON.stringify({
          id: "resp_without_usage",
          output_text: JSON.stringify({
            answer: "PASS",
            confidence: "HIGH",
          }),
        }),
        { status: 200 },
      ),
  );

  await assert.rejects(
    provider.invokeStructured(request()),
    /VNEXT_AZURE_USAGE_MISSING/,
  );
});

test("Gate 13 exposes provider HTTP failures without exposing configuration secrets", async () => {
  const provider = new AzureProvider(
    config(),
    async () =>
      new Response(
        JSON.stringify({
          error: {
            code: "RateLimitReached",
            message: "Synthetic 429",
          },
        }),
        {
          status: 429,
          statusText: "Too Many Requests",
          headers: {
            "retry-after-ms": "2000",
          },
        },
      ),
  );

  await assert.rejects(
    async () => provider.invokeStructured(request()),
    (error: unknown) => {
      assert.ok(error instanceof Error);
      assert.match(error.message, /VNEXT_AZURE_HTTP_ERROR:429/);
      assert.doesNotMatch(error.message, /synthetic-test-key/);
      return true;
    },
  );
});

test("Gate 13 environment constructor requires explicit cost provenance", () => {
  assert.throws(
    () =>
      createAzureProviderFromEnv({
        AZURE_OPENAI_ENDPOINT:
          "https://orotitan.openai.azure.com/",
        AZURE_OPENAI_DEPLOYMENT: "fixture",
        AZURE_OPENAI_API_KEY: "secret",
      }),
    /VNEXT_AZURE_ENV_MISSING:AZURE_OPENAI_PRICING_SOURCE/,
  );
});

test("Gate 13 provider contract is replaceable and not Azure-specific", async () => {
  const alternative: AnalyticalModelProvider = {
    providerId: "SYNTHETIC_ALTERNATIVE",
    async invokeStructured<T>() {
      return {
        providerId: "SYNTHETIC_ALTERNATIVE",
        deployment: "fixture",
        providerRequestId: "alt-1",
        modelReported: null,
        output: {
          answer: "PASS",
          confidence: "MEDIUM",
        } as T,
        rawOutputText:
          '{"answer":"PASS","confidence":"MEDIUM"}',
        usage: {
          inputTokens: 10,
          cachedInputTokens: 0,
          outputTokens: 5,
          totalTokens: 15,
        },
        rateLimits: {
          limitRequests: null,
          limitTokens: null,
          remainingRequests: null,
          remainingTokens: null,
          resetRequests: null,
          resetTokens: null,
          retryAfterMs: null,
        },
        cost: {
          currency: "USD",
          pricingSource: "fixture://alternative",
          pricingEffectiveDate: "2026-09-22",
          inputTokens: 10,
          cachedInputTokens: 0,
          uncachedInputTokens: 10,
          outputTokens: 5,
          inputUsdPerMillionTokens: 1,
          cachedInputUsdPerMillionTokens: 1,
          outputUsdPerMillionTokens: 1,
          uncachedInputCostUsd: 0.00001,
          cachedInputCostUsd: 0,
          outputCostUsd: 0.000005,
          totalEstimatedCostUsd: 0.000015,
          usageSource: "PROVIDER_RESPONSE",
          pricingMode: "PINNED_RETAIL_ESTIMATE",
        },
      };
    },
  };

  const result = await alternative.invokeStructured(request());
  assert.equal(result.providerId, "SYNTHETIC_ALTERNATIVE");
});


test("Gate 13 provider layer has no Registry, Supabase, state-machine or publication dependency", async () => {
  const { readFileSync } = await import("node:fs");
  const { resolve } = await import("node:path");

  for (const relativePath of [
    "runtime/vnext/analytical-model-provider.ts",
    "runtime/vnext/azure-provider.ts",
  ]) {
    const source = readFileSync(
      resolve(process.cwd(), relativePath),
      "utf8",
    );

    assert.doesNotMatch(source, /from\s+["'][^"']*supabase/i);
    assert.doesNotMatch(source, /from\s+["'][^"']*state-machine/i);
    assert.doesNotMatch(source, /from\s+["'][^"']*run-controller/i);
    assert.doesNotMatch(source, /from\s+["'][^"']*post-stage/i);
    assert.doesNotMatch(source, /from\s+["'][^"']*publication/i);
    assert.doesNotMatch(source, /SUPABASE_SERVICE_ROLE_KEY/);
    assert.doesNotMatch(source, /OROTITAN_PUBLICATION_ENABLED/);
  }
});
