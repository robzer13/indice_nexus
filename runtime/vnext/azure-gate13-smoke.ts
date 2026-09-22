import { createAzureProviderFromEnv } from "./azure-provider";

const smokeSchema = {
  type: "object",
  additionalProperties: false,
  properties: {
    gate: { const: 13 },
    status: { enum: ["PASS"] },
    provider_role: {
      const: "ANALYTICAL_MODEL_ONLY",
    },
    publication_authority: {
      const: false,
    },
  },
  required: [
    "gate",
    "status",
    "provider_role",
    "publication_authority",
  ],
} as const;

async function main(): Promise<void> {
  const provider = createAzureProviderFromEnv();

  const result = await provider.invokeStructured<{
    gate: 13;
    status: "PASS";
    provider_role: "ANALYTICAL_MODEL_ONLY";
    publication_authority: false;
  }>({
    operationId:
      "gate13-live-smoke-" +
      new Date().toISOString().replace(/[^0-9A-Za-z_-]/g, "_"),
    systemPrompt:
      "You are a bounded analytical model adapter smoke test. " +
      "Return only the strict JSON object required by the supplied schema. " +
      "You have no authority over runtime state, databases, or publication.",
    input:
      "Return the Gate 13 smoke-test object confirming the bounded provider role.",
    schemaName: "OroTitanGate13Smoke",
    schema: smokeSchema,
    maxOutputTokens: 120,
    metadata: {
      orotitan_environment: "VNEXT_SHADOW",
      gate: "13",
    },
  });

  process.stdout.write(
    JSON.stringify(
      {
        gate: 13,
        live_call: "PASS",
        provider_id: result.providerId,
        deployment: result.deployment,
        provider_request_id: result.providerRequestId,
        model_reported: result.modelReported,
        structured_output: result.output,
        usage: result.usage,
        rate_limits: result.rateLimits,
        cost: result.cost,
      },
      null,
      2,
    ) + "\n",
  );
}

main().catch((error: unknown) => {
  const message =
    error instanceof Error ? error.message : String(error);
  process.stderr.write(
    JSON.stringify(
      {
        gate: 13,
        live_call: "FAIL",
        error: message,
      },
      null,
      2,
    ) + "\n",
  );
  process.exitCode = 1;
});
