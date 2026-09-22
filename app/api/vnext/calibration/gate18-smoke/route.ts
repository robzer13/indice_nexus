import { generateText, Output } from "ai";
import { NextResponse } from "next/server";
import { z } from "zod";

import {
  GATE18_MODEL_CANDIDATES,
  assertGate18SmokeReceipt,
  type Gate18SmokeReceipt,
} from "@/runtime/vnext/model-calibration";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const smokeSchema = z.object({
  gate: z.literal(18),
  status: z.literal("PASS"),
  provider_role: z.literal("CALIBRATION_ONLY"),
  publication_authority: z.literal(false),
});

function tokenOrNull(value: number | undefined): number | null {
  return value ?? null;
}

export async function GET(request: Request) {
  if (process.env.VERCEL_ENV === "production") {
    return NextResponse.json(
      { error: "GATE18_SMOKE_PREVIEW_ONLY" },
      { status: 403 },
    );
  }

  const url = new URL(request.url);

  if (url.searchParams.get("run") !== "gate18-smoke-v0-1") {
    return NextResponse.json(
      { error: "GATE18_SMOKE_EXPLICIT_RUN_REQUIRED" },
      { status: 400 },
    );
  }

  const receipts = await Promise.all(
    GATE18_MODEL_CANDIDATES.map(async (candidate) => {
      const startedAt = performance.now();

      const result = await generateText({
        model: candidate.modelId,
        reasoning: candidate.reasoning,
        output: Output.object({
          name: "orotitan_gate18_smoke",
          description:
            "OroTitan Gate 18 transport and structured-output calibration smoke.",
          schema: smokeSchema,
        }),
        system:
          "You are a bounded calibration model. Return only the requested structured output. Do not use tools.",
        prompt:
          "Return Gate 18 PASS for the calibration-only provider role. Publication authority must be false.",
        maxOutputTokens: 128,
        providerOptions: {
          gateway: {
            tags: [
              "project:orotitan",
              "gate:18",
              "purpose:model-calibration-smoke",
              `model:${candidate.label.toLowerCase()}`,
            ],
          },
        },
      });

      const receipt: Gate18SmokeReceipt = {
        label: candidate.label,
        modelId: candidate.modelId,
        intendedTier: candidate.intendedTier,
        requestedReasoning: candidate.reasoning,
        schemaValid:
          result.output.gate === 18 &&
          result.output.status === "PASS" &&
          result.output.provider_role === "CALIBRATION_ONLY" &&
          result.output.publication_authority === false,
        latencyMs: Math.round(performance.now() - startedAt),
        inputTokens: tokenOrNull(result.totalUsage.inputTokens),
        outputTokens: tokenOrNull(result.totalUsage.outputTokens),
        reasoningTokens: tokenOrNull(
          result.totalUsage.reasoningTokens,
        ),
        totalTokens: tokenOrNull(result.totalUsage.totalTokens),
        finishReason: result.finishReason ?? null,
        providerMetadata: result.providerMetadata ?? null,
      };

      assertGate18SmokeReceipt(receipt);

      return receipt;
    }),
  );

  return NextResponse.json({
    gate: 18,
    status: "SMOKE_PASS",
    calibrationOnly: true,
    publicationAuthority: false,
    modelWinnerSelected: false,
    receipts,
  });
}
