import { randomUUID } from "node:crypto";

import { generateText, Output } from "ai";
import { NextResponse } from "next/server";
import { z } from "zod";

import pilotJson from "@/calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import {
  GATE18_MODEL_CANDIDATES,
} from "@/runtime/vnext/model-calibration";
import {
  assertValidGate18EngineeringReceipt,
  type Gate18EngineeringReceipt,
} from "@/runtime/vnext/model-calibration-evidence";
import {
  GATE18_CONFLICT_REASONING_PROBE_ID,
  GATE18_CONFLICT_REASONING_PROMPT,
  GATE18_CONFLICT_REASONING_PROMPT_SHA256,
  GATE18_CONFLICT_REASONING_PROMPT_VERSION,
  GATE18_CONFLICT_REASONING_SCHEMA_SHA256,
  buildGate18ConflictReasoningUserPrompt,
  gate18ConflictReasoningOutputSchema,
  parseGate18ProbeSources,
  sha256Utf8,
  validateGate18ProbeReferences,
} from "@/runtime/vnext/model-calibration-probe";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 300;

const requestSchema = z.object({
  run: z.literal("gate18-pilot-v0-1"),
  displayName: z.string().min(1),
  sourceRunId: z.string().min(1),
  dataCutoff: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  modelLabel: z.enum(["LUNA", "TERRA", "SOL", "ASTRA"]),
  repetition: z.number().int().min(1).max(10),
  evidenceLedgerRaw: z.string().min(2).max(500_000),
  conflictLedgerRaw: z.string().min(2).max(250_000),
});

function nullableToken(value: number | undefined): number | null {
  return value ?? null;
}

function errorResponse(
  code: string,
  status: number,
  detail?: string,
) {
  return NextResponse.json(
    {
      gate: 18,
      status: "FAILED",
      code,
      detail: detail ?? null,
      modelWinnerSelected: false,
      publicationAuthority: false,
    },
    { status },
  );
}

export async function POST(request: Request) {
  if (process.env.VERCEL_ENV === "production") {
    return errorResponse(
      "GATE18_COMPANY_PROBE_PREVIEW_ONLY",
      403,
    );
  }

  let body: z.infer<typeof requestSchema>;

  try {
    body = requestSchema.parse(await request.json());
  } catch {
    return errorResponse("GATE18_REQUEST_INVALID", 400);
  }

  const company = pilotJson.companies.find(
    (candidate) =>
      candidate.display_name === body.displayName &&
      candidate.source_run_id === body.sourceRunId &&
      candidate.data_cutoff === body.dataCutoff,
  );

  if (!company) {
    return errorResponse(
      "GATE18_PILOT_MEMBER_NOT_PINNED",
      400,
    );
  }

  const evidenceSha256 = sha256Utf8(body.evidenceLedgerRaw);
  const conflictSha256 = sha256Utf8(body.conflictLedgerRaw);

  if (evidenceSha256 !== company.evidence_ledger.sha256) {
    return errorResponse(
      "GATE18_EVIDENCE_LEDGER_HASH_MISMATCH",
      400,
    );
  }

  if (conflictSha256 !== company.conflict_ledger.sha256) {
    return errorResponse(
      "GATE18_CONFLICT_LEDGER_HASH_MISMATCH",
      400,
    );
  }

  let sources;

  try {
    sources = parseGate18ProbeSources(
      body.evidenceLedgerRaw,
      body.conflictLedgerRaw,
    );
  } catch (error) {
    return errorResponse(
      error instanceof Error
        ? error.message
        : "GATE18_SOURCE_PACKET_INVALID",
      400,
    );
  }

  if (sources.runId !== body.sourceRunId) {
    return errorResponse(
      "GATE18_SOURCE_RUN_ID_MISMATCH",
      400,
    );
  }

  if (sources.dataCutoff !== body.dataCutoff) {
    return errorResponse(
      "GATE18_SOURCE_DATA_CUTOFF_MISMATCH",
      400,
    );
  }

  const model = GATE18_MODEL_CANDIDATES.find(
    (candidate) => candidate.label === body.modelLabel,
  );

  if (!model) {
    return errorResponse(
      "GATE18_MODEL_CANDIDATE_NOT_PINNED",
      400,
    );
  }

  const executionId = randomUUID();
  const inputPacketSha256 = sha256Utf8(
    [
      evidenceSha256,
      conflictSha256,
      body.sourceRunId,
      body.dataCutoff,
    ].join("\n"),
  );
  const userPrompt = buildGate18ConflictReasoningUserPrompt(
    body.displayName,
    body.dataCutoff,
    body.evidenceLedgerRaw,
    body.conflictLedgerRaw,
  );

  const startedAt = performance.now();

  try {
    const result = await generateText({
      model: model.modelId,
      reasoning: model.reasoning,
      system: GATE18_CONFLICT_REASONING_PROMPT,
      prompt: userPrompt,
      maxOutputTokens: 4_000,
      output: Output.object({
        name: "orotitan_gate18_conflict_reasoning_probe",
        description:
          "Calibration-only evidence conflict reasoning output.",
        schema: gate18ConflictReasoningOutputSchema,
      }),
      providerOptions: {
        gateway: {
          tags: [
            "project:orotitan",
            "gate:18",
            "purpose:company-calibration",
            `company:${body.displayName}`,
            `model:${model.label.toLowerCase()}`,
            `repetition:${body.repetition}`,
          ],
        },
      },
    });

    const semantic = validateGate18ProbeReferences(
      result.output,
      sources,
    );
    const outputRaw = JSON.stringify(result.output);

    const receipt: Gate18EngineeringReceipt = {
      invocation: {
        caseId: `${body.sourceRunId}:${GATE18_CONFLICT_REASONING_PROBE_ID}`,
        displayName: body.displayName,
        sourceRunId: body.sourceRunId,
        dataCutoff: body.dataCutoff,
        moduleId: GATE18_CONFLICT_REASONING_PROBE_ID,
        modelLabel: model.label,
        modelId: model.modelId,
        repetition: body.repetition,
        promptTemplateId: GATE18_CONFLICT_REASONING_PROBE_ID,
        promptTemplateVersion:
          GATE18_CONFLICT_REASONING_PROMPT_VERSION,
        promptTemplateSha256:
          GATE18_CONFLICT_REASONING_PROMPT_SHA256,
        generationSchemaId:
          "GATE18_CONFLICT_REASONING_OUTPUT_SCHEMA",
        generationSchemaVersion: "0.1.0",
        generationSchemaSha256:
          GATE18_CONFLICT_REASONING_SCHEMA_SHA256,
        evidencePacketSha256: inputPacketSha256,
      },
      executionId,
      providerRequestId: null,
      schemaValid: true,
      semanticValid: semantic.valid,
      latencyMs: Math.round(performance.now() - startedAt),
      inputTokens: nullableToken(result.usage.inputTokens),
      cachedInputTokens: nullableToken(
        result.usage.inputTokenDetails.cacheReadTokens,
      ),
      outputTokens: nullableToken(result.usage.outputTokens),
      reasoningTokens: nullableToken(
        result.usage.outputTokenDetails.reasoningTokens,
      ),
      totalTokens: nullableToken(result.usage.totalTokens),
      retryCount: 0,
      estimatedCostUsd: null,
      costProvenance: null,
      finishReason: result.finishReason ?? null,
      responseSha256: sha256Utf8(outputRaw),
    };

    assertValidGate18EngineeringReceipt(receipt);

    return NextResponse.json({
      gate: 18,
      status: semantic.valid
        ? "PROBE_COMPLETE"
        : "PROBE_SEMANTIC_INVALID",
      calibrationOnly: true,
      publicationAuthority: false,
      modelWinnerSelected: false,
      receipt,
      semanticIssues: semantic.issues,
      output: result.output,
    });
  } catch (error) {
    console.error("GATE18_COMPANY_PROBE_MODEL_CALL_FAILED", {
      executionId,
      modelLabel: model.label,
      modelId: model.modelId,
      error,
    });

    return errorResponse(
      "GATE18_COMPANY_PROBE_MODEL_CALL_FAILED",
      502,
      error instanceof Error ? error.message : undefined,
    );
  }
}
