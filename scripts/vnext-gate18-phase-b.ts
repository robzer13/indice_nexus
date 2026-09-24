import { createHash, randomUUID } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  mkdirSync,
  writeFileSync,
} from "node:fs";
import { basename, dirname, join, resolve } from "node:path";

import {
  generateText,
  NoObjectGeneratedError,
  Output,
} from "ai";

import pilotJson from "../calibration/vnext/OROTITAN_GATE18_PILOT_V0.1.json";
import {
  GATE18_PHASE_B_MODEL_CANDIDATES,
} from "../runtime/vnext/model-calibration-phase-b-profiles";
import {
  assertValidGate18EngineeringReceipt,
  type Gate18EngineeringReceipt,
} from "../runtime/vnext/model-calibration-evidence";
import {
  type Gate18ArtifactPin,
  type Gate18PilotCompany,
} from "../runtime/vnext/model-calibration-pilot";
import {
  GATE18_PHASE_B_V06_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V06_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V06_MAX_OUTPUT_TOKENS,
  GATE18_PHASE_B_PROTOCOL_VERSION,
  GATE18_PHASE_B_V06_MODULE_ID,
  GATE18_PHASE_B_V06_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V06_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V06_SCOPE,
  GATE18_PHASE_B_V06_SYSTEM_PROMPT,
  assertGate18PhaseBV06Semantics,
  buildGate18PhaseBV06ModelInput,
  buildVerifiedGate18V06MoatPacket,
  gate18PhaseBV06GenerationSchemaSha256,
  gate18PhaseBV06OutputSchema,
  gate18PhaseBV06PromptTemplateSha256,
  type Gate18PhaseBV06Output,
} from "../runtime/vnext/model-calibration-pilot-v06";

interface CliOptions {
  caseSelector: string;
  privateRepoRoot: string;
  outputDir: string;
  execute: boolean;
  maxCaseSpendUsd: number | null;
  modelLabels: string[];
  allowMultiModel: boolean;
}

interface ModelPricing {
  input: number;
  output: number;
}

interface SafeFailure {
  label: string;
  modelId: string;
  error: string;
  errorType: string;
  cause: string | null;
  finishReason: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  totalTokens: number | null;
  generatedTextChars: number | null;
  generatedTextSha256: string | null;
  providerRequestId: string | null;
  gatewayCostUsd: number | null;
}

interface StepDiagnosticSnapshot {
  finishReason: string | null;
  inputTokens: number | null;
  outputTokens: number | null;
  reasoningTokens: number | null;
  totalTokens: number | null;
  textChars: number;
  textSha256: string;
  providerRequestId: string | null;
  gatewayCostUsd: number | null;
  providerMetadata: unknown;
}

const DEFAULT_OUTPUT_DIR =
  "calibration/vnext/private-runs";

function sha256Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

function parseArgs(argv: readonly string[]): CliOptions {
  let caseSelector = "";
  let privateRepoRoot = "";
  let outputDir = DEFAULT_OUTPUT_DIR;
  let execute = false;
  let maxCaseSpendUsd: number | null = null;
  const modelLabels: string[] = [];
  let allowMultiModel = false;

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--case") {
      caseSelector = argv[++index] ?? "";
      continue;
    }
    if (arg === "--private-repo-root") {
      privateRepoRoot = argv[++index] ?? "";
      continue;
    }
    if (arg === "--output-dir") {
      outputDir = argv[++index] ?? "";
      continue;
    }
    if (arg === "--max-case-spend-usd") {
      const raw = argv[++index] ?? "";
      const parsed = Number(raw);
      if (!Number.isFinite(parsed) || parsed <= 0) {
        throw new Error(
          "VNEXT_GATE18_PHASE_B_MAX_CASE_SPEND_INVALID",
        );
      }
      maxCaseSpendUsd = parsed;
      continue;
    }
    if (arg === "--model") {
      modelLabels.push((argv[++index] ?? "").trim().toUpperCase());
      continue;
    }
    if (arg === "--allow-multi-model") {
      allowMultiModel = true;
      continue;
    }
    if (arg === "--execute") {
      execute = true;
      continue;
    }

    throw new Error(
      `VNEXT_GATE18_PHASE_B_UNKNOWN_ARG:${arg}`,
    );
  }

  if (caseSelector.trim().length === 0) {
    throw new Error("VNEXT_GATE18_PHASE_B_CASE_REQUIRED");
  }
  if (privateRepoRoot.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_PRIVATE_REPO_ROOT_REQUIRED",
    );
  }
  if (outputDir.trim().length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_OUTPUT_DIR_REQUIRED",
    );
  }
  if (execute && maxCaseSpendUsd === null) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_EXECUTION_SPEND_CAP_REQUIRED",
    );
  }

  if (execute && modelLabels.length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_EXPLICIT_MODEL_REQUIRED",
    );
  }

  if (
    execute &&
    modelLabels.length > 1 &&
    !allowMultiModel
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_MULTI_MODEL_REQUIRES_EXPLICIT_ALLOW",
    );
  }

  for (const label of modelLabels) {
    if (
      !GATE18_PHASE_B_MODEL_CANDIDATES.some(
        (candidate) => candidate.label === label,
      )
    ) {
      throw new Error(
        `VNEXT_GATE18_PHASE_B_UNKNOWN_MODEL_LABEL:${label}`,
      );
    }
  }

  return {
    caseSelector,
    privateRepoRoot: resolve(privateRepoRoot),
    outputDir: resolve(outputDir),
    execute,
    maxCaseSpendUsd,
    modelLabels,
    allowMultiModel,
  };
}

function findCompany(selector: string): Gate18PilotCompany {
  const normalized = selector.trim().toLowerCase();

  const matches = pilotJson.companies.filter((company) => {
    return (
      company.display_name.toLowerCase() === normalized ||
      company.role.toLowerCase() === normalized ||
      company.source_run_id.toLowerCase() === normalized
    );
  });

  if (matches.length !== 1) {
    throw new Error(
      matches.length === 0
        ? "VNEXT_GATE18_PHASE_B_CASE_NOT_FOUND"
        : "VNEXT_GATE18_PHASE_B_CASE_AMBIGUOUS",
    );
  }

  return matches[0] as Gate18PilotCompany;
}

function artifactReader(
  privateRepoRoot: string,
): (pin: Gate18ArtifactPin) => Uint8Array {
  return (pin) => {
    const normalizedRoot = privateRepoRoot.replaceAll("\\", "/");
    const absolute = resolve(privateRepoRoot, pin.path);
    const normalizedAbsolute = absolute.replaceAll("\\", "/");
    const normalizedPrefix = normalizedRoot.endsWith("/")
      ? normalizedRoot
      : `${normalizedRoot}/`;

    if (!normalizedAbsolute.startsWith(normalizedPrefix)) {
      throw new Error(
        "VNEXT_GATE18_PHASE_B_PRIVATE_PATH_ESCAPE",
      );
    }

    try {
      execFileSync(
        "git",
        [
          "-C",
          privateRepoRoot,
          "cat-file",
          "-e",
          `${pin.commit_sha}^{commit}`,
        ],
        {
          stdio: ["ignore", "ignore", "pipe"],
        },
      );

      return execFileSync(
        "git",
        [
          "-C",
          privateRepoRoot,
          "show",
          `${pin.commit_sha}:${pin.path.replaceAll("\\", "/")}`,
        ],
        {
          encoding: "buffer",
          maxBuffer: 20 * 1024 * 1024,
          stdio: ["ignore", "pipe", "pipe"],
        },
      );
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message.replace(/[\r\n\t]+/g, " ").slice(0, 300)
          : "UNKNOWN_GIT_READ_ERROR";
      throw new Error(
        `VNEXT_GATE18_PHASE_B_PINNED_GIT_READ_FAILED:${message}`,
      );
    }
  };
}

function asRecord(
  value: unknown,
): Record<string, unknown> | null {
  if (
    value === null ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    return null;
  }
  return value as Record<string, unknown>;
}

function parseGatewayCost(
  providerMetadata: unknown,
): number | null {
  const root = asRecord(providerMetadata);
  const gateway = asRecord(root?.gateway);
  const raw = gateway?.gatewayCost;

  if (typeof raw === "number" && Number.isFinite(raw)) {
    return raw;
  }
  if (typeof raw === "string") {
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function providerRequestId(
  providerMetadata: unknown,
): string | null {
  const root = asRecord(providerMetadata);
  const gateway = asRecord(root?.gateway);
  const routing = asRecord(gateway?.routing);
  const attempts = routing?.modelAttempts;

  if (!Array.isArray(attempts)) {
    return null;
  }

  for (const modelAttempt of attempts) {
    const providers = asRecord(modelAttempt)?.providerAttempts;
    if (!Array.isArray(providers)) {
      continue;
    }

    for (const providerAttempt of providers) {
      const id = asRecord(providerAttempt)?.providerRequestId;
      if (typeof id === "string" && id.length > 0) {
        return id;
      }
    }
  }

  return null;
}

function safeError(error: unknown): string {
  if (!(error instanceof Error)) {
    return "UNKNOWN_ERROR";
  }

  return error.message
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 500);
}

function safeCause(error: unknown): string | null {
  if (!(error instanceof Error)) {
    return null;
  }

  const candidate = (error as Error & { cause?: unknown }).cause;
  if (candidate === undefined) {
    return null;
  }

  if (candidate instanceof Error) {
    return safeError(candidate);
  }

  if (typeof candidate === "string") {
    return candidate
      .replace(/[\r\n\t]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 500);
  }

  return "NON_ERROR_CAUSE";
}

function failureDiagnostic(
  label: string,
  modelId: string,
  error: unknown,
  step: StepDiagnosticSnapshot | null,
): SafeFailure {
  let finishReason = step?.finishReason ?? null;
  let inputTokens = step?.inputTokens ?? null;
  let outputTokens = step?.outputTokens ?? null;
  let reasoningTokens = step?.reasoningTokens ?? null;
  let totalTokens = step?.totalTokens ?? null;
  let generatedTextChars = step?.textChars ?? null;
  let generatedTextSha256 = step?.textSha256 ?? null;

  if (NoObjectGeneratedError.isInstance(error)) {
    finishReason = error.finishReason ?? finishReason;

    if (error.usage) {
      inputTokens = error.usage.inputTokens ?? inputTokens;
      outputTokens = error.usage.outputTokens ?? outputTokens;
      reasoningTokens =
        error.usage.outputTokenDetails.reasoningTokens ??
        reasoningTokens;
      totalTokens = error.usage.totalTokens ?? totalTokens;
    }

    if (typeof error.text === "string") {
      generatedTextChars = error.text.length;
      generatedTextSha256 = sha256Hex(error.text);
    }
  }

  return {
    label,
    modelId,
    error: safeError(error),
    errorType:
      error instanceof Error
        ? error.name || "Error"
        : "UNKNOWN_ERROR",
    cause: safeCause(error),
    finishReason,
    inputTokens,
    outputTokens,
    reasoningTokens,
    totalTokens,
    generatedTextChars,
    generatedTextSha256,
    providerRequestId: step?.providerRequestId ?? null,
    gatewayCostUsd: step?.gatewayCostUsd ?? null,
  };
}

async function loadPricing(): Promise<
  ReadonlyMap<string, ModelPricing>
> {
  const response = await fetch(
    "https://ai-gateway.vercel.sh/v1/models",
  );

  if (!response.ok) {
    throw new Error(
      `VNEXT_GATE18_PHASE_B_PRICE_CATALOG_HTTP_${response.status}`,
    );
  }

  const payload = (await response.json()) as {
    data?: Array<{
      id?: string;
      pricing?: {
        input?: string;
        output?: string;
      };
    }>;
  };

  if (!Array.isArray(payload.data)) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_PRICE_CATALOG_INVALID",
    );
  }

  const pricing = new Map<string, ModelPricing>();

  for (const model of payload.data) {
    if (
      typeof model.id !== "string" ||
      typeof model.pricing?.input !== "string" ||
      typeof model.pricing?.output !== "string"
    ) {
      continue;
    }

    const input = Number(model.pricing.input);
    const output = Number(model.pricing.output);

    if (
      Number.isFinite(input) &&
      input >= 0 &&
      Number.isFinite(output) &&
      output >= 0
    ) {
      pricing.set(model.id, { input, output });
    }
  }

  return pricing;
}

function conservativeCaseCostCeiling(
  input: string,
  pricing: ReadonlyMap<string, ModelPricing>,
  candidates: readonly (typeof GATE18_PHASE_B_MODEL_CANDIDATES)[number][],
): {
  approximateInputTokenCeiling: number;
  perModel: Array<{
    label: string;
    modelId: string;
    costCeilingUsd: number;
  }>;
  totalCostCeilingUsd: number;
} {
  // Deliberately conservative for English/JSON: assume at most
  // two UTF-16 code units per input token, then price the full
  // output-token allowance. This is an execution guard, not a
  // methodology or permanent budget rule.
  const approximateInputTokenCeiling = Math.ceil(
    input.length / 2,
  );

  const perModel = candidates.map(
    (candidate) => {
      const modelPricing = pricing.get(candidate.modelId);
      if (!modelPricing) {
        throw new Error(
          `VNEXT_GATE18_PHASE_B_PRICING_MISSING:${candidate.modelId}`,
        );
      }

      const inferenceCeiling =
        approximateInputTokenCeiling *
          modelPricing.input +
        GATE18_PHASE_B_V06_MAX_OUTPUT_TOKENS *
          modelPricing.output;

      // Reserve one tenth of a cent for gateway/reporting
      // overhead per call. Current observed smoke overhead was
      // lower, but execution fails closed if the aggregate cap
      // cannot absorb this reserve.
      const costCeilingUsd = inferenceCeiling + 0.001;

      return {
        label: candidate.label,
        modelId: candidate.modelId,
        costCeilingUsd,
      };
    },
  );

  return {
    approximateInputTokenCeiling,
    perModel,
    totalCostCeilingUsd: perModel.reduce(
      (sum, item) => sum + item.costCeilingUsd,
      0,
    ),
  };
}

function writePrivateResult(
  outputDir: string,
  caseId: string,
  payload: unknown,
): string {
  mkdirSync(outputDir, { recursive: true });

  const stamp = new Date()
    .toISOString()
    .replaceAll(":", "")
    .replaceAll(".", "");
  const safeCase = caseId.replace(/[^A-Za-z0-9_-]+/g, "_");
  const path = join(
    outputDir,
    `${stamp}__${safeCase}.json`,
  );

  writeFileSync(
    path,
    `${JSON.stringify(payload, null, 2)}\n`,
    "utf8",
  );

  return path;
}

async function main(): Promise<void> {
  const options = parseArgs(process.argv.slice(2));
  const company = findCompany(options.caseSelector);

  const verified = buildVerifiedGate18V06MoatPacket(
    company,
    artifactReader(options.privateRepoRoot),
  );

  const modelInput = buildGate18PhaseBV06ModelInput(
    verified.packet,
  );
  const selectedCandidates =
    options.modelLabels.length === 0
      ? [...GATE18_PHASE_B_MODEL_CANDIDATES]
      : GATE18_PHASE_B_MODEL_CANDIDATES.filter((candidate) =>
          options.modelLabels.includes(candidate.label),
        );

  if (selectedCandidates.length === 0) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_MODEL_SELECTION_EMPTY",
    );
  }

  const pricing = await loadPricing();
  const ceiling = conservativeCaseCostCeiling(
    modelInput,
    pricing,
    selectedCandidates,
  );

  const dryRunSummary = {
    gate: 18,
    phase: "B_COMPANY_CALIBRATION",
    mode: options.execute ? "EXECUTE" : "DRY_RUN",
    publicationAuthority: false,
    productionMutation: false,
    modelWinnerSelected: false,
    case: {
      caseId: verified.packet.case_id,
      displayName: verified.packet.display_name,
      role: verified.packet.role,
      sourceRunId: verified.packet.source_run_id,
      dataCutoff: verified.packet.data_cutoff,
    },
    packet: {
      scope: GATE18_PHASE_B_V06_SCOPE,
      sourceEvidenceItems: verified.sourceEvidenceCount,
      sourceConflicts: verified.sourceConflictCount,
      evidenceItems: verified.packet.evidence_items.length,
      conflicts: verified.packet.conflicts.length,
      packetSha256: verified.packetSha256,
      evidenceLedgerSha256:
        verified.evidenceLedgerSha256,
      conflictLedgerSha256:
        verified.conflictLedgerSha256,
      serializedChars: JSON.stringify(verified.packet).length,
    },
    prompt: {
      moduleId: GATE18_PHASE_B_V06_MODULE_ID,
      promptTemplateId:
        GATE18_PHASE_B_V06_PROMPT_TEMPLATE_ID,
      promptTemplateVersion:
        GATE18_PHASE_B_V06_PROMPT_TEMPLATE_VERSION,
      promptTemplateSha256:
        gate18PhaseBV06PromptTemplateSha256(),
      generationSchemaId:
        GATE18_PHASE_B_V06_GENERATION_SCHEMA_ID,
      generationSchemaVersion:
        GATE18_PHASE_B_V06_GENERATION_SCHEMA_VERSION,
      generationSchemaSha256:
        gate18PhaseBV06GenerationSchemaSha256(),
      maxOutputTokens:
        GATE18_PHASE_B_V06_MAX_OUTPUT_TOKENS,
    },
    conservativeCostGuard: ceiling,
    explicitSpendCapUsd: options.maxCaseSpendUsd,
    allowMultiModel: options.allowMultiModel,
    selectedModels: selectedCandidates.map((candidate) => ({
      label: candidate.label,
      modelId: candidate.modelId,
      reasoning: candidate.reasoning,
    })),
  };

  if (!options.execute) {
    console.log(JSON.stringify(dryRunSummary, null, 2));
    return;
  }

  if (
    !process.env.VERCEL_OIDC_TOKEN ||
    process.env.VERCEL_OIDC_TOKEN.trim().length === 0
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_VERCEL_OIDC_TOKEN_REQUIRED",
    );
  }

  if (
    options.maxCaseSpendUsd === null ||
    ceiling.totalCostCeilingUsd >
      options.maxCaseSpendUsd
  ) {
    throw new Error(
      "VNEXT_GATE18_PHASE_B_CONSERVATIVE_COST_CEILING_EXCEEDS_CAP",
    );
  }

  const executions: Array<{
    engineering: Gate18EngineeringReceipt;
    output: Gate18PhaseBV06Output;
    providerMetadata: unknown;
  }> = [];
  const failures: SafeFailure[] = [];
  let observedGatewayCostUsd = 0;

  for (const candidate of selectedCandidates) {
    const candidateCeiling =
      ceiling.perModel.find(
        (item) => item.label === candidate.label,
      )?.costCeilingUsd ?? null;

    if (candidateCeiling === null) {
      throw new Error(
        `VNEXT_GATE18_PHASE_B_COST_CEILING_MISSING:${candidate.label}`,
      );
    }

    if (
      observedGatewayCostUsd + candidateCeiling >
      options.maxCaseSpendUsd
    ) {
      failures.push({
        label: candidate.label,
        modelId: candidate.modelId,
        error:
          "VNEXT_GATE18_PHASE_B_PRECALL_SPEND_CAP_WOULD_BE_EXCEEDED",
        errorType: "SPEND_GUARD",
        cause: null,
        finishReason: null,
        inputTokens: null,
        outputTokens: null,
        reasoningTokens: null,
        totalTokens: null,
        generatedTextChars: null,
        generatedTextSha256: null,
        providerRequestId: null,
        gatewayCostUsd: null,
      });
      break;
    }

    const startedAt = performance.now();
    let stepSnapshot: StepDiagnosticSnapshot | null = null;
    let callCostAccounted = false;

    try {
      const result = await generateText({
        model: candidate.modelId,
        reasoning: candidate.reasoning,
        output: Output.object({
          name: "orotitan_gate18_phase_b_evidence_audit",
          description:
            "OroTitan Gate 18 assisted evidence-audit calibration output.",
          schema: gate18PhaseBV06OutputSchema,
        }),
        system: GATE18_PHASE_B_V06_SYSTEM_PROMPT,
        prompt: modelInput,
        maxOutputTokens:
          GATE18_PHASE_B_V06_MAX_OUTPUT_TOKENS,
        providerOptions: {
          gateway: {
            tags: [
              "project:orotitan",
              "gate:18",
              "phase:b-company-calibration",
              `case:${company.source_run_id}`,
              `model:${candidate.label.toLowerCase()}`,
            ],
          },
        },
        onStepFinish(step) {
          const textValue =
            typeof step.text === "string" ? step.text : "";
          stepSnapshot = {
            finishReason: step.finishReason ?? null,
            inputTokens: step.usage.inputTokens ?? null,
            outputTokens: step.usage.outputTokens ?? null,
            reasoningTokens:
              step.usage.outputTokenDetails.reasoningTokens ??
              null,
            totalTokens: step.usage.totalTokens ?? null,
            textChars: textValue.length,
            textSha256: sha256Hex(textValue),
            providerRequestId:
              providerRequestId(step.providerMetadata) ??
              step.response.id ??
              null,
            gatewayCostUsd: parseGatewayCost(
              step.providerMetadata,
            ),
            providerMetadata:
              step.providerMetadata ?? null,
          };
        },
      });

      const capturedStep =
        stepSnapshot as StepDiagnosticSnapshot | null;
      const gatewayCost =
        parseGatewayCost(result.providerMetadata) ??
        capturedStep?.gatewayCostUsd ??
        null;

      if (gatewayCost === null) {
        throw new Error(
          "VNEXT_GATE18_PHASE_B_GATEWAY_COST_MISSING",
        );
      }

      observedGatewayCostUsd += gatewayCost;
      callCostAccounted = true;

      const output = result.output;
      let semanticValid = true;
      try {
        assertGate18PhaseBV06Semantics(
          verified.packet,
          output,
        );
      } catch {
        semanticValid = false;
      }

      const outputJson = JSON.stringify(output);
      const receipt: Gate18EngineeringReceipt = {
        invocation: {
          caseId: company.source_run_id,
          displayName: company.display_name,
          sourceRunId: company.source_run_id,
          dataCutoff: company.data_cutoff,
          moduleId: GATE18_PHASE_B_V06_MODULE_ID,
          modelLabel: candidate.label,
          modelId: candidate.modelId,
          repetition: 1,
          promptTemplateId:
            GATE18_PHASE_B_V06_PROMPT_TEMPLATE_ID,
          promptTemplateVersion:
            GATE18_PHASE_B_V06_PROMPT_TEMPLATE_VERSION,
          promptTemplateSha256:
            gate18PhaseBV06PromptTemplateSha256(),
          generationSchemaId:
            GATE18_PHASE_B_V06_GENERATION_SCHEMA_ID,
          generationSchemaVersion:
            GATE18_PHASE_B_V06_GENERATION_SCHEMA_VERSION,
          generationSchemaSha256:
            gate18PhaseBV06GenerationSchemaSha256(),
          evidencePacketSha256:
            verified.packetSha256,
        },
        executionId: randomUUID(),
        providerRequestId: providerRequestId(
          result.providerMetadata,
        ),
        schemaValid: true,
        semanticValid,
        latencyMs: Math.round(
          performance.now() - startedAt,
        ),
        inputTokens:
          result.totalUsage.inputTokens ?? null,
        cachedInputTokens: null,
        outputTokens:
          result.totalUsage.outputTokens ?? null,
        reasoningTokens:
          result.usage.outputTokenDetails
            .reasoningTokens ?? null,
        totalTokens:
          result.totalUsage.totalTokens ?? null,
        retryCount: 0,
        estimatedCostUsd: gatewayCost,
        costProvenance:
          "VERCEL_AI_GATEWAY_PROVIDER_METADATA.gatewayCost",
        finishReason: result.finishReason ?? null,
        responseSha256: sha256Hex(outputJson),
      };

      assertValidGate18EngineeringReceipt(receipt);

      executions.push({
        engineering: receipt,
        output,
        providerMetadata:
          result.providerMetadata ?? null,
      });
    } catch (error) {
      const capturedStep =
        stepSnapshot as StepDiagnosticSnapshot | null;

      if (
        !callCostAccounted &&
        capturedStep?.gatewayCostUsd !== null &&
        capturedStep?.gatewayCostUsd !== undefined
      ) {
        observedGatewayCostUsd +=
          capturedStep.gatewayCostUsd;
        callCostAccounted = true;
      }

      failures.push(
        failureDiagnostic(
          candidate.label,
          candidate.modelId,
          error,
          capturedStep,
        ),
      );
    }
  }

  const payload = {
    format:
      `OROTITAN_GATE18_PHASE_B_PRIVATE_RUN_V${GATE18_PHASE_B_PROTOCOL_VERSION}`,
    privateArtifact: true,
    publicationAuthority: false,
    productionMutation: false,
    modelWinnerSelected: false,
    createdAt: new Date().toISOString(),
    dryRunSummary,
    observedGatewayCostUsd,
    executions,
    failures,
  };

  const outputPath = writePrivateResult(
    options.outputDir,
    company.source_run_id,
    payload,
  );

  console.log(
    JSON.stringify(
      {
        gate: 18,
        phase: "B_COMPANY_CALIBRATION",
        status:
          failures.length === 0 &&
          executions.length ===
            selectedCandidates.length
            ? "COMPLETE"
            : "PARTIAL_OR_FAILED",
        displayName: company.display_name,
        modelsCompleted: executions.length,
        failures: failures.map((failure) => ({
          label: failure.label,
          modelId: failure.modelId,
          error: failure.error,
          errorType: failure.errorType,
          cause: failure.cause,
          finishReason: failure.finishReason,
          inputTokens: failure.inputTokens,
          outputTokens: failure.outputTokens,
          reasoningTokens: failure.reasoningTokens,
          totalTokens: failure.totalTokens,
          generatedTextChars: failure.generatedTextChars,
          generatedTextSha256:
            failure.generatedTextSha256,
          providerRequestId:
            failure.providerRequestId,
          gatewayCostUsd: failure.gatewayCostUsd,
        })),
        observedGatewayCostUsd,
        outputFile: join(
          basename(dirname(outputPath)),
          basename(outputPath),
        ),
        outputIsPrivateAndGitignored: true,
      },
      null,
      2,
    ),
  );

  if (
    failures.length > 0 ||
    executions.length !==
      selectedCandidates.length
  ) {
    process.exitCode = 1;
  }
}

void main().catch((error: unknown) => {
  console.error(
    JSON.stringify(
      {
        gate: 18,
        phase: "B_COMPANY_CALIBRATION",
        status: "FAILED_CLOSED",
        error: safeError(error),
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
});
