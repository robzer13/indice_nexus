import 'server-only';

import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { generateText, Output } from 'ai';
import { z } from 'zod';
import { computeOqs, DIMENSION_KEYS, type DimensionKey } from '@/lib/orotitan-equity/v1/quality';

const scoreSchema = z.number().int().min(0).max(100).refine((value) => value % 5 === 0, {
  message: 'Scores must use 5-point increments.',
}).nullable();

const dimensionJudgmentSchema = z.object({
  score: scoreSchema,
  rationale: z.string().min(1).max(2000),
  decisive_factors: z.array(z.string().min(1).max(500)).min(1).max(5),
}).strict();

const replayOutputSchema = z.object({
  dimensions: z.object({
    MOAT: dimensionJudgmentSchema,
    RUNWAY: dimensionJudgmentSchema,
    RETURN_QUALITY: dimensionJudgmentSchema,
    CASH_ECONOMICS: dimensionJudgmentSchema,
    CAPITAL_ALLOCATION: dimensionJudgmentSchema,
    MANAGEMENT_GOVERNANCE: dimensionJudgmentSchema,
    RESILIENCE_RISK: dimensionJudgmentSchema,
  }).strict(),
  global_limitations: z.array(z.string().min(1).max(500)).max(10),
}).strict();

const inputPackSchema = z.object({
  pack_schema_version: z.literal('1.0.0'),
  pack_type: z.literal('OROTITAN_V2_PRE_SCORE_REPLAY_INPUT'),
  case_id: z.string().regex(/^V2REF-\d{3}$/),
  company: z.string().min(1),
  run_id: z.string().uuid(),
  sanitizer_version: z.literal('1.0.0'),
  prompt_version: z.literal('OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1'),
  source: z.object({
    repository: z.string(),
    path: z.string(),
    commit_sha: z.string(),
    blob_sha: z.string(),
    registry_content_sha256: z.string().regex(/^[a-f0-9]{64}$/),
  }).strict(),
  leakage_controls: z.object({
    historical_dimension_scores_removed: z.literal(true),
    aggregate_scores_removed: z.literal(true),
    certification_removed: z.literal(true),
    valuation_removed: z.literal(true),
    terminal_verdict_removed: z.literal(true),
    forbidden_key_scan: z.literal('PASS'),
  }).strict(),
  analytical_input: z.unknown(),
}).strict();

export type ScoringReplayOutput = z.infer<typeof replayOutputSchema>;

export interface ScoringReplayRequest {
  caseId: string;
  repetitionIndex: number;
  model: string;
}

export interface ScoringReplayExecutionRecord {
  campaign_id: 'OROTITAN_V2_REPRODUCIBILITY_V0.1';
  phase_id: 'PHASE_A1_SCORING_JUDGMENT_REPLAY';
  case_id: string;
  repetition_index: number;
  engine_fingerprint: string;
  model_provider: string;
  model_name: string;
  model_version: string;
  reasoning_config: 'PROVIDER_DEFAULT';
  prompt_artifact_version: 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1';
  input_package_sha256: string;
  started_at: string;
  finished_at: string;
  duration_ms: number;
  dimension_scores: Record<DimensionKey, number | null>;
  dimension_rationales: Record<DimensionKey, string>;
  oqs_raw: number | null;
  weak_link_cap: number | null;
  oqs: number | null;
  i2_computation_status: 'PASS' | 'NOT_COMPUTABLE';
  finish_reason: string;
  usage: unknown;
  output_sha256: string;
}

function sha256Text(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

function parseModelId(model: string): { provider: string; name: string; version: string } {
  const slash = model.indexOf('/');
  const provider = slash >= 0 ? model.slice(0, slash) : 'gateway';
  const name = slash >= 0 ? model.slice(slash + 1) : model;
  return { provider, name, version: name };
}

async function loadReplayAssets(caseId: string) {
  if (!/^V2REF-\d{3}$/.test(caseId)) throw new Error('Invalid benchmark case id.');
  const root = process.cwd();
  const [packText, systemPrompt] = await Promise.all([
    readFile(path.join(root, 'benchmark', 'input-packs', 'v2-reference-20', caseId + '.json'), 'utf8'),
    readFile(path.join(root, 'benchmark', 'prompts', 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1.md'), 'utf8'),
  ]);
  const pack = inputPackSchema.parse(JSON.parse(packText));
  if (pack.case_id !== caseId) throw new Error('Benchmark input pack identity mismatch.');
  return { pack, packText, systemPrompt };
}

function computeI2(scores: Record<DimensionKey, number | null>) {
  if (DIMENSION_KEYS.some((dimension) => scores[dimension] === null)) {
    return { oqsRaw: null, weakLinkCap: null, oqs: null, status: 'NOT_COMPUTABLE' as const };
  }
  const numeric = Object.fromEntries(
    DIMENSION_KEYS.map((dimension) => [dimension, scores[dimension] as number]),
  ) as Record<DimensionKey, number>;
  const result = computeOqs(numeric);
  if (typeof result.oqsRaw !== 'number' || typeof result.weakLinkCap !== 'number' || typeof result.oqs !== 'number') {
    throw new Error('Unexpected range result for scalar replay dimensions.');
  }
  return { oqsRaw: result.oqsRaw, weakLinkCap: result.weakLinkCap, oqs: result.oqs, status: 'PASS' as const };
}

export async function runScoringReplay(request: ScoringReplayRequest): Promise<ScoringReplayExecutionRecord> {
  if (!Number.isInteger(request.repetitionIndex) || request.repetitionIndex < 1) {
    throw new Error('repetitionIndex must be a positive integer.');
  }
  if (!request.model || !request.model.includes('/')) {
    throw new Error('A fully-qualified Gateway model id (provider/model) is required.');
  }

  const { pack, packText, systemPrompt } = await loadReplayAssets(request.caseId);
  const inputHash = sha256Text(packText);
  const started = new Date();
  const startedMs = Date.now();

  const result = await generateText({
    model: request.model,
    system: systemPrompt,
    prompt: [
      'CASE_ID = ' + pack.case_id,
      'COMPANY = ' + pack.company,
      'SOURCE_RUN_ID = ' + pack.run_id,
      '',
      'FROZEN ANALYTICAL INPUT:',
      JSON.stringify(pack.analytical_input),
    ].join('\n'),
    output: Output.object({ schema: replayOutputSchema }),
    maxRetries: 0,
  });

  const output = result.output;
  const dimensionScores = Object.fromEntries(
    DIMENSION_KEYS.map((dimension) => [dimension, output.dimensions[dimension].score]),
  ) as Record<DimensionKey, number | null>;
  const dimensionRationales = Object.fromEntries(
    DIMENSION_KEYS.map((dimension) => [dimension, output.dimensions[dimension].rationale]),
  ) as Record<DimensionKey, string>;
  const i2 = computeI2(dimensionScores);
  const finished = new Date();
  const modelParts = parseModelId(request.model);

  const recordWithoutHash = {
    campaign_id: 'OROTITAN_V2_REPRODUCIBILITY_V0.1' as const,
    phase_id: 'PHASE_A1_SCORING_JUDGMENT_REPLAY' as const,
    case_id: pack.case_id,
    repetition_index: request.repetitionIndex,
    engine_fingerprint: '1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e',
    model_provider: modelParts.provider,
    model_name: modelParts.name,
    model_version: modelParts.version,
    reasoning_config: 'PROVIDER_DEFAULT' as const,
    prompt_artifact_version: 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1' as const,
    input_package_sha256: inputHash,
    started_at: started.toISOString(),
    finished_at: finished.toISOString(),
    duration_ms: Date.now() - startedMs,
    dimension_scores: dimensionScores,
    dimension_rationales: dimensionRationales,
    oqs_raw: i2.oqsRaw,
    weak_link_cap: i2.weakLinkCap,
    oqs: i2.oqs,
    i2_computation_status: i2.status,
    finish_reason: String(result.finishReason),
    usage: result.usage,
  };

  return {
    ...recordWithoutHash,
    output_sha256: sha256Text(JSON.stringify(recordWithoutHash)),
  };
}
