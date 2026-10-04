import { timingSafeEqual } from 'node:crypto';
import { NextResponse } from 'next/server';
import { z } from 'zod';
import {
  benchmarkExecutionProfile,
  BENCHMARK_CAMPAIGN_ID,
  BENCHMARK_PHASE_ID,
  isBenchmarkReasoningLevel,
} from '@/lib/orotitan-equity/benchmark/execution-profile';
import {
  getExistingBenchmarkExecution,
  recordBenchmarkExecution,
} from '@/lib/orotitan-equity/benchmark/persistence';
import { runScoringReplay } from '@/lib/orotitan-equity/benchmark/scoring-replay';

export const runtime = 'nodejs';
export const maxDuration = 300;

const bodySchema = z.object({
  caseId: z.string().regex(/^V2REF-\d{3}$/),
  repetitionIndex: z.number().int().min(1).max(1000),
}).strict();

function unauthorized() {
  return NextResponse.json({ error: 'UNAUTHORIZED' }, { status: 401 });
}

function secretMatches(actual: string | null, expected: string): boolean {
  if (!actual?.startsWith('Bearer ')) return false;
  const supplied = Buffer.from(actual.slice('Bearer '.length), 'utf8');
  const target = Buffer.from(expected, 'utf8');
  return supplied.length === target.length && timingSafeEqual(supplied, target);
}

export async function POST(request: Request) {
  if (process.env.OROTITAN_BENCHMARK_RUNNER_ENABLED !== 'true') {
    return NextResponse.json({ error: 'BENCHMARK_RUNNER_DISABLED' }, { status: 503 });
  }

  const secret = process.env.BENCHMARK_RUNNER_SECRET;
  if (!secret) return NextResponse.json({ error: 'BENCHMARK_RUNNER_SECRET_MISSING' }, { status: 503 });
  if (!secretMatches(request.headers.get('authorization'), secret)) return unauthorized();

  const model = process.env.OROTITAN_BENCHMARK_MODEL;
  if (!model) return NextResponse.json({ error: 'OROTITAN_BENCHMARK_MODEL_MISSING' }, { status: 503 });

  const reasoningValue = process.env.OROTITAN_BENCHMARK_REASONING;
  if (!reasoningValue || !isBenchmarkReasoningLevel(reasoningValue)) {
    return NextResponse.json({ error: 'OROTITAN_BENCHMARK_REASONING_INVALID' }, { status: 503 });
  }

  const parsed = bodySchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: 'INVALID_REQUEST', details: parsed.error.flatten() }, { status: 400 });
  }

  try {
    const profile = benchmarkExecutionProfile({ model, reasoning: reasoningValue });
    const lookup = {
      campaignId: BENCHMARK_CAMPAIGN_ID,
      phaseId: BENCHMARK_PHASE_ID,
      caseId: parsed.data.caseId,
      repetitionIndex: parsed.data.repetitionIndex,
      executionProfileId: profile.execution_profile_id,
    };

    const existing = await getExistingBenchmarkExecution(lookup);
    if (existing) {
      return NextResponse.json({
        status: 'EXISTING',
        execution: existing,
      });
    }

    const execution = await runScoringReplay({
      caseId: parsed.data.caseId,
      repetitionIndex: parsed.data.repetitionIndex,
      model,
      reasoning: reasoningValue,
    });

    if (execution.execution_profile_id !== profile.execution_profile_id) {
      throw new Error('Benchmark execution profile mismatch.');
    }

    const persistence = await recordBenchmarkExecution(execution);
    return NextResponse.json({
      status: 'RECORDED',
      persistence,
      execution,
    });
  } catch (error) {
    return NextResponse.json({
      error: 'BENCHMARK_EXECUTION_FAILED',
      message: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
