import { NextResponse } from 'next/server';
import { z } from 'zod';
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

export async function POST(request: Request) {
  if (process.env.OROTITAN_BENCHMARK_RUNNER_ENABLED !== 'true') {
    return NextResponse.json({ error: 'BENCHMARK_RUNNER_DISABLED' }, { status: 503 });
  }

  const secret = process.env.BENCHMARK_RUNNER_SECRET;
  if (!secret) return NextResponse.json({ error: 'BENCHMARK_RUNNER_SECRET_MISSING' }, { status: 503 });

  const auth = request.headers.get('authorization');
  if (auth !== 'Bearer ' + secret) return unauthorized();

  const model = process.env.OROTITAN_BENCHMARK_MODEL;
  if (!model) return NextResponse.json({ error: 'OROTITAN_BENCHMARK_MODEL_MISSING' }, { status: 503 });

  const parsed = bodySchema.safeParse(await request.json().catch(() => null));
  if (!parsed.success) {
    return NextResponse.json({ error: 'INVALID_REQUEST', details: parsed.error.flatten() }, { status: 400 });
  }

  try {
    const result = await runScoringReplay({
      caseId: parsed.data.caseId,
      repetitionIndex: parsed.data.repetitionIndex,
      model,
    });
    return NextResponse.json(result);
  } catch (error) {
    return NextResponse.json({
      error: 'BENCHMARK_EXECUTION_FAILED',
      message: error instanceof Error ? error.message : 'Unknown error',
    }, { status: 500 });
  }
}
