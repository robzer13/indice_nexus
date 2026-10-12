/**
 * This is an additional fail-closed safeguard for the single-case V2 smoke test.
 * It does not approve spending or enable the runner. Full campaigns require
 * separately authorized code and deployment changes.
 */
export const BENCHMARK_SMOKE_MODE = 'SMOKE_ONLY' as const;
export const BENCHMARK_SMOKE_CASE_ID = 'V2REF-010' as const;
export const BENCHMARK_SMOKE_REPETITION_INDEX = 1 as const;
export const BENCHMARK_SMOKE_MODEL = 'openai/gpt-5.4' as const;
export const BENCHMARK_SMOKE_REASONING = 'medium' as const;

export interface BenchmarkSmokeRequest {
  mode: string | undefined;
  model: string | undefined;
  reasoning: string | undefined;
  caseId: string;
  repetitionIndex: number;
}

export type BenchmarkSmokeRejection =
  | 'BENCHMARK_SMOKE_ONLY_MODE_REQUIRED'
  | 'BENCHMARK_SMOKE_PROFILE_MISMATCH'
  | 'BENCHMARK_SMOKE_CASE_NOT_ALLOWED';

export function validateBenchmarkSmokeGate(
  input: BenchmarkSmokeRequest,
): BenchmarkSmokeRejection | null {
  if (input.mode !== BENCHMARK_SMOKE_MODE) {
    return 'BENCHMARK_SMOKE_ONLY_MODE_REQUIRED';
  }
  if (input.model !== BENCHMARK_SMOKE_MODEL || input.reasoning !== BENCHMARK_SMOKE_REASONING) {
    return 'BENCHMARK_SMOKE_PROFILE_MISMATCH';
  }
  if (input.caseId !== BENCHMARK_SMOKE_CASE_ID || input.repetitionIndex !== BENCHMARK_SMOKE_REPETITION_INDEX) {
    return 'BENCHMARK_SMOKE_CASE_NOT_ALLOWED';
  }
  return null;
}
