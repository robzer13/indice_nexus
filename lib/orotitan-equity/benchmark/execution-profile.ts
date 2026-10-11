import { createHash } from 'node:crypto';

export const BENCHMARK_CAMPAIGN_ID = 'OROTITAN_V2_REPRODUCIBILITY_CAMPAIGN_V0.2' as const;
export const BENCHMARK_PHASE_ID = 'PHASE_A1_SCORING_JUDGMENT_REPLAY' as const;
export const BENCHMARK_ENGINE_FINGERPRINT = '1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e' as const;
export const BENCHMARK_RUNNER_VERSION = 'OROTITAN_V2_SCORING_REPLAY_RUNNER_V0.2' as const;
export const BENCHMARK_PROMPT_VERSION = 'OROTITAN_V2_SCORING_REPLAY_PROMPT_V0.1' as const;
export const BENCHMARK_SANITIZER_VERSION = '1.0.0' as const;
export const BENCHMARK_MAX_OUTPUT_TOKENS = 6000 as const;

export const BENCHMARK_REASONING_LEVELS = [
  'none',
  'minimal',
  'low',
  'medium',
  'high',
  'xhigh',
] as const;

export type BenchmarkReasoningLevel = typeof BENCHMARK_REASONING_LEVELS[number];

export function isBenchmarkReasoningLevel(value: string): value is BenchmarkReasoningLevel {
  return (BENCHMARK_REASONING_LEVELS as readonly string[]).includes(value);
}

function sha256(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex');
}

export function benchmarkExecutionProfile(input: {
  model: string;
  reasoning: BenchmarkReasoningLevel;
}) {
  if (!input.model.includes('/')) throw new Error('Benchmark model must be a provider/model Gateway id.');
  const provider = input.model.slice(0, input.model.indexOf('/'));
  if (!provider) throw new Error('Benchmark model provider prefix is empty.');

  const profile = {
    campaign_id: BENCHMARK_CAMPAIGN_ID,
    phase_id: BENCHMARK_PHASE_ID,
    engine_fingerprint: BENCHMARK_ENGINE_FINGERPRINT,
    runner_version: BENCHMARK_RUNNER_VERSION,
    prompt_artifact_version: BENCHMARK_PROMPT_VERSION,
    sanitizer_version: BENCHMARK_SANITIZER_VERSION,
    model: input.model,
    reasoning_config: input.reasoning,
    gateway_provider_only: provider,
    max_output_tokens: BENCHMARK_MAX_OUTPUT_TOKENS,
  };

  return {
    ...profile,
    execution_profile_id: sha256(JSON.stringify(profile)),
  };
}
