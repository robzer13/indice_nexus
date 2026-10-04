import 'server-only';

import { createServerSupabaseClient } from '@/lib/supabase/server';
import type { ScoringReplayExecutionRecord } from '@/lib/orotitan-equity/benchmark/scoring-replay';

export interface BenchmarkExecutionLookup {
  campaignId: string;
  phaseId: string;
  caseId: string;
  repetitionIndex: number;
  executionProfileId: string;
}

export async function getExistingBenchmarkExecution(key: BenchmarkExecutionLookup) {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase
    .from('orotitan_engine_benchmark_executions')
    .select('*')
    .eq('campaign_id', key.campaignId)
    .eq('phase_id', key.phaseId)
    .eq('case_id', key.caseId)
    .eq('repetition_index', key.repetitionIndex)
    .eq('execution_profile_id', key.executionProfileId)
    .maybeSingle();

  if (error) throw new Error(`Unable to query benchmark execution registry: ${error.message}`);
  return data;
}

export async function recordBenchmarkExecution(record: ScoringReplayExecutionRecord) {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase.rpc('record_orotitan_benchmark_execution', {
    p_execution: record,
  });
  if (error) throw new Error(`Unable to persist benchmark execution: ${error.message}`);
  return data;
}
