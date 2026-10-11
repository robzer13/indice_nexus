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

/** Permanently claim the exact paid inference key; uncertain claims fail closed. */
export async function claimBenchmarkExecution(key: BenchmarkExecutionLookup): Promise<'CLAIMED'|'IN_PROGRESS'|'EXISTING'> {
  const supabase = createServerSupabaseClient();
  const { data, error } = await supabase.rpc('claim_orotitan_benchmark_execution', {
    p_campaign_id: key.campaignId,
    p_phase_id: key.phaseId,
    p_case_id: key.caseId,
    p_repetition_index: key.repetitionIndex,
    p_execution_profile_id: key.executionProfileId,
  });
  if (error) throw new Error('Benchmark claim RPC failed: '+error.message);
  const status = (data as {status?: unknown} | null)?.status;
  if (status !== 'CLAIMED' && status !== 'IN_PROGRESS' && status !== 'EXISTING') {
    throw new Error('Invalid claim receipt');
  }
  return status;
}
