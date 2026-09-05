import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import {
  assertSnapshotDoesNotExist,
  DuplicateSnapshotError,
  type SnapshotInput,
} from '@/lib/domain/snapshot';

export async function createImmutableSnapshot(input: SnapshotInput): Promise<void> {
  const supabase = createServerSupabaseClient();
  const { data: existing, error: lookupError } = await supabase
    .from('snapshots')
    .select('id')
    .eq('company_id', input.company_id)
    .eq('analysis_date', input.analysis_date)
    .eq('model_version', input.model_version)
    .maybeSingle();
  if (lookupError) throw new Error(`Unable to verify snapshot uniqueness: ${lookupError.message}`);
  assertSnapshotDoesNotExist(Boolean(existing));

  const { error } = await supabase.from('snapshots').insert(input);
  if (error?.code === '23505') throw new DuplicateSnapshotError();
  if (error) throw new Error(`Unable to create snapshot: ${error.message}`);
}
