import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { getAllCompanies } from '@/lib/data/company-admin';
import {
  unpublishedTrackedCompanies,
  type PublishedDossierPointer,
  type PublishedSnapshotIdentity,
} from '@/lib/domain/company-intake';
import type { LegacyCompanyQuoteLink } from '@/lib/domain/canonical-quote-identity';
import type { CompanyRecord } from '@/lib/domain/types';

/** Admin-only intake; never use legacy rows as published investment research. */
export async function getUnpublishedTrackedCompanies(): Promise<CompanyRecord[]> {
  const supabase = createServerSupabaseClient();
  const [companies, pointersResult, linksResult] = await Promise.all([
    getAllCompanies(),
    supabase.from('research_dossiers')
      .select('dossier_id,issuer_id,current_snapshot_id')
      .eq('active', true)
      .not('current_snapshot_id', 'is', null),
    supabase.from('legacy_company_identity_map')
      .select('legacy_company_id,issuer_id,security_id,dossier_id'),
  ]);
  if (pointersResult.error)
    throw new Error(`Unable to load intake publication pointers: ${pointersResult.error.message}`);
  if (linksResult.error)
    throw new Error(`Unable to load intake canonical links: ${linksResult.error.message}`);

  const pointers = (pointersResult.data ?? []) as PublishedDossierPointer[];
  const links = (linksResult.data ?? []) as LegacyCompanyQuoteLink[];
  if (pointers.length === 0) {
    // Valid bootstrap state: nothing is published.
    return unpublishedTrackedCompanies(companies, links, [], []);
  }

  const snapshotIds = pointers.map((d) => d.current_snapshot_id);
  const { data: snapshots, error } = await supabase.from('research_snapshots')
    .select('snapshot_id,dossier_id,issuer_id,security_id')
    .in('snapshot_id', snapshotIds);
  if (error) throw new Error(`Unable to resolve intake published snapshots: ${error.message}`);
  if (!snapshots || snapshots.length !== snapshotIds.length)
    throw new Error('INTAKE_PUBLICATION_SNAPSHOT_COHORT_INCOMPLETE');
  return unpublishedTrackedCompanies(
    companies, links, pointers, snapshots as PublishedSnapshotIdentity[],
  );
}
