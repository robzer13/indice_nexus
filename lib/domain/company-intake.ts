import type { CompanyRecord } from '@/lib/domain/types';
import type { LegacyCompanyQuoteLink } from '@/lib/domain/canonical-quote-identity';

export type PublishedDossierPointer = {
  dossier_id: string;
  issuer_id: string;
  current_snapshot_id: string;
};
export type PublishedSnapshotIdentity = {
  snapshot_id: string;
  dossier_id: string;
  issuer_id: string;
  security_id: string;
};

/** The tracking universe is NOT the published, scored research universe.
 * A security is published only if its *exact* mapped issuer+security+dossier
 * agrees with the active dossier's current canonical snapshot.
 * Never equate legacy company_id with issuer_id or match by ticker.
 */
export function unpublishedTrackedCompanies(
  companies: readonly CompanyRecord[],
  links: readonly LegacyCompanyQuoteLink[],
  pointers: readonly PublishedDossierPointer[],
  snapshots: readonly PublishedSnapshotIdentity[],
): CompanyRecord[] {
  const publishedTriples = new Set<string>();
  const snapshotById = new Map(snapshots.map((s) => [s.snapshot_id, s]));
  for (const dossier of pointers) {
    const snapshot = snapshotById.get(dossier.current_snapshot_id);
    if (!snapshot ||
      snapshot.dossier_id !== dossier.dossier_id ||
      snapshot.issuer_id !== dossier.issuer_id) continue;
    publishedTriples.add([snapshot.issuer_id, snapshot.security_id, snapshot.dossier_id].join('|'));
  }
  const publishedCompanyIds = new Set(links
    .filter((link) => publishedTriples.has(
      [link.issuer_id, link.security_id, link.dossier_id].join('|')))
    .map((link) => link.legacy_company_id));

  return companies
    .filter((company) => company.active && !publishedCompanyIds.has(company.id))
    .sort((a, b) => a.name.localeCompare(b.name));
}
