/**
 * Resolves a historical price owner exclusively through the audited
 * issuer + current security + dossier crosswalk. No UUID equality shortcut,
 * ticker search, or fallback to a different listing is permitted.
 */
export type CurrentCanonicalQuoteIdentity = {
  issuer_id: string;
  security_id: string;
  dossier_id: string;
};
export type LegacyCompanyQuoteLink = CurrentCanonicalQuoteIdentity & {
  legacy_company_id: string;
};
export function resolveCanonicalQuoteCompanyId(
  current: CurrentCanonicalQuoteIdentity,
  links: readonly LegacyCompanyQuoteLink[],
): string | null {
  if (!current.issuer_id || !current.security_id || !current.dossier_id) return null;
  const matches = links.filter(link =>
    link.issuer_id === current.issuer_id &&
    link.security_id === current.security_id &&
    link.dossier_id === current.dossier_id &&
    Boolean(link.legacy_company_id),
  );
  // Unlinked or ambiguous means no price; do not make up identity.
  return matches.length === 1 ? matches[0].legacy_company_id : null;
}
