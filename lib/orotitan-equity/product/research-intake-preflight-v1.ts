/**
 * Read-only Pilotage intake preflight.
 *
 * This is a routing advisory, NOT an execution admission gate and NOT a GO.
 * The frozen contract requires explicit issuer confirmation, canonical-mode
 * selection, contract pinning, data cutoff, and a separate user execution GO.
 * A Research request is never used as a RUN_ID.
 */
import type { ResearchInboxRequest } from './local-research-inbox-v1';

export type ResearchSecurityCandidate = Readonly<{
  securityId: string;
  issuerId: string;
  ticker: string;
  exchange: string;
}>;

export type ResearchIdentitySnapshot = Readonly<{
  verified: boolean;
  securities: readonly ResearchSecurityCandidate[];
}>;

export type PreflightBlockerCode =
  | 'REGISTRY_LOOKUP_UNAVAILABLE'
  | 'ISSUER_NOT_FOUND'
  | 'ISSUER_AMBIGUOUS'
  | 'ISSUER_REVIEW_REQUIRED'
  | 'MODE_AND_BASELINE_UNRESOLVED'
  | 'DATA_CUTOFF_MISSING'
  | 'CONTRACT_PINS_MISSING'
  | 'EXECUTION_GO_MISSING';

export type ResearchPreflight = Readonly<{
  requestId: string;
  verdict: 'BLOCKED_NOT_AUTHORIZED';
  candidateCount: number | null;
  candidateIssuerId: string | null;
  blockerCodes: readonly PreflightBlockerCode[];
}>;

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** Never infer an issuer from a fuzzy company name or a ticker alone. */
export function assessResearchIntakePreflight(
  request: ResearchInboxRequest,
  snapshot: ResearchIdentitySnapshot,
): ResearchPreflight {
  if (!UUID.test(request.requestId) || !UUID.test(request.companyId) ||
    request.status !== 'REQUESTED_NOT_EXECUTED' || request.registryRunId !== null ||
    !request.ticker.trim() || !request.exchange.trim()) {
    throw new Error('PRODUCT_PREFLIGHT_INVALID_REQUEST');
  }

  const blockers: PreflightBlockerCode[] = [];
  let candidateCount: number | null = null;
  let candidateIssuerId: string | null = null;

  if (!snapshot.verified) {
    blockers.push('REGISTRY_LOOKUP_UNAVAILABLE');
  } else {
    const matches = snapshot.securities.filter((candidate) =>
      UUID.test(candidate.securityId) &&
      UUID.test(candidate.issuerId) &&
      candidate.ticker.trim().toUpperCase() === request.ticker.trim().toUpperCase() &&
      candidate.exchange.trim().toUpperCase() === request.exchange.trim().toUpperCase(),
    );
    candidateCount = matches.length;
    if (matches.length === 0) {
      blockers.push('ISSUER_NOT_FOUND');
    } else if (matches.length > 1) {
      blockers.push('ISSUER_AMBIGUOUS');
    } else {
      // Even one exact security match is a CANDIDATE, not human validation.
      candidateIssuerId = matches[0].issuerId;
      blockers.push('ISSUER_REVIEW_REQUIRED');
    }
  }

  blockers.push(
    'MODE_AND_BASELINE_UNRESOLVED',
    'DATA_CUTOFF_MISSING',
    'CONTRACT_PINS_MISSING',
    'EXECUTION_GO_MISSING',
  );

  return {
    requestId: request.requestId,
    verdict: 'BLOCKED_NOT_AUTHORIZED',
    candidateCount,
    candidateIssuerId,
    blockerCodes: blockers,
  };
}

export const PREFLIGHT_BLOCKER_TEXT: Readonly<Record<PreflightBlockerCode, string>> = {
  REGISTRY_LOOKUP_UNAVAILABLE: 'Correspondance Registry non vérifiée ; vérification indisponible.',
  ISSUER_NOT_FOUND: 'Aucune ligne de sécurité correspondante trouvée dans le Registry.',
  ISSUER_AMBIGUOUS: 'Plusieurs lignes correspondent : identité émetteur à clarifier.',
  ISSUER_REVIEW_REQUIRED: 'Une ligne candidate trouvée ; identité émetteur à confirmer humainement.',
  MODE_AND_BASELINE_UNRESOLVED: 'Mode ANALYZE/INITIAL ou REFRESH et dossier canonique à établir.',
  DATA_CUTOFF_MISSING: 'Date de coupure des données (DATA_CUTOFF) non fixée.',
  CONTRACT_PINS_MISSING: 'Versions des contrats et références des artefacts non vérifiées.',
  EXECUTION_GO_MISSING: 'Autorisation explicite GO <ENTREPRISE> non reçue.',
};
