/**
 * Pre-run routing card for the Research Intake pilot.
 *
 * Read-only, fail-closed proposal. NOT a persisted Run, an issuer
 * certification, a contract-pin manifest, or an execution authorization.
 * Frozen execution authority: Pilotage v1.0.1 §§4–10 and Registry v1.0.
 */
import type { ResearchInboxRequest } from './local-research-inbox-v1';
import type { ResearchIdentitySnapshot } from './research-intake-preflight-v1';

export type RegistryIssuer = Readonly<{
  issuerId: string; displayName: string;
  legalName: string | null; country: string | null;
  reportingCurrency: string | null;
}>;
export type RegistryDossier = Readonly<{
  dossierId: string; issuerId: string; active: boolean;
  currentSnapshotId: string | null;
}>;
export type RegistryCanonicalSnapshot = Readonly<{
  snapshotId: string; issuerId: string; dossierId: string;
}>;
export type RegistryRunReference = Readonly<{
  runId: string; issuerId: string; runStatus: string;
}>;
export type ResearchRegistryRoutingSnapshot = Readonly<{
  verified: boolean;
  issuers: readonly RegistryIssuer[];
  dossiers: readonly RegistryDossier[];
  snapshots: readonly RegistryCanonicalSnapshot[];
  runs: readonly RegistryRunReference[];
}>;

export type PreRunCard = Readonly<{
  requestId: string;
  displayedCompany: string;
  entryPath: 'IMPOSED_COMPANY';
  stageSequence: readonly ['RESEARCH', 'DEEP_DIVE', 'INTEGRATION'];
  candidateIssuerId: string | null;
  candidateSecurityId: string | null;
  issuerDisplayName: string | null;
  legalIssuerName: string | null;
  country: string | null;
  reportingCurrency: string | null;
  candidateIdentityConfirmed: false;
  canonicalModeProposal: 'ANALYZE' | 'REFRESH' | null;
  runTypeProposal: 'INITIAL' | 'REFRESH' | null;
  existingCanonicalDossierId: string | null;
  baselineSnapshotId: string | null;
  openRunCount: number | null;
  cutoffPolicy: 'ON_RUN_CREATION_OR_EXPLICIT_HISTORICAL_REQUEST';
  cutoffValue: null;
  contractPinsVerified: false;
  executionGoReceived: false;
  publicationGoReceived: false;
  executionAdmitted: false;
  blockingReasons: readonly string[];
}>;
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const VALID_RUN_STATUS = new Set([
  'CREATED','ACTIVE','PAUSED','BLOCKED','READY_TO_PUBLISH','PUBLISHED','CANCELLED',
]);
const TERMINAL = new Set(['PUBLISHED','CANCELLED']);

export function buildResearchPreRunCard(
  request: ResearchInboxRequest,
  identity: ResearchIdentitySnapshot,
  registry: ResearchRegistryRoutingSnapshot,
): PreRunCard {
  if (!UUID.test(request.requestId) || !UUID.test(request.companyId) ||
    !request.companyName.trim() || !request.ticker.trim() || !request.exchange.trim() ||
    request.status !== 'REQUESTED_NOT_EXECUTED' || request.registryRunId !== null) {
    throw new Error('PRODUCT_PRE_RUN_INVALID_RECEIPT');
  }

  const reasons: string[] = [];
  let securityId: string | null = null;
  let issuerId: string | null = null;
  let issuerDisplayName: string | null = null;
  let legalName: string | null = null;
  let country: string | null = null;
  let currency: string | null = null;
  let mode: PreRunCard['canonicalModeProposal'] = null;
  let runType: PreRunCard['runTypeProposal'] = null;
  let dossierId: string | null = null;
  let baselineSnapshotId: string | null = null;
  let openRunCount: number | null = null;

  const matches = identity.verified ? identity.securities.filter(s =>
    UUID.test(s.issuerId) && UUID.test(s.securityId) &&
    s.ticker.trim().toUpperCase() === request.ticker.trim().toUpperCase() &&
    s.exchange.trim().toUpperCase() === request.exchange.trim().toUpperCase()
  ) : [];

  if (!identity.verified) reasons.push('REGISTRY_SECURITY_LOOKUP_UNAVAILABLE');
  else if (matches.length === 0) reasons.push('NO_EXACT_SECURITY_MATCH');
  else if (matches.length > 1) reasons.push('AMBIGUOUS_SECURITY_IDENTITY');

  if (!registry.verified) {
    reasons.push('REGISTRY_CANONICAL_STATE_UNVERIFIED');
  } else if (identity.verified && matches.length === 1) {
    const security = matches[0];
    const issuers = registry.issuers.filter(i => i.issuerId === security.issuerId);
    if (issuers.length !== 1 || !UUID.test(security.issuerId) || !UUID.test(security.securityId)) {
      reasons.push('ISSUER_REGISTRY_MISSING_OR_AMBIGUOUS');
    } else {
      const issuer = issuers[0];
      issuerId = issuer.issuerId;
      securityId = security.securityId;
      issuerDisplayName = issuer.displayName.trim() || null;
      legalName = issuer.legalName?.trim() || null;
      country = issuer.country?.trim() || null;
      currency = issuer.reportingCurrency?.trim() || null;

      if (!legalName || !country || !issuerDisplayName) {
        reasons.push('LEGAL_ISSUER_IDENTITY_INCOMPLETE');
      }
      // Registry linkage and names are evidence of a candidate, NOT final
      // issuer/security identity approval (one human validation still required).
      reasons.push('ISSUER_HUMAN_CONFIRMATION_REQUIRED');

      const dossiers = registry.dossiers.filter(d => d.issuerId === issuerId && d.active);
      if (dossiers.length > 1) {
        reasons.push('MULTIPLE_ACTIVE_CANONICAL_DOSSIERS');
      } else if (dossiers.length === 0) {
        // A historical/inactive dossier is NOT equivalent to no dossier.
        if (registry.dossiers.some(d => d.issuerId === issuerId)) {
          reasons.push('HISTORICAL_DOSSIER_STATE_NEEDS_REVIEW');
        } else {
          // Proposal only. Alias and issuer identity still require confirmation.
          mode = 'ANALYZE';
          runType = 'INITIAL';
        }
      } else {
        const dossier = dossiers[0];
        const baselines = registry.snapshots.filter(s =>
          s.snapshotId === dossier.currentSnapshotId &&
          s.issuerId === issuerId &&
          s.dossierId === dossier.dossierId
        );
        if (!UUID.test(dossier.dossierId) || !dossier.currentSnapshotId ||
          !UUID.test(dossier.currentSnapshotId) || baselines.length !== 1) {
          reasons.push('CANONICAL_BASELINE_INVALID_OR_MISSING');
        } else {
          dossierId = dossier.dossierId;
          baselineSnapshotId = dossier.currentSnapshotId;
          mode = 'REFRESH';
          runType = 'REFRESH';
        }
      }

      const relevantRuns = registry.runs.filter(run => run.issuerId === issuerId);
      if (relevantRuns.some(run => !UUID.test(run.runId) ||
        !VALID_RUN_STATUS.has(run.runStatus))) {
        reasons.push('RUN_REGISTRY_STATE_INVALID');
      } else {
        openRunCount = relevantRuns.filter(run => !TERMINAL.has(run.runStatus)).length;
        if (openRunCount > 0) reasons.push('EXISTING_NONTERMINAL_RUN_REVIEW');
      }
    }
  }

  if (!mode) reasons.push('CANONICAL_MODE_UNRESOLVED');
  reasons.push('DATA_CUTOFF_NOT_ESTABLISHED');
  reasons.push('IMMUTABLE_CONTRACT_PINS_NOT_VERIFIED');
  reasons.push('EXPLICIT_EXECUTION_GO_NOT_RECEIVED');

  return {
    requestId: request.requestId,
    displayedCompany: request.companyName,
    entryPath: 'IMPOSED_COMPANY',
    stageSequence: ['RESEARCH','DEEP_DIVE','INTEGRATION'],
    candidateIssuerId: issuerId,
    candidateSecurityId: securityId,
    issuerDisplayName,
    legalIssuerName: legalName,
    country,
    reportingCurrency: currency,
    candidateIdentityConfirmed: false,
    canonicalModeProposal: mode,
    runTypeProposal: runType,
    existingCanonicalDossierId: dossierId,
    baselineSnapshotId,
    openRunCount,
    cutoffPolicy: 'ON_RUN_CREATION_OR_EXPLICIT_HISTORICAL_REQUEST',
    cutoffValue: null,
    contractPinsVerified: false,
    executionGoReceived: false,
    publicationGoReceived: false,
    executionAdmitted: false,
    blockingReasons: reasons,
  };
}

export const PRE_RUN_REASON_LABELS: Readonly<Record<string, string>> = {
  REGISTRY_SECURITY_LOOKUP_UNAVAILABLE: 'Liste des titres Registry indisponible ou incomplète.',
  NO_EXACT_SECURITY_MATCH: 'Aucun titre Registry ne correspond exactement au ticker et à la place.',
  AMBIGUOUS_SECURITY_IDENTITY: 'Plusieurs titres candidats ; identification manuelle nécessaire.',
  REGISTRY_CANONICAL_STATE_UNVERIFIED: 'Émetteurs, dossiers et snapshots Registry non vérifiés.',
  ISSUER_REGISTRY_MISSING_OR_AMBIGUOUS: 'Émetteur Registry introuvable ou ambigu.',
  LEGAL_ISSUER_IDENTITY_INCOMPLETE: 'Nom légal, pays ou identité de l’émetteur incomplets.',
  ISSUER_HUMAN_CONFIRMATION_REQUIRED: 'Confirmer explicitement émetteur et titre avant tout GO.',
  MULTIPLE_ACTIVE_CANONICAL_DOSSIERS: 'Plusieurs dossiers actifs ; ne pas choisir automatiquement.',
  HISTORICAL_DOSSIER_STATE_NEEDS_REVIEW: 'Dossier historique inactif présent ; choix INITIAL ou REFRESH non autorisé.',
  CANONICAL_BASELINE_INVALID_OR_MISSING: 'Snapshot canonique de référence absent ou incohérent.',
  RUN_REGISTRY_STATE_INVALID: 'Statut ou identité d’un run existant invalide.',
  EXISTING_NONTERMINAL_RUN_REVIEW: 'Un run non terminal existe ; vérifier la reprise avant création.',
  CANONICAL_MODE_UNRESOLVED: 'Mode ANALYZE / INITIAL ou REFRESH non déterminé.',
  DATA_CUTOFF_NOT_ESTABLISHED: 'DATA_CUTOFF fixé seulement au GO ou selon une demande historique explicite.',
  IMMUTABLE_CONTRACT_PINS_NOT_VERIFIED: 'Pins contractuels immuables et hashes non vérifiés.',
  EXPLICIT_EXECUTION_GO_NOT_RECEIVED: 'GO <ENTREPRISE> distinct non reçu. Aucun RUN_ID.',
};
