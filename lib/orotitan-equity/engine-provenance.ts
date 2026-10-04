export const CURRENT_ENGINE = {
  generation: 'V2_CURRENT',
  processVersion: '2.0',
  pilotageContractVersion: '2.0',
  contractSetSha256: '1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e',
} as const;

export type EngineStatus = 'CURRENT' | 'PREVIOUS' | 'LEGACY' | 'UNKNOWN';
export type ResearchFreshnessStatus = 'RECENT' | 'AGING' | 'STALE' | 'UNKNOWN';

export interface EngineProvenanceInput {
  processVersion: string | null;
  pilotageContractVersion: string | null;
  contractSetSha256: string | null;
}

export interface EngineProvenance {
  status: EngineStatus;
  generation: string;
  fingerprint: string | null;
  shortFingerprint: string | null;
  processVersion: string | null;
  pilotageContractVersion: string | null;
}

function majorVersion(value: string | null): number | null {
  if (!value) return null;
  const match = value.match(/^(\d+)/);
  return match ? Number(match[1]) : null;
}

export function classifyEngineProvenance(input: EngineProvenanceInput): EngineProvenance {
  const fingerprint = input.contractSetSha256;
  const shortFingerprint = fingerprint ? fingerprint.slice(0, 12) : null;

  if (
    fingerprint === CURRENT_ENGINE.contractSetSha256 &&
    input.processVersion === CURRENT_ENGINE.processVersion &&
    input.pilotageContractVersion === CURRENT_ENGINE.pilotageContractVersion
  ) {
    return {
      status: 'CURRENT',
      generation: CURRENT_ENGINE.generation,
      fingerprint,
      shortFingerprint,
      processVersion: input.processVersion,
      pilotageContractVersion: input.pilotageContractVersion,
    };
  }

  const processMajor = majorVersion(input.processVersion);
  if (processMajor !== null && processMajor < majorVersion(CURRENT_ENGINE.processVersion)!) {
    return {
      status: 'LEGACY',
      generation: `V${processMajor}_LEGACY`,
      fingerprint,
      shortFingerprint,
      processVersion: input.processVersion,
      pilotageContractVersion: input.pilotageContractVersion,
    };
  }

  if (input.processVersion || input.pilotageContractVersion || fingerprint) {
    return {
      status: 'PREVIOUS',
      generation: input.processVersion ? `V${input.processVersion}_PREVIOUS` : 'PREVIOUS_ENGINE',
      fingerprint,
      shortFingerprint,
      processVersion: input.processVersion,
      pilotageContractVersion: input.pilotageContractVersion,
    };
  }

  return {
    status: 'UNKNOWN',
    generation: 'UNKNOWN_ENGINE',
    fingerprint: null,
    shortFingerprint: null,
    processVersion: null,
    pilotageContractVersion: null,
  };
}

export const RESEARCH_FRESHNESS_THRESHOLDS_DAYS = {
  recentMax: 90,
  agingMax: 180,
} as const;

/**
 * Operational freshness only. This never changes a score or certification state.
 * RECENT <= 90d, AGING 91-180d, STALE > 180d.
 */
export function classifyResearchFreshness(
  dataCutoff: string | null,
  now: Date = new Date(),
): { status: ResearchFreshnessStatus; ageDays: number | null } {
  if (!dataCutoff) return { status: 'UNKNOWN', ageDays: null };
  const cutoff = Date.parse(`${dataCutoff}T00:00:00Z`);
  if (!Number.isFinite(cutoff)) return { status: 'UNKNOWN', ageDays: null };

  const ageDays = Math.max(0, Math.floor((now.getTime() - cutoff) / 86_400_000));
  if (ageDays <= RESEARCH_FRESHNESS_THRESHOLDS_DAYS.recentMax) return { status: 'RECENT', ageDays };
  if (ageDays <= RESEARCH_FRESHNESS_THRESHOLDS_DAYS.agingMax) return { status: 'AGING', ageDays };
  return { status: 'STALE', ageDays };
}

export function engineStatusLabel(status: EngineStatus): string {
  if (status === 'CURRENT') return 'Moteur actuel';
  if (status === 'PREVIOUS') return 'Moteur précédent';
  if (status === 'LEGACY') return 'Moteur legacy';
  return 'Moteur inconnu';
}

export function researchFreshnessLabel(status: ResearchFreshnessStatus): string {
  if (status === 'RECENT') return 'Analyse récente';
  if (status === 'AGING') return 'Analyse vieillissante';
  if (status === 'STALE') return 'Analyse à rafraîchir';
  return 'Fraîcheur inconnue';
}
