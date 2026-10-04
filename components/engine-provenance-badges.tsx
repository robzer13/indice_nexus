import type { CompanyState } from '@/lib/domain/types';

type EngineStatus = NonNullable<CompanyState['engine_status']>;
type ResearchFreshnessStatus = NonNullable<CompanyState['research_freshness_status']>;

const ENGINE_STYLE: Record<EngineStatus, string> = {
  CURRENT: 'border-emerald-700/60 bg-emerald-950/30 text-emerald-200',
  PREVIOUS: 'border-amber-700/60 bg-amber-950/30 text-amber-200',
  LEGACY: 'border-rose-800/60 bg-rose-950/30 text-rose-200',
  UNKNOWN: 'border-slate-700 bg-slate-900 text-slate-400',
};

const ENGINE_LABEL: Record<EngineStatus, string> = {
  CURRENT: 'Moteur actuel',
  PREVIOUS: 'Moteur précédent',
  LEGACY: 'Moteur legacy',
  UNKNOWN: 'Moteur inconnu',
};

const FRESHNESS_STYLE: Record<ResearchFreshnessStatus, string> = {
  RECENT: 'border-cyan-800/60 bg-cyan-950/25 text-cyan-200',
  AGING: 'border-amber-800/60 bg-amber-950/25 text-amber-200',
  STALE: 'border-rose-800/60 bg-rose-950/25 text-rose-200',
  UNKNOWN: 'border-slate-700 bg-slate-900 text-slate-400',
};

const FRESHNESS_LABEL: Record<ResearchFreshnessStatus, string> = {
  RECENT: 'Analyse récente',
  AGING: 'Analyse vieillissante',
  STALE: 'Analyse à rafraîchir',
  UNKNOWN: 'Fraîcheur inconnue',
};

export function EngineBadge({
  status,
  shortFingerprint,
}: {
  status: EngineStatus;
  shortFingerprint?: string | null;
}) {
  return (
    <span
      title={shortFingerprint ? `Fingerprint moteur : ${shortFingerprint}…` : undefined}
      className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium ${ENGINE_STYLE[status]}`}
    >
      {ENGINE_LABEL[status]}
    </span>
  );
}

export function ResearchFreshnessBadge({
  status,
  ageDays,
}: {
  status: ResearchFreshnessStatus;
  ageDays?: number | null;
}) {
  const suffix = ageDays === null || ageDays === undefined ? '' : ` · ${ageDays} j`;
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-medium ${FRESHNESS_STYLE[status]}`}>
      {FRESHNESS_LABEL[status]}{suffix}
    </span>
  );
}
