import type { MarketSyncRun } from './types';

export type SyncHealthStatus = 'NO_RUN' | 'INVALID_RUN' | 'SUCCESS' | 'PARTIAL' | 'FAILED';

export interface SyncErrorGroup {
  message: string;
  count: number;
  companies: string[];
}

export interface SyncSourceHealth {
  source: MarketSyncRun['trigger_source'];
  run: MarketSyncRun | null;
  status: SyncHealthStatus;
  successRatePct: number | null;
  durationSeconds: number | null;
  errors: SyncErrorGroup[];
}

function isCount(value: number): boolean {
  return Number.isSafeInteger(value) && value >= 0;
}

function inspectSource(runs: readonly MarketSyncRun[], source: MarketSyncRun['trigger_source']): SyncSourceHealth {
  // The append-only identity is monotonically increasing, independent of input order.
  const run = runs.filter(row => row.trigger_source === source)
    .reduce<MarketSyncRun | null>((latest, row) => latest === null || row.id > latest.id ? row : latest, null);

  if (!run) return { source, run: null, status: 'NO_RUN', successRatePct: null, durationSeconds: null, errors: [] };

  const entries = Array.isArray(run.results) ? run.results : null;
  const errorGroups = new Map<string, SyncErrorGroup>();
  let insertedItems = 0;
  let failedItems = 0;
  if (entries) {
    for (const value of entries) {
      if (!value || typeof value !== 'object' || Array.isArray(value)) continue;
      if (value.status === 'inserted') {
        insertedItems++;
      } else if (value.status === 'failed') {
        failedItems++;
        const message = typeof value.error === 'string' && value.error.trim()
          ? value.error.trim() : 'ERREUR_NON_DETAILLEE';
        const slug = typeof value.slug === 'string' && value.slug.trim() ? value.slug.trim() : null;
        const group = errorGroups.get(message) ?? { message, count: 0, companies: [] };
        group.count++;
        if (slug) group.companies.push(slug);
        errorGroups.set(message, group);
      }
    }
  }

  const start = Date.parse(run.started_at);
  const finish = Date.parse(run.finished_at);
  const validTimes = Number.isFinite(start) && Number.isFinite(finish) && finish >= start;
  const validTotals = isCount(run.companies) && run.companies > 0
    && isCount(run.inserted) && isCount(run.failed)
    && run.inserted + run.failed === run.companies;
  const validDetails = entries !== null && entries.length === run.companies
    && insertedItems === run.inserted && failedItems === run.failed;
  const valid = validTimes && validTotals && validDetails;

  return {
    source,
    run,
    status: !valid ? 'INVALID_RUN' : run.failed === 0 ? 'SUCCESS' : run.inserted === 0 ? 'FAILED' : 'PARTIAL',
    successRatePct: valid ? Math.round(run.inserted / run.companies * 1000) / 10 : null,
    durationSeconds: validTimes ? Math.round((finish - start) / 100) / 10 : null,
    errors: [...errorGroups.values()]
      .map(group => ({ ...group, companies: group.companies.sort() }))
      .sort((a, b) => b.count - a.count || a.message.localeCompare(b.message)),
  };
}

/** Read-only operational interpretation of the persisted sync journal. */
export function summarizeMarketSyncHealth(runs: readonly MarketSyncRun[]): {
  cron: SyncSourceHealth;
  admin: SyncSourceHealth;
} {
  return {
    cron: inspectSource(runs, 'CRON'),
    admin: inspectSource(runs, 'ADMIN'),
  };
}
