import Link from 'next/link';
import type { ReactNode } from 'react';
import {
  blockerTitle,
  buildRunHref,
  deriveStageStates,
  formatCutoff,
  lifecycleLabel,
  runStatusLabel,
  shortId,
  stageLabel,
} from '@/lib/orotitan-ui/presentation';
import type {
  ArtifactMeta,
  ArtifactRef,
  Blocker,
  CompanyIdentity,
  LoadResult,
  RunSummary,
  StageCode,
  StageLifecycle,
} from '@/lib/orotitan-ui/types';

function StatusDot({ lifecycle }: { lifecycle: StageLifecycle }) {
  const classes: Record<StageLifecycle, string> = {
    NOT_STARTED: 'border-slate-700 bg-slate-950',
    IN_PROGRESS: 'border-amber-400 bg-amber-400',
    PAUSED: 'border-amber-600 bg-amber-950',
    BLOCKED: 'border-rose-400 bg-rose-400',
    COMPLETE: 'border-emerald-400 bg-emerald-400',
  };
  return <span className={'inline-block h-2.5 w-2.5 rounded-full border ' + classes[lifecycle]} />;
}

export function RunStatusBadge({ status }: { status: string | null }) {
  const classes =
    status === 'BLOCKED'
      ? 'text-rose-300'
      : status === 'PUBLISHED'
        ? 'text-emerald-300'
        : status === 'ACTIVE'
          ? 'text-amber-300'
          : 'text-slate-400';
  return <span className={'text-xs font-semibold uppercase tracking-[0.14em] ' + classes}>{runStatusLabel(status)}</span>;
}

export function CompanyHeader({
  identity,
  load,
}: {
  identity: CompanyIdentity;
  load: LoadResult;
}) {
  return (
    <header className="border-b border-slate-800 pb-5">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-white sm:text-3xl">{identity.legalName}</h1>
          <p className="mt-1.5 text-sm text-slate-500">{identity.ticker} · {identity.marketDataSymbol}</p>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <span className="text-xs text-slate-600">Données {formatCutoff(load.data_cutoff)}</span>
          <RunStatusBadge status={load.run_status} />
        </div>
      </div>
    </header>
  );
}

export function StageProgress({ load }: { load: LoadResult }) {
  const states = deriveStageStates(load.current_stage, load.stage?.lifecycle_status ?? null);

  return (
    <section aria-label="Progression de l'analyse" className="border-b border-slate-800 py-5">
      <div className="grid gap-3 md:grid-cols-3 md:gap-0">
        {states.map(({ stage, lifecycle }, index) => (
          <div key={stage} className="relative flex items-start gap-3 md:pr-5">
            <div className="mt-1 flex shrink-0 items-center">
              <StatusDot lifecycle={lifecycle} />
              {index < states.length - 1 ? <span className="ml-2 hidden h-px w-8 bg-slate-800 md:block" /> : null}
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.13em] text-slate-300">{stageLabel(stage)}</div>
              <div className={'mt-1 text-xs ' + (lifecycle === 'BLOCKED' ? 'text-rose-300' : lifecycle === 'COMPLETE' ? 'text-emerald-400' : 'text-slate-600')}>
                {lifecycleLabel(lifecycle)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function blockerPresentation(blocker: Blocker): { summary?: string; resolution?: string } {
  if (blocker.code === 'ECONOMIC_SHARE_COUNT_UNRESOLVED') {
    return {
      summary: "La valorisation par action reste bloquée tant que le nombre économique d'actions au 22 septembre 2026 n'est pas démontré de manière conforme.",
      resolution: "Fermer toutes les classes de mouvements susceptibles de modifier le dénominateur jusqu'au 22/09/2026 avec une preuve exacte ou une borne rigoureuse.",
    };
  }
  return {};
}

export function BlockerPanel({ blockers }: { blockers: Blocker[] }) {
  if (blockers.length === 0) {
    return <section className="border-b border-slate-800 py-6 text-sm text-emerald-300">Aucun problème actif.</section>;
  }

  return (
    <section className="border-b border-slate-800 py-6">
      {blockers.map((blocker) => {
        const presentation = blockerPresentation(blocker);
        return (
          <article key={blocker.code} className="border-l-2 border-rose-500 pl-5">
            <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-rose-400">Problème actif</div>
            <h2 className="mt-2 text-xl font-semibold text-white">{blockerTitle(blocker)}</h2>
            <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
              {presentation.summary ?? blocker.detail ?? 'Ce problème bloque la poursuite normale de l’analyse.'}
            </p>

            <div className="mt-5 max-w-3xl">
              <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-600">À résoudre</div>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                {presentation.resolution ?? blocker.resolution_required ?? 'Résoudre le problème selon les exigences du stage courant.'}
              </p>
            </div>

            <details className="mt-5 max-w-4xl border-t border-slate-800 pt-3">
              <summary className="cursor-pointer text-xs text-slate-600 hover:text-slate-400">Détails techniques</summary>
              <dl className="mt-3 grid gap-x-6 gap-y-3 text-xs sm:grid-cols-2">
                <TechRow label="Code" value={blocker.code} />
                {blocker.classification ? <TechRow label="Classification" value={blocker.classification} /> : null}
                {blocker.scope ? <TechRow label="Portée" value={blocker.scope} /> : null}
                {blocker.diagnostic ? <TechRow label="Diagnostic" value={blocker.diagnostic} /> : null}
              </dl>
            </details>
          </article>
        );
      })}
    </section>
  );
}

function artifactName(ref: ArtifactRef, catalog: Record<string, ArtifactMeta>): string {
  return catalog[ref.artifact_id]?.logicalName ?? ref.artifact_id;
}

function authorityLabel(metadata: ArtifactMeta | undefined): string {
  if (!metadata) return '—';
  return metadata.authorityState === 'AUTHORITATIVE' ? 'Autoritatif' : 'Checkpoint';
}

export function ArtifactList({
  refs,
  catalog,
  compact = false,
}: {
  refs: ArtifactRef[];
  catalog: Record<string, ArtifactMeta>;
  compact?: boolean;
}) {
  if (refs.length === 0) {
    return <div className="border-y border-slate-800 py-5 text-sm text-slate-600">Aucun document dans ce périmètre.</div>;
  }

  return (
    <div className="border-t border-slate-800">
      <div className="hidden grid-cols-[minmax(0,1fr)_120px_120px_60px] gap-4 border-b border-slate-800 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-700 sm:grid">
        <div>Document</div>
        <div>Étape</div>
        <div>Autorité</div>
        <div>Version</div>
      </div>
      {refs.map((ref) => {
        const metadata = catalog[ref.artifact_id];
        return (
          <details key={ref.artifact_id + ':' + ref.version} className="group border-b border-slate-800/90">
            <summary className="grid cursor-pointer list-none gap-2 py-3.5 text-sm transition hover:bg-slate-900/35 sm:grid-cols-[minmax(0,1fr)_120px_120px_60px] sm:items-center sm:gap-4">
              <div className="min-w-0">
                <div className="truncate text-slate-200 group-open:text-white">{artifactName(ref, catalog)}</div>
                {!compact ? <div className="mt-1 truncate font-mono text-[10px] text-slate-700">{metadata?.artifactType ?? 'ARTIFACT'}</div> : null}
              </div>
              <div className="text-xs text-slate-500">{metadata ? stageLabel(metadata.stageCode) : '—'}</div>
              <div className={'text-xs ' + (metadata?.authorityState === 'AUTHORITATIVE' ? 'text-emerald-400' : 'text-amber-300')}>
                {authorityLabel(metadata)}
              </div>
              <div className="font-mono text-xs text-slate-600">v{ref.version}</div>
            </summary>
            <div className="grid gap-3 bg-slate-950/35 px-3 py-4 text-xs sm:grid-cols-2">
              <TechRow label="Artifact ID" value={ref.artifact_id} />
              <TechRow label="SHA-256" value={ref.content_sha256 ?? '—'} />
              <TechRow label="Authority class" value={ref.required_authority_class ?? '—'} />
              <TechRow label="Stockage" value={metadata?.storageBackend ?? '—'} />
            </div>
          </details>
        );
      })}
    </div>
  );
}

function TechRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-slate-700">{label}</dt>
      <dd className="mt-1 break-all font-mono text-slate-400">{value}</dd>
    </div>
  );
}

export function ContextSummary({ load }: { load: LoadResult }) {
  const items = [
    ['Essentiel', 'L0', load.context_plan.l0.length],
    ['Stage', 'L1', load.context_plan.l1.length],
    ['Étendu', 'L2', load.context_plan.l2.length],
    ['Historique', 'L3', load.context_plan.l3.length],
  ] as const;

  return (
    <div className="grid grid-cols-2 gap-x-5 gap-y-4 sm:grid-cols-4">
      {items.map(([label, code, count]) => (
        <div key={code}>
          <div className="flex items-baseline gap-2">
            <span className="text-xl font-semibold text-slate-200">{count}</span>
            <span className="text-[10px] uppercase tracking-[0.13em] text-slate-700">{code}</span>
          </div>
          <div className="mt-1 text-xs text-slate-500">{label}</div>
        </div>
      ))}
    </div>
  );
}

export function ContextPlan({
  load,
  catalog,
}: {
  load: LoadResult;
  catalog: Record<string, ArtifactMeta>;
}) {
  const tiers = [
    ['L0', 'Essentiel', "Ce qui définit l'état courant de l'analyse.", load.context_plan.l0],
    ['L1', 'Stage', 'Documents nécessaires pour travailler sur le stage courant.', load.context_plan.l1],
    ['L2', 'Étendu', 'Informations supplémentaires chargées à la demande.', load.context_plan.l2],
    ['L3', 'Historique', 'Contexte ancien ou élargi.', load.context_plan.l3],
  ] as const;

  return (
    <div className="border-t border-slate-800">
      {tiers.map(([code, label, description, refs]) => (
        <section key={code} className="grid gap-4 border-b border-slate-800 py-5 md:grid-cols-[180px_minmax(0,1fr)]">
          <div>
            <div className="flex items-baseline gap-2">
              <h2 className="font-medium text-slate-200">{label}</h2>
              <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-700">{code}</span>
            </div>
            <div className="mt-1 text-xs text-slate-600">{refs.length} document{refs.length === 1 ? '' : 's'}</div>
          </div>
          <div>
            <p className="text-sm text-slate-500">{description}</p>
            {refs.length > 0 ? (
              <ul className="mt-3 space-y-1.5 text-sm text-slate-300">
                {refs.map((ref) => <li key={ref.artifact_id + ':' + ref.version}>— {artifactName(ref, catalog)}</li>)}
              </ul>
            ) : (
              <div className="mt-3 text-xs text-slate-700">Aucun document chargé.</div>
            )}
          </div>
        </section>
      ))}
    </div>
  );
}

export function RunTechnicalDetails({ load }: { load: LoadResult }) {
  return (
    <details className="border-t border-slate-800 pt-4">
      <summary className="cursor-pointer text-xs text-slate-600 hover:text-slate-400">⋯ Détails techniques de l&apos;analyse</summary>
      <dl className="mt-4 grid gap-4 text-xs sm:grid-cols-2 lg:grid-cols-3">
        <TechRow label="Run" value={load.run_id ?? '—'} />
        <TechRow label="Type" value={load.run_type ?? '—'} />
        <TechRow label="Mode" value={load.canonical_mode ?? '—'} />
        <TechRow label="Version run" value={String(load.run_state_version ?? '—')} />
        <TechRow label="Stage" value={load.current_stage ?? '—'} />
        <TechRow label="Révision stage" value={String(load.stage?.stage_revision ?? '—')} />
        <TechRow label="Version stage" value={String(load.stage?.stage_state_version ?? '—')} />
        <TechRow label="Handoff gate" value={load.stage?.handoff_gate_state ?? '—'} />
        <TechRow label="Manifest actif" value={load.stage?.active_manifest ? shortId(load.stage.active_manifest.artifact_id, 12) + ' / v' + load.stage.active_manifest.version : '—'} />
        <TechRow label="Contract set" value={load.contract_set_sha256 ?? '—'} />
        <TechRow label="Process state" value={load.process_state_artifact ? shortId(load.process_state_artifact.artifact_id, 12) : 'Aucun'} />
      </dl>
    </details>
  );
}

export function RunSelector({
  issuerSlug,
  runs,
}: {
  issuerSlug: string;
  runs: RunSummary[];
}) {
  return (
    <section>
      <div className="mb-6">
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-600">Analyses disponibles</div>
        <h1 className="mt-2 text-2xl font-semibold text-white">Choisir une analyse</h1>
        <p className="mt-2 text-sm text-slate-500">La sélection reste explicite lorsqu&apos;il existe plusieurs runs.</p>
      </div>

      <div className="border-t border-slate-800">
        <div className="hidden grid-cols-[120px_120px_120px_minmax(0,1fr)_100px] gap-4 border-b border-slate-800 py-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-700 md:grid">
          <div>Données</div>
          <div>Étape</div>
          <div>État</div>
          <div>Run</div>
          <div></div>
        </div>
        {runs.map((run) => (
          <div key={run.runId} className="grid gap-2 border-b border-slate-800 py-4 md:grid-cols-[120px_120px_120px_minmax(0,1fr)_100px] md:items-center md:gap-4">
            <div className="text-sm text-slate-300">{formatCutoff(run.dataCutoff)}</div>
            <div className="text-sm text-slate-400">{stageLabel(run.currentStage)}</div>
            <RunStatusBadge status={run.runStatus} />
            <div className="truncate font-mono text-xs text-slate-600">{run.runId}</div>
            <div>
              {run.detailedMockAvailable ? (
                <Link href={buildRunHref('/orotitan/' + issuerSlug, run.runId)} className="text-sm font-medium text-cyan-300 hover:text-cyan-200">
                  Ouvrir →
                </Link>
              ) : (
                <span className="text-xs text-slate-700">Indisponible</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function UnavailableRunState({ summary }: { summary: RunSummary }) {
  return (
    <section className="border-l-2 border-slate-700 pl-5">
      <div className="text-xs font-semibold uppercase tracking-[0.17em] text-slate-600">Analyse indisponible</div>
      <h1 className="mt-2 text-xl font-semibold text-white">Cette analyse n&apos;est pas disponible dans cette version</h1>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
        Le run existe, mais son contenu détaillé n&apos;est pas chargé. OroTitan ne substitue jamais les documents d&apos;une autre analyse.
      </p>
      <div className="mt-4 flex flex-wrap gap-4 text-xs">
        <span className="text-slate-400">{stageLabel(summary.currentStage)}</span>
        <RunStatusBadge status={summary.runStatus} />
        <span className="font-mono text-slate-700">{summary.runId}</span>
      </div>
    </section>
  );
}

export function LoadErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <section className="border-l-2 border-rose-500 pl-5">
      <h2 className="text-lg font-semibold text-white">Impossible de charger l&apos;analyse.</h2>
      <p className="mt-2 text-sm text-slate-500">Les données OroTitan n&apos;ont pas été modifiées.</p>
      {onRetry ? <button type="button" onClick={onRetry} className="mt-4 text-sm font-medium text-cyan-300 hover:text-cyan-200">Réessayer →</button> : null}
    </section>
  );
}

export function StageFilterLink({
  href,
  active,
  children,
}: {
  href: string;
  active: boolean;
  children: ReactNode;
}) {
  return (
    <Link
      href={href}
      className={
        'border-b-2 px-1 pb-2 text-xs font-semibold transition ' +
        (active ? 'border-cyan-400 text-slate-200' : 'border-transparent text-slate-600 hover:text-slate-400')
      }
    >
      {children}
    </Link>
  );
}

export function countStageArtifacts(refs: ArtifactRef[], catalog: Record<string, ArtifactMeta>, stage: StageCode): number {
  return refs.filter((ref) => catalog[ref.artifact_id]?.stageCode === stage).length;
}
