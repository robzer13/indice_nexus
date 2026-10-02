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

const stateDot: Record<StageLifecycle, string> = {
  NOT_STARTED: 'border-slate-600 bg-[#07111d]',
  IN_PROGRESS: 'border-cyan-300 bg-cyan-400',
  PAUSED: 'border-amber-400 bg-amber-400',
  BLOCKED: 'border-rose-300 bg-rose-400',
  COMPLETE: 'border-emerald-300 bg-emerald-400',
};

export function RunStatusBadge({ status }: { status: string | null }) {
  const classes =
    status === 'BLOCKED'
      ? 'border-rose-400/30 bg-rose-400/10 text-rose-300'
      : status === 'PUBLISHED'
        ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300'
        : status === 'ACTIVE'
          ? 'border-amber-400/30 bg-amber-400/10 text-amber-300'
          : 'border-slate-700 bg-slate-900 text-slate-400';

  return (
    <span className={'inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] ' + classes}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {runStatusLabel(status)}
    </span>
  );
}

export function CompanyHeader({
  identity,
  load,
}: {
  identity: CompanyIdentity;
  load: LoadResult;
}) {
  return (
    <header
      className="relative overflow-hidden rounded-xl border border-[rgba(123,190,235,.18)] bg-[#07111d] px-5 py-6 shadow-[0_18px_48px_rgba(0,0,0,.16)] sm:px-7 sm:py-7"
      style={{
        backgroundImage:
          'radial-gradient(circle at 84% 30%, rgba(43,200,255,.13), transparent 0 24%), radial-gradient(circle at 72% 75%, rgba(36,99,235,.09), transparent 0 18%), linear-gradient(110deg, rgba(3,10,18,.96), rgba(7,17,29,.88))',
      }}
    >
      <div aria-hidden="true" className="absolute -right-20 top-1/2 h-48 w-80 -translate-y-1/2 rotate-[-8deg] rounded-full border border-cyan-300/10" />
      <div aria-hidden="true" className="absolute -right-4 top-1/2 h-28 w-64 -translate-y-1/2 rotate-[-8deg] rounded-full border border-blue-300/10" />

      <div className="relative flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-[0.20em] text-cyan-400">Equity Research</div>
          <h1 className="mt-3 font-serif text-3xl font-semibold tracking-wide text-white sm:text-[38px]">{identity.legalName}</h1>
          <p className="mt-2 text-sm text-slate-400">{identity.ticker} · {identity.marketDataSymbol}</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs text-slate-500">Données au {formatCutoff(load.data_cutoff)}</span>
          <RunStatusBadge status={load.run_status} />
        </div>
      </div>
    </header>
  );
}

export function StageProgress({ load }: { load: LoadResult }) {
  const states = deriveStageStates(load.current_stage, load.stage?.lifecycle_status ?? null);

  return (
    <section aria-label="Progression de l'analyse" className="px-1 py-5">
      <div className="grid gap-4 md:grid-cols-3 md:gap-0">
        {states.map(({ stage, lifecycle }, index) => (
          <div key={stage} className="relative flex items-center gap-3 md:pr-5">
            <div className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-[rgba(123,173,214,.18)] bg-[#07111d]">
              <span className={'h-3 w-3 rounded-full border ' + stateDot[lifecycle]} />
            </div>
            <div className="min-w-0">
              <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300">{stageLabel(stage)}</div>
              <div className={'mt-1 text-xs ' + (lifecycle === 'BLOCKED' ? 'text-rose-300' : lifecycle === 'COMPLETE' ? 'text-emerald-400' : 'text-slate-600')}>
                {lifecycleLabel(lifecycle)}
              </div>
            </div>
            {index < states.length - 1 ? <span className="ml-2 hidden h-px flex-1 bg-[rgba(123,173,214,.18)] md:block" /> : null}
          </div>
        ))}
      </div>
    </section>
  );
}

function blockerPresentation(blocker: Blocker): { summary?: string; impact?: string; resolution?: string } {
  if (blocker.code === 'ECONOMIC_SHARE_COUNT_UNRESOLVED') {
    return {
      summary: "La valorisation par action reste impossible tant que le nombre économique d'actions au 22 septembre 2026 n'est pas démontré de manière conforme.",
      impact: 'Valorisation par action et résultats dépendants.',
      resolution: "Fermer toutes les classes de mouvements susceptibles de modifier le dénominateur jusqu'au 22/09/2026 avec une preuve exacte ou une borne rigoureuse.",
    };
  }
  return {};
}

export function BlockerPanel({ blockers }: { blockers: Blocker[] }) {
  if (blockers.length === 0) {
    return (
      <section className="rounded-xl border border-emerald-400/20 bg-emerald-400/[.04] px-5 py-5 text-sm text-emerald-300">
        Aucun problème actif.
      </section>
    );
  }

  return (
    <section className="space-y-4">
      {blockers.map((blocker) => {
        const presentation = blockerPresentation(blocker);
        return (
          <article
            key={blocker.code}
            className="overflow-hidden rounded-xl border border-rose-400/45 bg-[linear-gradient(120deg,rgba(86,20,30,.34),rgba(20,11,19,.72))] p-5 shadow-[0_0_34px_rgba(255,93,104,.045)] sm:p-6"
          >
            <div className="flex gap-4">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-rose-400/35 bg-rose-400/10 text-lg font-semibold text-rose-300">!</div>
              <div className="min-w-0 flex-1">
                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-rose-300">Problème actif</div>
                <h2 className="mt-2 font-serif text-2xl font-semibold text-white">{blockerTitle(blocker)}</h2>
                <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-300">
                  {presentation.summary ?? blocker.detail ?? 'Ce problème bloque la poursuite normale de l’analyse.'}
                </p>
              </div>
            </div>

            <div className="mt-5 grid gap-5 border-t border-rose-300/15 pt-5 md:grid-cols-2">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-rose-300">Impact</div>
                <p className="mt-2 text-sm leading-6 text-slate-300">{presentation.impact ?? blocker.scope ?? 'Impact non précisé.'}</p>
              </div>
              <div className="md:border-l md:border-rose-300/15 md:pl-5">
                <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-rose-300">À résoudre</div>
                <p className="mt-2 text-sm leading-6 text-slate-300">
                  {presentation.resolution ?? blocker.resolution_required ?? 'Résoudre le problème selon les exigences du stage courant.'}
                </p>
              </div>
            </div>

            <details className="mt-5 border-t border-rose-300/15 pt-4">
              <summary className="cursor-pointer text-xs text-rose-200/60 hover:text-rose-100">Voir les détails techniques →</summary>
              <dl className="mt-4 grid gap-4 text-xs sm:grid-cols-2">
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

export function ContextSummary({ load }: { load: LoadResult }) {
  const items = [
    ['Essentiel', 'L0', load.context_plan.l0.length, "Ce qui définit l'état courant de l'analyse.", 'border-cyan-400/25 bg-cyan-400/[.055]', 'text-cyan-300'],
    ['Stage', 'L1', load.context_plan.l1.length, 'Documents nécessaires pour travailler sur le stage courant.', 'border-purple-400/20 bg-purple-400/[.055]', 'text-purple-300'],
    ['Étendu', 'L2', load.context_plan.l2.length, 'Informations supplémentaires chargées à la demande.', 'border-[rgba(123,173,214,.16)] bg-[#0a1624]/55', 'text-slate-300'],
    ['Historique', 'L3', load.context_plan.l3.length, 'Contexte ancien ou élargi.', 'border-[rgba(123,173,214,.16)] bg-[#0a1624]/55', 'text-slate-300'],
  ] as const;

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {items.map(([label, code, count, description, surface, accent]) => (
        <div key={code} className={'rounded-[10px] border p-4 ' + surface}>
          <div className="flex items-start justify-between gap-3">
            <div className={'text-sm font-medium ' + accent}>{label}</div>
            <span className="rounded-md border border-white/10 bg-white/[.035] px-2 py-1 text-[10px] font-semibold text-slate-500">{code}</span>
          </div>
          <div className="mt-2 text-sm text-slate-200">{count} document{count === 1 ? '' : 's'}</div>
          <p className="mt-2 text-xs leading-5 text-slate-500">{description}</p>
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
    <div className="divide-y divide-[rgba(123,173,214,.14)] rounded-xl border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.52)]">
      {tiers.map(([code, label, description, refs]) => (
        <section key={code} className="grid gap-5 px-5 py-6 md:grid-cols-[190px_minmax(0,1fr)]">
          <div>
            <div className="flex items-baseline gap-2">
              <h2 className="text-base font-medium text-slate-200">{label}</h2>
              <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-600">{code}</span>
            </div>
            <div className="mt-2 text-xs text-slate-600">{refs.length} document{refs.length === 1 ? '' : 's'}</div>
          </div>
          <div>
            <p className="text-sm text-slate-500">{description}</p>
            {refs.length > 0 ? (
              <ul className="mt-4 space-y-2 text-sm text-slate-300">
                {refs.map((ref) => <li key={ref.artifact_id + ':' + ref.version}>— {catalog[ref.artifact_id]?.logicalName ?? ref.artifact_id}</li>)}
              </ul>
            ) : (
              <div className="mt-4 text-xs text-slate-700">Aucun document chargé.</div>
            )}
          </div>
        </section>
      ))}
    </div>
  );
}

export function AnalysisRail({ load }: { load: LoadResult }) {
  return (
    <aside className="space-y-4 xl:sticky xl:top-[82px]">
      <section className="rounded-xl border border-[rgba(123,190,235,.18)] bg-[rgba(8,22,36,.72)] p-4">
        <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-300">Analyse sélectionnée</div>
        <dl className="mt-4 space-y-3 text-xs">
          <RailRow label="Date des données" value={formatCutoff(load.data_cutoff)} />
          <RailRow label="Étape" value={load.current_stage ? stageLabel(load.current_stage) : '—'} />
          <RailRow label="État" value={runStatusLabel(load.run_status)} danger={load.run_status === 'BLOCKED'} />
          <RailRow label="Version" value={load.run_id ? 'Run ' + shortId(load.run_id, 12) : '—'} mono />
        </dl>
      </section>

      <details className="rounded-xl border border-[rgba(123,190,235,.18)] bg-[rgba(8,22,36,.72)] p-4">
        <summary className="cursor-pointer text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-400">Détails techniques</summary>
        <dl className="mt-4 space-y-3 text-xs">
          <RailRow label="Run ID" value={load.run_id ? shortId(load.run_id, 16) : '—'} mono />
          <RailRow label="Run state version" value={String(load.run_state_version ?? '—')} />
          <RailRow label="Stage state version" value={String(load.stage?.stage_state_version ?? '—')} />
          <RailRow label="Contract set" value={load.contract_set_sha256 ? shortId(load.contract_set_sha256, 16) : '—'} mono />
          <RailRow label="Manifest" value={load.stage?.active_manifest ? shortId(load.stage.active_manifest.artifact_id, 16) : 'Aucun'} mono />
          <RailRow label="Handoff" value={load.stage?.handoff_gate_state ?? '—'} />
          <RailRow label="Process state" value={load.process_state_artifact ? shortId(load.process_state_artifact.artifact_id, 16) : 'Aucun'} mono />
          <RailRow label="Mode" value="Lecture seule" />
        </dl>
      </details>
    </aside>
  );
}

function RailRow({ label, value, danger = false, mono = false }: { label: string; value: string; danger?: boolean; mono?: boolean }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-[rgba(123,173,214,.10)] pb-3 last:border-0 last:pb-0">
      <dt className="text-slate-600">{label}</dt>
      <dd className={'text-right ' + (danger ? 'text-rose-300' : 'text-slate-300') + (mono ? ' font-mono text-[11px]' : '')}>{value}</dd>
    </div>
  );
}

function TechRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-slate-600">{label}</dt>
      <dd className="mt-1 break-all font-mono text-[11px] text-slate-400">{value}</dd>
    </div>
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
    <section className="rounded-xl border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.52)] p-5 sm:p-6">
      <div className="mb-6">
        <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">Analyses disponibles</div>
        <h1 className="mt-2 text-2xl font-semibold text-white">Choisir une analyse</h1>
        <p className="mt-2 text-sm text-slate-500">Aucune analyse n&apos;est sélectionnée automatiquement.</p>
      </div>

      <div className="overflow-hidden rounded-[10px] border border-[rgba(123,173,214,.14)]">
        <div className="hidden grid-cols-[120px_120px_120px_minmax(0,1fr)_100px] gap-4 border-b border-[rgba(123,173,214,.14)] bg-[#0a1624]/55 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600 md:grid">
          <div>Données</div>
          <div>Étape</div>
          <div>État</div>
          <div>Run</div>
          <div></div>
        </div>
        {runs.map((run) => (
          <div key={run.runId} className="grid gap-2 border-b border-[rgba(123,173,214,.10)] px-4 py-4 last:border-b-0 md:grid-cols-[120px_120px_120px_minmax(0,1fr)_100px] md:items-center md:gap-4">
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
    <section className="rounded-xl border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.62)] p-6">
      <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">Analyse indisponible</div>
      <h1 className="mt-2 text-xl font-semibold text-white">Cette analyse n&apos;est pas disponible dans cette version</h1>
      <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
        Le run existe, mais son contenu détaillé n&apos;est pas chargé. OroTitan ne substitue jamais les documents d&apos;une autre analyse.
      </p>
      <div className="mt-4 flex flex-wrap items-center gap-3 text-xs">
        <span className="text-slate-400">{stageLabel(summary.currentStage)}</span>
        <RunStatusBadge status={summary.runStatus} />
        <span className="font-mono text-slate-700">{summary.runId}</span>
      </div>
    </section>
  );
}

export function LoadErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <section className="rounded-xl border border-rose-400/30 bg-rose-400/[.04] p-6">
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
        'rounded-full border px-3 py-1.5 text-xs font-medium transition ' +
        (active
          ? 'border-cyan-400/35 bg-cyan-400/10 text-cyan-100'
          : 'border-[rgba(123,173,214,.14)] bg-[#07111d]/50 text-slate-500 hover:text-slate-300')
      }
    >
      {children}
    </Link>
  );
}

export function countStageArtifacts(refs: ArtifactRef[], catalog: Record<string, ArtifactMeta>, stage: StageCode): number {
  return refs.filter((ref) => catalog[ref.artifact_id]?.stageCode === stage).length;
}
