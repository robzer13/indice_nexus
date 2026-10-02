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

function Pill({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'success' | 'warning' | 'danger' }) {
  const tones = {
    neutral: 'border-slate-700 bg-slate-900 text-slate-300',
    success: 'border-emerald-800 bg-emerald-950/50 text-emerald-200',
    warning: 'border-amber-800 bg-amber-950/40 text-amber-200',
    danger: 'border-rose-800 bg-rose-950/45 text-rose-200',
  };
  return <span className={'inline-flex rounded-full border px-2.5 py-1 text-xs font-semibold ' + tones[tone]}>{children}</span>;
}

export function RunStatusBadge({ status }: { status: string | null }) {
  const tone = status === 'BLOCKED' ? 'danger' : status === 'ACTIVE' ? 'warning' : status === 'PUBLISHED' ? 'success' : 'neutral';
  return <Pill tone={tone}>{runStatusLabel(status)}</Pill>;
}

export function CompanyHeader({
  identity,
  load,
}: {
  identity: CompanyIdentity;
  load: LoadResult;
}) {
  return (
    <section className="flex flex-col gap-5 border-b border-slate-800 pb-6 lg:flex-row lg:items-end lg:justify-between">
      <div>
        <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Analyse OroTitan</div>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">{identity.legalName}</h1>
        <p className="mt-2 text-sm text-slate-400">{identity.ticker} · {identity.marketDataSymbol}</p>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <span className="text-slate-500">Données au {formatCutoff(load.data_cutoff)}</span>
        <RunStatusBadge status={load.run_status} />
      </div>
    </section>
  );
}

function lifecycleTone(value: StageLifecycle): 'neutral' | 'success' | 'warning' | 'danger' {
  if (value === 'COMPLETE') return 'success';
  if (value === 'BLOCKED') return 'danger';
  if (value === 'IN_PROGRESS' || value === 'PAUSED') return 'warning';
  return 'neutral';
}

export function StageProgress({ load }: { load: LoadResult }) {
  const states = deriveStageStates(load.current_stage, load.stage?.lifecycle_status ?? null);
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/45 p-5">
      <div className="grid gap-3 md:grid-cols-3">
        {states.map(({ stage, lifecycle }, index) => (
          <div key={stage} className="relative rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">{stageLabel(stage)}</span>
              <span className="font-mono text-sm text-slate-600">{index + 1}/3</span>
            </div>
            <div className="mt-3"><Pill tone={lifecycleTone(lifecycle)}>{lifecycleLabel(lifecycle)}</Pill></div>
          </div>
        ))}
      </div>
    </section>
  );
}

export function BlockerPanel({ blockers }: { blockers: Blocker[] }) {
  if (blockers.length === 0) {
    return <section className="rounded-2xl border border-emerald-900/70 bg-emerald-950/20 p-5 text-sm text-emerald-200">Aucun blocker actif.</section>;
  }

  return (
    <section className="space-y-4">
      {blockers.map((blocker) => (
        <article key={blocker.code} className="rounded-2xl border border-rose-900/70 bg-rose-950/20 p-5">
          <div className="text-xs font-semibold uppercase tracking-[0.18em] text-rose-300">Problème à résoudre</div>
          <h2 className="mt-2 text-xl font-semibold text-white">{blockerTitle(blocker)}</h2>
          {blocker.detail ? <p className="mt-3 max-w-4xl text-sm leading-6 text-slate-300">{blocker.detail}</p> : null}
          <div className="mt-5 grid gap-3 md:grid-cols-2">
            {blocker.classification ? <Info label="Classification" value={blocker.classification} /> : null}
            {blocker.scope ? <Info label="Portée" value={blocker.scope} /> : null}
            {blocker.diagnostic ? <Info label="Diagnostic" value={blocker.diagnostic} /> : null}
            {blocker.resolution_required ? <Info label="Condition de résolution" value={blocker.resolution_required} /> : null}
          </div>
          <details className="mt-4 text-xs text-slate-500">
            <summary className="cursor-pointer select-none hover:text-slate-300">Code machine</summary>
            <code className="mt-2 block break-all text-slate-400">{blocker.code}</code>
          </details>
        </article>
      ))}
    </section>
  );
}

function Info({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
      <div className="text-xs text-slate-600">{label}</div>
      <div className="mt-1 break-words text-sm text-slate-300">{value}</div>
    </div>
  );
}

function artifactName(ref: ArtifactRef, catalog: Record<string, ArtifactMeta>): string {
  return catalog[ref.artifact_id]?.logicalName ?? ref.artifact_id;
}

export function ArtifactList({
  refs,
  catalog,
}: {
  refs: ArtifactRef[];
  catalog: Record<string, ArtifactMeta>;
}) {
  if (refs.length === 0) {
    return <div className="rounded-xl border border-slate-800 bg-slate-900/45 p-5 text-sm text-slate-500">Aucun document dans ce périmètre.</div>;
  }

  return (
    <div className="space-y-3">
      {refs.map((ref) => {
        const metadata = catalog[ref.artifact_id];
        return (
          <article key={ref.artifact_id + ':' + ref.version} className="rounded-xl border border-slate-800 bg-slate-900/55 p-4">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div className="min-w-0">
                <div className="font-medium text-slate-100">{artifactName(ref, catalog)}</div>
                <div className="mt-1 break-all font-mono text-xs text-slate-600">{metadata?.artifactType ?? 'ARTIFACT'}</div>
              </div>
              <div className="flex shrink-0 flex-wrap gap-2">
                {metadata ? <Pill>{stageLabel(metadata.stageCode)}</Pill> : null}
                <Pill tone={metadata?.authorityState === 'AUTHORITATIVE' ? 'success' : 'warning'}>{metadata?.authorityState === 'AUTHORITATIVE' ? 'Autoritatif' : 'Checkpoint'}</Pill>
                <Pill>v{ref.version}</Pill>
              </div>
            </div>
            <details className="mt-4 border-t border-slate-800 pt-3">
              <summary className="cursor-pointer text-xs font-medium text-slate-500 hover:text-slate-300">Voir les détails techniques</summary>
              <dl className="mt-3 grid gap-3 text-xs sm:grid-cols-2">
                <TechRow label="Artifact ID" value={ref.artifact_id} />
                <TechRow label="SHA-256" value={ref.content_sha256 ?? '—'} />
                <TechRow label="Authority class" value={ref.required_authority_class ?? '—'} />
                <TechRow label="Stockage" value={metadata?.storageBackend ?? '—'} />
              </dl>
            </details>
          </article>
        );
      })}
    </div>
  );
}

function TechRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-slate-600">{label}</dt>
      <dd className="mt-1 break-all font-mono text-slate-400">{value}</dd>
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
    ['L0', 'État essentiel', load.context_plan.l0],
    ['L1', 'Contexte du stage', load.context_plan.l1],
    ['L2', 'Contexte étendu', load.context_plan.l2],
    ['L3', 'Contexte historique', load.context_plan.l3],
  ] as const;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {tiers.map(([code, label, refs]) => (
        <section key={code} className="rounded-2xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="flex items-baseline justify-between gap-3">
            <h2 className="font-semibold text-white">{code} — {label}</h2>
            <span className="font-mono text-xs text-slate-500">{refs.length}</span>
          </div>
          {refs.length === 0 ? (
            <p className="mt-4 text-sm text-slate-600">Aucun document chargé.</p>
          ) : (
            <ul className="mt-4 space-y-2 text-sm text-slate-300">
              {refs.map((ref) => <li key={ref.artifact_id + ':' + ref.version}>• {artifactName(ref, catalog)}</li>)}
            </ul>
          )}
        </section>
      ))}
    </div>
  );
}

export function RunTechnicalDetails({ load }: { load: LoadResult }) {
  return (
    <details className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
      <summary className="cursor-pointer text-sm font-medium text-slate-400 hover:text-slate-200">Détails techniques du run</summary>
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
      <div className="mb-5">
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">Sélection explicite</div>
        <h1 className="mt-2 text-2xl font-semibold text-white">Plusieurs analyses sont disponibles</h1>
        <p className="mt-2 text-sm text-slate-500">Aucun run n&apos;est choisi automatiquement.</p>
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        {runs.map((run) => (
          <article key={run.runId} className="rounded-2xl border border-slate-800 bg-slate-900/55 p-5">
            <div className="flex items-start justify-between gap-3">
              <RunStatusBadge status={run.runStatus} />
              <span className="text-xs text-slate-600">v{run.stateVersion}</span>
            </div>
            <div className="mt-4 text-lg font-semibold text-slate-100">{stageLabel(run.currentStage)}</div>
            <div className="mt-1 text-sm text-slate-500">Données au {formatCutoff(run.dataCutoff)}</div>
            <div className="mt-3 font-mono text-xs text-slate-600">{shortId(run.runId, 12)}</div>
            {run.detailedMockAvailable ? (
              <Link href={buildRunHref('/orotitan/' + issuerSlug, run.runId)} className="mt-5 inline-flex rounded-lg border border-cyan-700 bg-cyan-950/40 px-3 py-2 text-sm font-semibold text-cyan-100 hover:bg-cyan-950/70">
                Ouvrir ce run
              </Link>
            ) : (
              <div className="mt-5 text-xs text-slate-600">Mock détaillé non chargé dans la V1.</div>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

export function UnavailableRunState({ summary }: { summary: RunSummary }) {
  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6">
      <h1 className="text-xl font-semibold text-white">Mock détaillé non chargé</h1>
      <p className="mt-2 text-sm leading-6 text-slate-400">
        Ce run existe dans le sélecteur, mais la V1 ne possède pas de LOAD_RESULT détaillé pour ce run. Aucun contenu provenant d&apos;une autre analyse n&apos;est substitué.
      </p>
      <div className="mt-4 flex flex-wrap gap-2">
        <RunStatusBadge status={summary.runStatus} />
        <Pill>{stageLabel(summary.currentStage)}</Pill>
        <Pill>v{summary.stateVersion}</Pill>
      </div>
      <div className="mt-3 font-mono text-xs text-slate-600">{summary.runId}</div>
    </section>
  );
}

export function MutationActions() {
  return (
    <section className="rounded-xl border border-slate-800 bg-slate-900/35 p-4">
      <div className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Actions de stage</div>
      <div className="mt-3 flex flex-wrap gap-2">
        {['Enregistrer le checkpoint', 'Finaliser le stage', 'Réouvrir le stage'].map((label) => (
          <button key={label} type="button" disabled title="Disponible après validation du bridge d'écriture OroTitan." className="cursor-not-allowed rounded-lg border border-slate-800 bg-slate-950 px-3 py-2 text-xs text-slate-600">
            {label}
          </button>
        ))}
      </div>
      <p className="mt-3 text-xs text-slate-600">Interface V1 mock-only · aucune mutation disponible.</p>
    </section>
  );
}

export function LoadErrorState({ onRetry }: { onRetry?: () => void }) {
  return (
    <section className="rounded-2xl border border-rose-900/60 bg-rose-950/20 p-6">
      <h2 className="text-lg font-semibold text-white">Impossible de charger l&apos;analyse.</h2>
      <p className="mt-2 text-sm text-slate-400">Les données OroTitan n&apos;ont pas été modifiées.</p>
      {onRetry ? <button type="button" onClick={onRetry} className="mt-4 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300">Réessayer</button> : null}
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
  return <Link href={href} className={'rounded-lg border px-3 py-2 text-xs font-semibold ' + (active ? 'border-cyan-700 bg-cyan-950/50 text-cyan-100' : 'border-slate-800 bg-slate-950/50 text-slate-500 hover:text-slate-300')}>{children}</Link>;
}

export function countStageArtifacts(refs: ArtifactRef[], catalog: Record<string, ArtifactMeta>, stage: StageCode): number {
  return refs.filter((ref) => catalog[ref.artifact_id]?.stageCode === stage).length;
}
