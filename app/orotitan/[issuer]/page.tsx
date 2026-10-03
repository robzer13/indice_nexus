import { notFound } from 'next/navigation';
import { ArtifactRegistry } from '@/components/orotitan/artifact-registry';
import {
  AnalysisRail,
  BlockerPanel,
  CompanyHeader,
  ContextSummary,
  RunSelector,
  StageProgress,
  UnavailableRunState,
  countStageArtifacts,
} from '@/components/orotitan/ui';
import { getUiDossierSelection } from '@/lib/orotitan-ui/server-data';

export default async function OroTitanOverviewPage({
  params,
  searchParams,
}: {
  params: Promise<{ issuer: string }>;
  searchParams: Promise<{ run?: string | string[] }>;
}) {
  const { issuer } = await params;
  const query = await searchParams;
  const selectedRun = typeof query.run === 'string' ? query.run : null;
  const selection = await getUiDossierSelection(issuer, selectedRun);

  if (selection.kind === 'issuer-not-found' || selection.kind === 'unknown') notFound();
  if (selection.kind === 'select') {
    return <RunSelector issuerSlug={selection.shell.identity.slug} runs={selection.shell.runSummaries} />;
  }
  if (selection.kind === 'unavailable') {
    return <UnavailableRunState summary={selection.summary} />;
  }

  const dossier = selection.dossier;
  const load = dossier.loadResult;
  if (!load.run_id) notFound();

  const deepDiveRefs = load.artifact_index.filter((ref) => dossier.artifactCatalog[ref.artifact_id]?.stageCode === 'DEEP_DIVE');

  return (
    <div>
      <CompanyHeader identity={dossier.identity} load={load} />
      <StageProgress load={load} />

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_310px]">
        <div className="min-w-0">
          <BlockerPanel blockers={load.blockers} />
        </div>
        <AnalysisRail load={load} />
      </div>

      <section className="mt-5 rounded-xl border border-[rgba(123,173,214,.18)] bg-[rgba(8,22,36,.58)] p-4 sm:p-5">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">Documents et preuves</div>
            <h2 className="mt-1.5 text-lg font-semibold text-white">Preuves du Deep Dive</h2>
          </div>
          <span className="rounded-full border border-slate-700 bg-slate-900/70 px-2.5 py-1 text-[10px] font-semibold text-slate-400">
            {countStageArtifacts(load.artifact_index, dossier.artifactCatalog, 'DEEP_DIVE')}
          </span>
        </div>
        <ArtifactRegistry
          refs={deepDiveRefs}
          catalog={dossier.artifactCatalog}
          issuerSlug={dossier.identity.slug}
          runId={load.run_id}
        />
      </section>

      <section className="mt-5 rounded-xl border border-[rgba(123,173,214,.18)] bg-[rgba(8,22,36,.58)] p-4 sm:p-5">
        <div className="mb-4">
          <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">Contexte de l&apos;analyse</div>
        </div>
        <ContextSummary load={load} />
      </section>
    </div>
  );
}
