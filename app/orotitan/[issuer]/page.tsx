import { notFound } from 'next/navigation';
import {
  ArtifactList,
  BlockerPanel,
  CompanyHeader,
  ContextSummary,
  RunSelector,
  RunTechnicalDetails,
  StageProgress,
  UnavailableRunState,
  countStageArtifacts,
} from '@/components/orotitan/ui';
import { getMockDossier, resolveMockRunSelection } from '@/lib/orotitan-ui/mock';

export default async function OroTitanOverviewPage({
  params,
  searchParams,
}: {
  params: Promise<{ issuer: string }>;
  searchParams: Promise<{ run?: string | string[] }>;
}) {
  const { issuer } = await params;
  const query = await searchParams;
  const dossier = getMockDossier(issuer);
  if (!dossier) notFound();

  const selectedRun = typeof query.run === 'string' ? query.run : null;
  const selection = resolveMockRunSelection(dossier, selectedRun);

  if (selection.kind === 'select') {
    return <RunSelector issuerSlug={dossier.identity.slug} runs={dossier.runSummaries} />;
  }
  if (selection.kind === 'unknown') notFound();
  if (selection.kind === 'unavailable') {
    return <UnavailableRunState summary={selection.summary} />;
  }

  const load = dossier.loadResult;
  const deepDiveRefs = load.artifact_index.filter((ref) => dossier.artifactCatalog[ref.artifact_id]?.stageCode === 'DEEP_DIVE');

  return (
    <div>
      <CompanyHeader identity={dossier.identity} load={load} />
      <StageProgress load={load} />
      <BlockerPanel blockers={load.blockers} />

      <section className="border-b border-slate-800 py-6">
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <div className="text-[11px] font-semibold uppercase tracking-[0.17em] text-slate-600">Preuves du Deep Dive</div>
            <h2 className="mt-1 text-lg font-semibold text-white">Documents du stage</h2>
          </div>
          <span className="text-xs text-slate-600">
            {countStageArtifacts(load.artifact_index, dossier.artifactCatalog, 'DEEP_DIVE')} disponibles
          </span>
        </div>
        <ArtifactList refs={deepDiveRefs} catalog={dossier.artifactCatalog} compact />
      </section>

      <section className="border-b border-slate-800 py-6">
        <div className="mb-4">
          <div className="text-[11px] font-semibold uppercase tracking-[0.17em] text-slate-600">Contexte</div>
          <h2 className="mt-1 text-lg font-semibold text-white">Contexte chargé</h2>
        </div>
        <ContextSummary load={load} />
      </section>

      <div className="py-5">
        <RunTechnicalDetails load={load} />
      </div>
    </div>
  );
}
