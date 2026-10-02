import { notFound } from 'next/navigation';
import {
  ArtifactList,
  BlockerPanel,
  CompanyHeader,
  MutationActions,
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
    <div className="space-y-6">
      <CompanyHeader identity={dossier.identity} load={load} />
      <StageProgress load={load} />
      <BlockerPanel blockers={load.blockers} />

      <section className="rounded-2xl border border-slate-800 bg-slate-900/45 p-5">
        <div className="flex items-baseline justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-white">Documents du Deep Dive</h2>
            <p className="mt-1 text-sm text-slate-500">Artefacts actifs du stage courant.</p>
          </div>
          <span className="font-mono text-xs text-slate-500">{countStageArtifacts(load.artifact_index, dossier.artifactCatalog, 'DEEP_DIVE')} disponibles</span>
        </div>
        <div className="mt-4"><ArtifactList refs={deepDiveRefs} catalog={dossier.artifactCatalog} /></div>
      </section>

      <section className="grid gap-4 sm:grid-cols-4">
        {[
          ['L0', load.context_plan.l0.length],
          ['L1', load.context_plan.l1.length],
          ['L2', load.context_plan.l2.length],
          ['L3', load.context_plan.l3.length],
        ].map(([label, count]) => (
          <div key={label} className="rounded-xl border border-slate-800 bg-slate-900/45 p-4">
            <div className="text-xs text-slate-600">{label}</div>
            <div className="mt-1 text-lg font-semibold text-slate-200">{count} <span className="text-xs font-normal text-slate-600">document{count === 1 ? '' : 's'}</span></div>
          </div>
        ))}
      </section>

      <RunTechnicalDetails load={load} />
      <MutationActions />
    </div>
  );
}
