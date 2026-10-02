import { notFound } from 'next/navigation';
import { ArtifactList, RunSelector, StageFilterLink, UnavailableRunState } from '@/components/orotitan/ui';
import { getMockDossier, resolveMockRunSelection } from '@/lib/orotitan-ui/mock';
import { buildRunHref } from '@/lib/orotitan-ui/presentation';
import type { StageCode } from '@/lib/orotitan-ui/types';

const allowedStages = new Set<StageCode>(['RESEARCH', 'DEEP_DIVE', 'INTEGRATION']);

export default async function OroTitanDocumentsPage({
  params,
  searchParams,
}: {
  params: Promise<{ issuer: string }>;
  searchParams: Promise<{ stage?: string | string[]; run?: string | string[] }>;
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

  const rawStage = typeof query.stage === 'string' ? query.stage : null;
  const stage = rawStage && allowedStages.has(rawStage as StageCode) ? rawStage as StageCode : null;
  const refs = stage
    ? dossier.loadResult.artifact_index.filter((ref) => dossier.artifactCatalog[ref.artifact_id]?.stageCode === stage)
    : dossier.loadResult.artifact_index;

  const path = '/orotitan/' + dossier.identity.slug + '/documents';

  return (
    <div>
      <header className="border-b border-slate-800 pb-5">
        <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600">Documents et preuves</div>
        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <h1 className="text-2xl font-semibold text-white">Registre des documents</h1>
          <span className="text-xs text-slate-600">{dossier.loadResult.artifact_index.length} documents disponibles</span>
        </div>
      </header>

      <nav className="flex flex-wrap gap-5 border-b border-slate-800 pt-5">
        <StageFilterLink href={buildRunHref(path, selectedRun)} active={!stage}>Tous</StageFilterLink>
        <StageFilterLink href={buildRunHref(path, selectedRun, { stage: 'RESEARCH' })} active={stage === 'RESEARCH'}>Recherche</StageFilterLink>
        <StageFilterLink href={buildRunHref(path, selectedRun, { stage: 'DEEP_DIVE' })} active={stage === 'DEEP_DIVE'}>Deep Dive</StageFilterLink>
        <StageFilterLink href={buildRunHref(path, selectedRun, { stage: 'INTEGRATION' })} active={stage === 'INTEGRATION'}>Intégration</StageFilterLink>
      </nav>

      <ArtifactList refs={refs} catalog={dossier.artifactCatalog} />
    </div>
  );
}
