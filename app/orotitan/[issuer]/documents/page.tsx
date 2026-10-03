import { notFound } from 'next/navigation';
import { ArtifactRegistry } from '@/components/orotitan/artifact-registry';
import { RunSelector, StageFilterLink, UnavailableRunState } from '@/components/orotitan/ui';
import { buildRunHref } from '@/lib/orotitan-ui/presentation';
import { getUiDossierSelection } from '@/lib/orotitan-ui/server-data';
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
  const rawStage = typeof query.stage === 'string' ? query.stage : null;
  const stage = rawStage && allowedStages.has(rawStage as StageCode) ? rawStage as StageCode : null;
  const refs = stage
    ? dossier.loadResult.artifact_index.filter((ref) => dossier.artifactCatalog[ref.artifact_id]?.stageCode === stage)
    : dossier.loadResult.artifact_index;

  const path = '/orotitan/' + dossier.identity.slug + '/documents';

  return (
    <div>
      <header className="mb-5 rounded-xl border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.54)] px-5 py-5">
        <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">Documents et preuves</div>
        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <h1 className="text-2xl font-semibold text-white">Registre des documents</h1>
          <span className="text-xs text-slate-600">{dossier.loadResult.artifact_index.length} documents disponibles</span>
        </div>
      </header>

      <div className="mb-4 flex flex-wrap gap-2">
        <StageFilterLink href={buildRunHref(path, selectedRun)} active={!stage}>Tous</StageFilterLink>
        <StageFilterLink href={buildRunHref(path, selectedRun, { stage: 'RESEARCH' })} active={stage === 'RESEARCH'}>Recherche</StageFilterLink>
        <StageFilterLink href={buildRunHref(path, selectedRun, { stage: 'DEEP_DIVE' })} active={stage === 'DEEP_DIVE'}>Deep Dive</StageFilterLink>
        <StageFilterLink href={buildRunHref(path, selectedRun, { stage: 'INTEGRATION' })} active={stage === 'INTEGRATION'}>Intégration</StageFilterLink>
      </div>

      <ArtifactRegistry refs={refs} catalog={dossier.artifactCatalog} searchable />
    </div>
  );
}
