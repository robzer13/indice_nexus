import { notFound } from 'next/navigation';
import { ArtifactList, StageFilterLink } from '@/components/orotitan/ui';
import { getMockDossier } from '@/lib/orotitan-ui/mock';
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

  const rawStage = typeof query.stage === 'string' ? query.stage : null;
  const stage = rawStage && allowedStages.has(rawStage as StageCode) ? rawStage as StageCode : null;
  const refs = stage
    ? dossier.loadResult.artifact_index.filter((ref) => dossier.artifactCatalog[ref.artifact_id]?.stageCode === stage)
    : dossier.loadResult.artifact_index;

  const base = '/orotitan/' + dossier.identity.slug + '/documents?run=' + encodeURIComponent(dossier.primaryRunId);

  return (
    <div className="space-y-6">
      <header>
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">Documents et preuves</div>
        <h1 className="mt-2 text-3xl font-semibold text-white">{dossier.identity.legalName}</h1>
        <p className="mt-2 text-sm text-slate-500">{dossier.loadResult.artifact_index.length} artefacts actifs dans artifact_index.</p>
      </header>
      <nav className="flex flex-wrap gap-2">
        <StageFilterLink href={base} active={!stage}>Tous</StageFilterLink>
        <StageFilterLink href={base + '&stage=RESEARCH'} active={stage === 'RESEARCH'}>Recherche</StageFilterLink>
        <StageFilterLink href={base + '&stage=DEEP_DIVE'} active={stage === 'DEEP_DIVE'}>Deep Dive</StageFilterLink>
        <StageFilterLink href={base + '&stage=INTEGRATION'} active={stage === 'INTEGRATION'}>Intégration</StageFilterLink>
      </nav>
      <ArtifactList refs={refs} catalog={dossier.artifactCatalog} />
    </div>
  );
}
