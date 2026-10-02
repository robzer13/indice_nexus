import { notFound } from 'next/navigation';
import { ContextPlan, RunSelector, UnavailableRunState } from '@/components/orotitan/ui';
import { getMockDossier, resolveMockRunSelection } from '@/lib/orotitan-ui/mock';

export default async function OroTitanContextPage({
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

  return (
    <div className="space-y-6">
      <header>
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">Contexte de travail</div>
        <h1 className="mt-2 text-3xl font-semibold text-white">L0–L3</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
          Représentation directe du context_plan du LOAD_RESULT. Un niveau vide est affiché comme vide et n&apos;est pas interprété comme une erreur.
        </p>
      </header>
      <ContextPlan load={dossier.loadResult} catalog={dossier.artifactCatalog} />
    </div>
  );
}
