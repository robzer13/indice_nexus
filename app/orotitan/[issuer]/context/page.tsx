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
    <div>
      <header className="mb-5 rounded-xl border border-[rgba(123,173,214,.16)] bg-[rgba(8,22,36,.54)] px-5 py-5">
        <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-400">Contexte de l&apos;analyse</div>
        <h1 className="mt-2 text-2xl font-semibold text-white">Ce qui est chargé pour travailler</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
          Les niveaux L0 à L3 organisent le contexte selon sa fonction. Un niveau vide reste un état normal.
        </p>
      </header>
      <ContextPlan load={dossier.loadResult} catalog={dossier.artifactCatalog} />
    </div>
  );
}
