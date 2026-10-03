import { notFound } from 'next/navigation';
import { ContextPlan, RunSelector, UnavailableRunState } from '@/components/orotitan/ui';
import { getUiDossierSelection } from '@/lib/orotitan-ui/server-data';

export default async function OroTitanContextPage({
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
