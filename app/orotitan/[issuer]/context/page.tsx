import { notFound } from 'next/navigation';
import { ContextPlan } from '@/components/orotitan/ui';
import { getMockDossier } from '@/lib/orotitan-ui/mock';

export default async function OroTitanContextPage({
  params,
}: {
  params: Promise<{ issuer: string }>;
}) {
  const { issuer } = await params;
  const dossier = getMockDossier(issuer);
  if (!dossier) notFound();

  return (
    <div className="space-y-6">
      <header>
        <div className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">Contexte de travail</div>
        <h1 className="mt-2 text-3xl font-semibold text-white">L0–L3</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
          Représentation directe du context_plan du LOAD_RESULT. Un niveau vide est affiché comme vide et n'est pas interprété comme une erreur.
        </p>
      </header>
      <ContextPlan load={dossier.loadResult} catalog={dossier.artifactCatalog} />
    </div>
  );
}
