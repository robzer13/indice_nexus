import Link from 'next/link';
import { notFound } from 'next/navigation';
import { IssuerNav } from '@/components/orotitan/issuer-nav';
import { getMockDossier } from '@/lib/orotitan-ui/mock';
import { runStatusLabel, stageLabel } from '@/lib/orotitan-ui/presentation';

export default async function OroTitanIssuerLayout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: Promise<{ issuer: string }>;
}>) {
  const { issuer } = await params;
  const dossier = getMockDossier(issuer);
  if (!dossier) notFound();

  const primaryRun = dossier.runSummaries.find((run) => run.runId === dossier.primaryRunId);

  return (
    <div className="grid gap-7 lg:grid-cols-[200px_minmax(0,1fr)]">
      <aside className="h-fit border-b border-slate-800 pb-5 lg:sticky lg:top-24 lg:border-b-0 lg:border-r lg:pr-6">
        <Link href="/orotitan" className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-600 hover:text-cyan-300">
          ← Equity Research
        </Link>
        <div className="mt-5">
          <div className="text-base font-semibold text-white">{dossier.identity.displayName}</div>
          <div className="mt-1 text-xs text-slate-600">{dossier.identity.ticker} · {dossier.identity.marketDataSymbol}</div>
        </div>

        <IssuerNav issuerSlug={dossier.identity.slug} />

        {primaryRun ? (
          <div className="mt-6 border-t border-slate-800 pt-4">
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-700">Analyse</div>
            <div className="mt-2 text-xs text-slate-400">
              {stageLabel(primaryRun.currentStage)} · <span className="text-rose-300">{runStatusLabel(primaryRun.runStatus)}</span>
            </div>
            <div className="mt-2 text-[11px] text-slate-700">Lecture seule</div>
          </div>
        ) : null}
      </aside>
      <div className="min-w-0">{children}</div>
    </div>
  );
}
