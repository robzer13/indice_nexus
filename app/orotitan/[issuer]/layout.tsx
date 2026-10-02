import Link from 'next/link';
import { notFound } from 'next/navigation';
import { getMockDossier } from '@/lib/orotitan-ui/mock';

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

  const runQuery = '?run=' + encodeURIComponent(dossier.primaryRunId);
  const base = '/orotitan/' + dossier.identity.slug;

  return (
    <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="h-fit rounded-2xl border border-slate-800 bg-slate-900/45 p-4 lg:sticky lg:top-24">
        <Link href="/orotitan" className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">OroTitan</Link>
        <div className="mt-4 border-t border-slate-800 pt-4">
          <div className="text-sm font-semibold text-white">{dossier.identity.displayName}</div>
          <div className="mt-1 text-xs text-slate-600">{dossier.identity.ticker} · {dossier.identity.marketDataSymbol}</div>
        </div>
        <nav className="mt-5 space-y-1 text-sm">
          <Link href={base + runQuery} className="block rounded-lg px-3 py-2 text-slate-300 hover:bg-slate-800">Vue d'ensemble</Link>
          <Link href={base + '/documents' + runQuery} className="block rounded-lg px-3 py-2 text-slate-400 hover:bg-slate-800 hover:text-slate-200">Documents</Link>
          <Link href={base + '/context' + runQuery} className="block rounded-lg px-3 py-2 text-slate-400 hover:bg-slate-800 hover:text-slate-200">Contexte</Link>
        </nav>
        <div className="mt-5 border-t border-slate-800 pt-4 text-xs leading-5 text-slate-600">
          Mock-only · read-only<br />Aucun accès Supabase frontend
        </div>
      </aside>
      <div className="min-w-0">{children}</div>
    </div>
  );
}
