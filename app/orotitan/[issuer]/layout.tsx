import Link from 'next/link';
import { notFound } from 'next/navigation';
import { IssuerNav } from '@/components/orotitan/issuer-nav';
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

  return (
    <div className="grid gap-6 lg:grid-cols-[220px_minmax(0,1fr)]">
      <aside className="h-fit rounded-2xl border border-slate-800 bg-slate-900/45 p-4 lg:sticky lg:top-24">
        <Link href="/orotitan" className="text-xs font-semibold uppercase tracking-[0.18em] text-cyan-400">OroTitan</Link>
        <div className="mt-4 border-t border-slate-800 pt-4">
          <div className="text-sm font-semibold text-white">{dossier.identity.displayName}</div>
          <div className="mt-1 text-xs text-slate-600">{dossier.identity.ticker} · {dossier.identity.marketDataSymbol}</div>
        </div>
        <IssuerNav issuerSlug={dossier.identity.slug} />
        <div className="mt-5 border-t border-slate-800 pt-4 text-xs leading-5 text-slate-600">
          Mock-only · read-only<br />Aucun accès Supabase frontend
        </div>
      </aside>
      <div className="min-w-0">{children}</div>
    </div>
  );
}
