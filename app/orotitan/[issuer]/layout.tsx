import Link from 'next/link';
import { notFound } from 'next/navigation';
import { IssuerNav } from '@/components/orotitan/issuer-nav';
import { getUiDossierShell } from '@/lib/orotitan-ui/server-data';

export default async function OroTitanIssuerLayout({
  children,
  params,
}: Readonly<{
  children: React.ReactNode;
  params: Promise<{ issuer: string }>;
}>) {
  const { issuer } = await params;
  const dossier = await getUiDossierShell(issuer);
  if (!dossier) notFound();

  return (
    <div className="grid gap-5 lg:grid-cols-[232px_minmax(0,1fr)]">
      <aside className="border-b border-[rgba(123,173,214,.14)] pb-5 lg:sticky lg:top-[82px] lg:h-[calc(100vh-106px)] lg:border-b-0 lg:border-r lg:pr-5">
        <Link href="/orotitan" className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600 transition hover:text-cyan-300">
          ← Retour
        </Link>

        <div className="mt-5">
          <div className="font-serif text-2xl font-semibold tracking-wide text-slate-100">{dossier.identity.displayName.toUpperCase()}</div>
          <div className="mt-1.5 text-sm text-slate-500">{dossier.identity.ticker} · {dossier.identity.marketDataSymbol}</div>
        </div>

        <IssuerNav issuerSlug={dossier.identity.slug} runs={dossier.runSummaries} />
      </aside>

      <div className="min-w-0">{children}</div>
    </div>
  );
}
