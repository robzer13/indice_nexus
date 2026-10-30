import { isHostedShadowResearchEnabled } from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import Link from 'next/link';

const links = [
  ['/admin', 'Vue d’ensemble'],
  ['/admin/companies', 'Sociétés'],
  ['/admin/snapshots/new', 'Nouveau snapshot'],
  ['/admin/prices', 'Cours'],
  ['/admin/paper', 'Portefeuille virtuel'],
  ['/admin/data-health', 'Data Health'],
  ['/admin/orotitan', 'Pilotage V3.1'],
] as const;

export function AdminNav() {
  return <nav className="flex flex-wrap gap-2 rounded-xl border border-slate-800 bg-slate-900/55 p-2">
    {links.map(([href, label]) => <Link key={href} href={href} className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-800 hover:text-white">{label}</Link>)}
    {((process.env.NODE_ENV === 'development' && (process.env.OROTITAN_LOCAL_REQUESTS_ENABLED === '1' || process.env.OROTITAN_SHARED_INBOX_TEST_ENABLED === '1')) || isHostedShadowResearchEnabled(process.env)) ?
      <Link href="/admin/research-requests" className="rounded-lg px-3 py-2 text-sm text-cyan-200 hover:bg-slate-800 hover:text-white">
        Demandes Research · pilote
      </Link> : null}
    {isHostedShadowResearchEnabled(process.env) ?
      <Link href="/admin/research-recovery" className="rounded-lg px-3 py-2 text-sm text-cyan-200 hover:bg-slate-800 hover:text-white">
        Research Recovery Loop · SHADOW
      </Link> : null}
  </nav>;
}
