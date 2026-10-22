import type { Metadata } from 'next';
import Link from 'next/link';
import { ScreenerTable } from '@/components/screener-table';
import { getCompanyStates } from '@/lib/data/companies';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getUnpublishedTrackedCompanies } from '@/lib/data/company-intake';

export const metadata: Metadata = { title: 'Screener' };
export const dynamic = 'force-dynamic';

export default async function ScreenerPage() {
  const [companies, admin] = await Promise.all([getCompanyStates(), isAdminAuthenticated()]);
  // Only authorized administrators see unreviewed tracking candidates. Public
  // visitors continue to see strictly published canonical research.
  const pending = admin ? await getUnpublishedTrackedCompanies() : [];
  return <div className="space-y-6">
    <div>
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Univers canonique publié · V1</div>
      <h1 className="mt-2 text-3xl font-semibold text-white">Screener OroTitan</h1>
      <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">Seuls les dossiers disposant d’un <code>current_snapshot_id</code> canonique sont intégrés aux classements et aux scores OQS, OVS et Investment.</p>
      {admin ? <div className="mt-3 flex flex-wrap gap-3 text-sm text-slate-400">
        <span>{companies.length} titres publiés</span>
        <span>·</span>
        <a className="text-cyan-300 underline underline-offset-4" href="#titres-en-attente">{pending.length} titre{pending.length>1?'s':''} en suivi non publié</a>
        <Link className="text-cyan-300 underline underline-offset-4" href="/admin/companies/new">Ajouter une action</Link>
      </div> : null}
    </div>
    {companies.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société publiée dans OroTitan.</div> : <ScreenerTable companies={companies}/>}
    {admin ? <section id="titres-en-attente" className="space-y-4 rounded-xl border border-slate-700 bg-slate-900/50 p-5">
      <div>
        <h2 className="text-lg font-semibold text-white">Titres ajoutés au suivi — non publiés ({pending.length})</h2>
        <p className="mt-1 text-sm text-slate-400">Vue réservée à l’administration. Ces titres ne possèdent pas encore de dossier canonique publié : aucun score, seuil H ou recommandation n’est calculé. Leur ajout au référentiel ne déclenche aucune publication.</p>
      </div>
      {pending.length === 0 ? <p className="text-sm text-emerald-300">Aucun titre en attente d’intégration canonique.</p> :
      <div className="overflow-x-auto rounded-lg border border-slate-800">
        <table className="min-w-[650px] w-full text-left text-sm">
          <thead className="bg-slate-950 text-xs uppercase tracking-wide text-slate-500"><tr>
            <th className="px-4 py-3">Société</th><th className="px-4 py-3">Ticker</th><th className="px-4 py-3">Marché</th><th className="px-4 py-3">Devise</th><th className="px-4 py-3">Statut</th><th className="px-4 py-3">Suivi</th>
          </tr></thead>
          <tbody className="divide-y divide-slate-800 bg-slate-950/40">{pending.map((company) => <tr key={company.id}>
            <td className="px-4 py-3 font-medium text-slate-100">{company.name}</td>
            <td className="px-4 py-3 font-mono text-slate-300">{company.ticker}</td>
            <td className="px-4 py-3 text-slate-400">{company.exchange}</td>
            <td className="px-4 py-3 text-slate-400">{company.currency}</td>
            <td className="px-4 py-3 text-amber-300">À analyser</td>
            <td className="px-4 py-3"><Link href={`/admin/companies/${company.slug}`} className="text-cyan-300 hover:text-cyan-200">Métadonnées</Link></td>
          </tr>)}</tbody>
        </table>
      </div>}
    </section> : null}
  </div>;
}
