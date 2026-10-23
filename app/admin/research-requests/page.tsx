import type { Metadata } from 'next';
import Link from 'next/link';
import { redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import {
  researchInboxPilotMode, getPilotResearchCompanies, loadPilotResearchInbox,
} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import { requestResearchPilotAction } from './actions';

export const dynamic = 'force-dynamic';
export const metadata: Metadata = { title: 'Demandes Research — pilote local OroTitan' };

type Params = Promise<Record<string, string | string[] | undefined>>;
function stringParam(value: string | string[] | undefined): string | null {
  return typeof value === 'string' ? value : null;
}

export default async function ResearchRequestsPage({ searchParams }: { searchParams: Params }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const mode = researchInboxPilotMode();
  const enabled = mode !== 'disabled';
  const query = await searchParams;

  if (!enabled) {
    return <main className="space-y-6">
      <header className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-400">
          OroTitan · Parcours Research
        </p>
        <h1 className="text-3xl font-semibold text-white">Demandes Research</h1>
      </header>
      <AdminNav />
      <section className="rounded-xl border border-amber-900 bg-amber-950/20 p-5 text-sm text-amber-100">
        Le pilote local est désactivé. Aucun formulaire d’écriture n’est accessible en production.
        Cette page ne crée ni entreprise, ni demande cloud, ni run OroTitan.
      </section>
    </main>;
  }

  const [companies, requests] = await Promise.all([
    getPilotResearchCompanies(), loadPilotResearchInbox(),
  ]);
  const activeCompanies = companies.filter(c => c.active);
  const requestId = stringParam(query.request);
  const outcome = stringParam(query.result);
  const error = stringParam(query.error);

  return <main className="space-y-6">
    <header className="space-y-2">
      <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-400">
        OroTitan · Produit · Pilote {mode === 'local' ? 'local' : 'partagé non-production'}
      </p>
      <h1 className="text-3xl font-semibold text-white">Boîte de réception Research</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-400">
        Parcours métier : sélectionner une entreprise du Screener, enregistrer une demande
        durable, récupérer son numéro et consulter son état. Aucune analyse n’est
        exécutée ni publiée par ce pilote.
      </p>
    </header>
    <AdminNav />
    <div role="status" className="rounded-xl border border-amber-800/60 bg-amber-950/20 p-4 text-sm text-amber-100">
      <strong>Environnement de développement uniquement.</strong> {mode === 'local'
        ? 'Les données restent dans un fichier privé sur cette machine : aucune synchronisation entre installations.'
        : 'Les données sont enregistrées dans le projet Supabase de TEST explicitement isolé de la production. Les sessions utilisant ce même backend peuvent retrouver les mêmes demandes.'}
      {' '}Aucun Registry run ni Research n’est lancé. Une société du Screener
      ne constitue pas automatiquement un émetteur Research identifié.
    </div>

    {requestId && (outcome === 'created' || outcome === 'existing') ?
      <section className="rounded-xl border border-cyan-900 bg-cyan-950/20 p-4 text-sm text-cyan-100">
        <strong>{outcome === 'created' ? 'Demande enregistrée.' : 'Demande déjà existante.'}</strong>
        <p className="mt-2">Référence de demande :
          <code className="ml-2 break-all font-mono text-cyan-200">{requestId}</code>
        </p>
        <p className="mt-1 text-xs">Statut : REQUESTED_NOT_EXECUTED — aucun Research lancé.</p>
      </section> : null}
    {error && <p role="alert" className="rounded-xl border border-rose-900 bg-rose-950/20 p-4 text-sm text-rose-200">
      {error === 'missing-company' ? 'Sélectionnez une société.' :
        'Impossible d’enregistrer la demande locale. Vérifiez le référentiel et le stockage du pilote.'}
    </p>}

    <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="text-lg font-semibold text-white">1. Choisir une entreprise</h2>
        <p className="mt-2 text-sm text-slate-400">
          La sélection utilise les sociétés actives réellement référencées dans le Screener.
          Cette action enregistre une intention, et non un mandat d’exécution.
        </p>
        <form action={requestResearchPilotAction} className="mt-5 space-y-4">
          <label htmlFor="company_id" className="block text-sm text-slate-300">Entreprise</label>
          <select id="company_id" name="company_id" required disabled={!activeCompanies.length}
            defaultValue="" className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-3 text-sm text-slate-100">
            <option value="" disabled>Sélectionner une société</option>
            {activeCompanies.map(c => <option key={c.id} value={c.id}>
              {c.name} · {c.ticker} · {c.exchange}
            </option>)}
          </select>
          <button disabled={!activeCompanies.length}
            className="rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 disabled:cursor-not-allowed disabled:opacity-50">
            Enregistrer une demande {mode === 'local' ? 'locale' : 'partagée de test'}
          </button>
        </form>
        <p className="mt-3 text-xs text-slate-500">
          Entreprise absente ? <Link href="/admin/companies/new" className="text-cyan-300 underline">
            Utiliser le formulaire Société existant
          </Link>. L’ajout au Screener et l’admission au Registry restent deux opérations distinctes.
        </p>
      </div>
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="text-lg font-semibold text-white">2. Vérifier la suite du parcours</h2>
        <dl className="mt-4 space-y-3 text-sm">
          <div className="flex justify-between gap-4 border-b border-slate-800 pb-3">
            <dt className="text-slate-400">Demandes enregistrées</dt>
            <dd className="font-semibold text-white">{requests.length}</dd>
          </div>
          <div className="flex justify-between gap-4 border-b border-slate-800 pb-3">
            <dt className="text-slate-400">Demandes non exécutées</dt>
            <dd className="font-semibold text-amber-200">{requests.length}</dd>
          </div>
          <div className="flex justify-between gap-4 border-b border-slate-800 pb-3">
            <dt className="text-slate-400">Runs Research créés par ce pilote</dt>
            <dd className="font-semibold text-white">0</dd>
          </div>
        </dl>
        <p className="mt-4 text-sm leading-6 text-slate-400">
          Pour les runs déjà existants, utilisez le <Link className="text-cyan-300 underline"
            href="/admin/orotitan">cockpit Registry V3.1 en lecture seule</Link>.
          Il ne faut pas confondre ces observations avec les demandes locales.
        </p>
      </div>
    </section>

    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <h2 className="text-lg font-semibold text-white">3. Retrouver les demandes</h2>
      <p className="mt-1 text-sm text-slate-400">
        {mode === 'local' ? 'Historique local sur cette machine.' : 'Historique partagé dans le projet de test.'}
        {' '}Deux demandes identiques pour une entreprise renvoient la même référence.
      </p>
      {requests.length === 0 ?
        <p className="mt-5 text-sm text-slate-400">Aucune demande enregistrée dans ce pilote.</p> :
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="border-b border-slate-700 text-xs uppercase text-slate-500">
              <tr><th className="py-3 pr-4">Entreprise</th><th className="py-3 pr-4">Date</th>
                <th className="py-3 pr-4">État</th><th className="py-3">Référence</th></tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {requests.map(r => <tr key={r.requestId}>
                <td className="py-4 pr-4 font-medium text-white">{r.companyName}
                  <span className="ml-2 text-xs text-slate-500">{r.ticker} · {r.exchange}</span>
                </td>
                <td className="py-4 pr-4 text-slate-400">
                  {new Date(r.createdAt).toLocaleString('fr-FR')}
                </td>
                <td className="py-4 pr-4 text-amber-200">Demandé · non exécuté</td>
                <td className="break-all py-4 font-mono text-xs text-slate-400">{r.requestId}</td>
              </tr>)}
            </tbody>
          </table>
        </div>}
    </section>
  </main>;
}
