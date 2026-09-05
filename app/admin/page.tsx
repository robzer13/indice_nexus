import type { Metadata } from 'next';
import { createSnapshotAction, loginAction, logoutAction } from '@/app/admin/actions';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getActiveCompanies } from '@/lib/data/companies';
import { snapshotStatuses } from '@/lib/domain/snapshot';

export const metadata: Metadata = { title: 'Admin' };
export const dynamic = 'force-dynamic';

type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function messageValue(value: string | string[] | undefined): string | null {
  return typeof value === 'string' ? value : null;
}

function NumberField({ name, label, min = 0, max, step = '0.01' }: { name: string; label: string; min?: number; max?: number; step?: string }) {
  return <label className="text-sm text-slate-400">{label}<input name={name} type="number" min={min} max={max} step={step} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500" /></label>;
}

function TextArea({ name, label, required = false, placeholder }: { name: string; label: string; required?: boolean; placeholder?: string }) {
  return <label className="text-sm text-slate-400">{label}<textarea name={name} required={required} placeholder={placeholder} rows={4} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500" /></label>;
}

export default async function AdminPage({ searchParams }: { searchParams: SearchParams }) {
  const query = await searchParams;
  const error = messageValue(query.error);
  const success = messageValue(query.success);
  const authenticated = await isAdminAuthenticated();

  if (!authenticated) {
    return <div className="mx-auto max-w-md rounded-xl border border-slate-800 bg-slate-900/70 p-6 shadow-panel"><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Zone privée</div><h1 className="mt-2 text-2xl font-semibold text-white">Administration OroTitan</h1><p className="mt-2 text-sm leading-6 text-slate-400">Le mot de passe est vérifié uniquement côté serveur. La session résultante est signée et stockée dans un cookie HTTP-only.</p>{error ? <div className="mt-4 rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}<form action={loginAction} className="mt-5 space-y-4"><label className="block text-sm text-slate-400">Mot de passe<input name="password" type="password" required autoComplete="current-password" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500" /></label><button className="w-full rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400">Se connecter</button></form></div>;
  }

  const companies = await getActiveCompanies();

  return <div className="space-y-6"><div className="flex items-end justify-between gap-4"><div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Snapshot writer</div><h1 className="mt-2 text-3xl font-semibold text-white">Nouveau snapshot analytique</h1><p className="mt-2 max-w-3xl text-sm text-slate-400">Insertion uniquement. La clé métier <span className="font-mono text-slate-300">company_id + analysis_date + model_version</span> ne peut pas être écrasée.</p></div><form action={logoutAction}><button className="rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-900">Déconnexion</button></form></div>{error ? <div className="rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}{success ? <div className="rounded-lg border border-emerald-900 bg-emerald-950/30 p-3 text-sm text-emerald-200">{success}</div> : null}{companies.length === 0 ? <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-6 text-slate-500">Aucune société active disponible.</div> : <form action={createSnapshotAction} className="space-y-7 rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-panel sm:p-6"><section><h2 className="text-sm font-semibold text-white">Identité du snapshot</h2><div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-4"><label className="text-sm text-slate-400">Société<select name="company_id" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Sélectionner</option>{companies.map((company) => <option key={company.id} value={company.id}>{company.name} ({company.ticker})</option>)}</select></label><label className="text-sm text-slate-400">Date d’analyse<input name="analysis_date" type="date" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100" /></label><label className="text-sm text-slate-400">Model version<input name="model_version" required maxLength={120} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100" /></label><label className="text-sm text-slate-400">Statut<select name="status" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Sélectionner</option>{snapshotStatuses.map((status) => <option key={status}>{status}</option>)}</select></label></div><div className="mt-4 grid gap-4 md:grid-cols-2"><label className="text-sm text-slate-400">Qualité OroTitan<select name="quality_orotitan" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Non renseigné</option><option value="true">Oui</option><option value="false">Non</option></select></label><label className="text-sm text-slate-400">Source title <span className="text-rose-300">obligatoire</span><input name="source_title" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100" /></label></div></section>

  <section><h2 className="text-sm font-semibold text-white">Scores</h2><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-5"><NumberField name="business_quality_score" label="Business quality" max={100}/><NumberField name="investment_score" label="Investment" max={100}/><NumberField name="valuation_score" label="Valuation" max={100}/><NumberField name="orotitan_score" label="OroTitan score" max={100}/><NumberField name="confidence_score" label="Confiance" max={10}/></div></section>

  <section><h2 className="text-sm font-semibold text-white">Valorisation et seuils</h2><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><NumberField name="fair_value_low" label="Fair value basse"/><NumberField name="fair_value_base" label="Fair value centrale"/><NumberField name="fair_value_high" label="Fair value haute"/><NumberField name="price_o85" label="O85"/><NumberField name="price_o90" label="O90"/><NumberField name="price_o92" label="O92"/><NumberField name="price_o95" label="O95"/></div><p className="mt-3 text-xs text-slate-500">Un champ laissé vide devient <code>NULL</code>. Aucun zéro n’est injecté pour remplacer une valeur inconnue.</p></section>

  <section><h2 className="text-sm font-semibold text-white">Analyse</h2><div className="mt-4 grid gap-4 lg:grid-cols-3"><TextArea name="thesis" label="Thesis"/><TextArea name="main_risk" label="Main risk"/><TextArea name="invalidation" label="Invalidation"/></div><div className="mt-4 grid gap-4 lg:grid-cols-2"><TextArea name="notes" label="Notes"/><TextArea name="score_components" label="Score components JSON" placeholder='{"moat": 90, "pricing": 95}'/></div></section>

  <div className="flex justify-end"><button className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400">Créer le snapshot immutable</button></div></form>}</div>;
}
