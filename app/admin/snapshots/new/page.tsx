import { redirect } from 'next/navigation';
import { AdminNav } from '@/components/admin/admin-nav';
import { SnapshotJsonPreview } from '@/components/admin/snapshot-json-preview';
import { createSnapshotAction, createSnapshotJsonAction } from '@/app/admin/actions';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { getActiveCompanies } from '@/lib/data/companies';
import { snapshotStatuses } from '@/lib/domain/snapshot';

export const dynamic = 'force-dynamic';
type SearchParams = Promise<Record<string, string | string[] | undefined>>;

function NumberField({ name, label, max, step = '0.01' }: { name: string; label: string; max?: number; step?: string }) {
  return <label className="text-sm text-slate-400">{label}<input name={name} type="number" min="0" max={max} step={step} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500"/></label>;
}

function TextArea({ name, label, required = false }: { name: string; label: string; required?: boolean }) {
  return <label className="text-sm text-slate-400">{label}<textarea name={name} required={required} rows={4} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500"/></label>;
}

export default async function NewSnapshotPage({ searchParams }: { searchParams: SearchParams }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const [companies, query] = await Promise.all([getActiveCompanies(), searchParams]);
  const error = typeof query.error === 'string' ? query.error : null;
  const success = typeof query.success === 'string' ? query.success : null;

  return <div className="space-y-6">
    <div><div className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-400">Analyse versionnée</div><h1 className="mt-2 text-3xl font-semibold text-white">Nouveau snapshot</h1><p className="mt-2 text-sm text-slate-400">Insertion uniquement. La combinaison société + date + model_version reste immutable.</p></div>
    <AdminNav/>
    {error ? <div className="rounded-lg border border-rose-900 bg-rose-950/30 p-3 text-sm text-rose-200">{error}</div> : null}
    {success ? <div className="rounded-lg border border-emerald-900 bg-emerald-950/25 p-3 text-sm text-emerald-200">{success}</div> : null}

    <section className="rounded-xl border border-cyan-900/50 bg-cyan-950/10 p-5">
      <div><h2 className="text-lg font-semibold text-white">Import JSON rapide</h2><p className="mt-1 text-sm text-slate-500">Sélectionne la société, colle le bloc analytique OroTitan et vérifie la prévalidation avant insertion.</p></div>
      <form action={createSnapshotJsonAction} className="mt-4 space-y-4">
        <label className="block text-sm text-slate-400">Société<select name="json_company_id" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Sélectionner</option>{companies.map((company) => <option key={company.id} value={company.id}>{company.name} ({company.ticker})</option>)}</select></label>
        <SnapshotJsonPreview/>
        <div className="flex justify-end"><button className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950">Insérer le JSON immutable</button></div>
      </form>
    </section>

    <details className="rounded-xl border border-slate-800 bg-slate-900/55 p-5">
      <summary className="cursor-pointer font-semibold text-white">Formulaire structuré</summary>
      <form action={createSnapshotAction} className="mt-6 space-y-7">
        <section><h2 className="text-sm font-semibold text-white">Identité</h2><div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <label className="text-sm text-slate-400">Société<select name="company_id" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Sélectionner</option>{companies.map((company) => <option key={company.id} value={company.id}>{company.name} ({company.ticker})</option>)}</select></label>
          <label className="text-sm text-slate-400">Date d’analyse<input name="analysis_date" type="date" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"/></label>
          <label className="text-sm text-slate-400">Model version<input name="model_version" required maxLength={120} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"/></label>
          <label className="text-sm text-slate-400">Statut<select name="status" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Sélectionner</option>{snapshotStatuses.map((status) => <option key={status}>{status}</option>)}</select></label>
        </div><div className="mt-4 grid gap-4 md:grid-cols-2"><label className="text-sm text-slate-400">Qualité OroTitan<select name="quality_orotitan" className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="">Non renseigné</option><option value="true">Oui</option><option value="false">Non</option></select></label><label className="text-sm text-slate-400">Source title <span className="text-rose-300">obligatoire</span><input name="source_title" required className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"/></label></div></section>
        <section><h2 className="text-sm font-semibold text-white">Scores</h2><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-5"><NumberField name="business_quality_score" label="Business quality" max={100}/><NumberField name="investment_score" label="Investment" max={100}/><NumberField name="valuation_score" label="Valuation" max={100}/><NumberField name="orotitan_score" label="OroTitan score" max={100}/><NumberField name="confidence_score" label="Confiance" max={10}/></div></section>
        <section><h2 className="text-sm font-semibold text-white">Valorisation</h2><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-4"><NumberField name="fair_value_low" label="FV basse"/><NumberField name="fair_value_base" label="FV centrale"/><NumberField name="fair_value_high" label="FV haute"/><NumberField name="price_o85" label="O85"/><NumberField name="price_o90" label="O90"/><NumberField name="price_o92" label="O92"/><NumberField name="price_o95" label="O95"/></div><p className="mt-3 text-xs text-slate-500">Vide = NULL. Aucun zéro de remplacement.</p></section>
        <section><h2 className="text-sm font-semibold text-white">Analyse</h2><div className="mt-4 grid gap-4 lg:grid-cols-3"><TextArea name="thesis" label="Thèse"/><TextArea name="main_risk" label="Risque principal"/><TextArea name="invalidation" label="Invalidation"/></div><div className="mt-4 grid gap-4 lg:grid-cols-2"><TextArea name="notes" label="Notes"/><label className="text-sm text-slate-400">Score components JSON<textarea name="score_components" rows={4} placeholder='{"moat":90}' className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 font-mono text-xs text-slate-100"/></label></div></section>
        <div className="flex justify-end"><button className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950">Créer le snapshot immutable</button></div>
      </form>
    </details>
  </div>;
}
