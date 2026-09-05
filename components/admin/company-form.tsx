import { createCompanyAction, updateCompanyAction } from '@/app/admin/actions';
import type { CompanyRecord } from '@/lib/domain/types';

export function CompanyForm({ company }: { company?: CompanyRecord }) {
  const editing = Boolean(company);
  const action = editing ? updateCompanyAction : createCompanyAction;
  return <form action={action} className="space-y-6 rounded-xl border border-slate-800 bg-slate-900/60 p-6">
    {company ? <><input type="hidden" name="company_id" value={company.id}/><input type="hidden" name="current_slug" value={company.slug}/></> : null}
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
      <Field name="name" label="Nom" required defaultValue={company?.name}/>
      <Field name="ticker" label="Ticker" required defaultValue={company?.ticker}/>
      <Field name="slug" label="Slug" required defaultValue={company?.slug} hint="minuscules, chiffres et tirets"/>
      <Field name="exchange" label="Place de cotation" required defaultValue={company?.exchange}/>
      <Field name="currency" label="Devise" required defaultValue={company?.currency ?? 'EUR'} hint="EUR, USD, GBP, NOK…"/>
      <label className="text-sm text-slate-400">Unité de cotation<select name="quote_unit" defaultValue={company?.quote_unit ?? 'MAJOR'} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="MAJOR">MAJOR</option><option value="MINOR">MINOR</option></select></label>
      <Field name="price_decimals" label="Décimales" type="number" required defaultValue={company?.price_decimals ?? 2} min="0" max="6" step="1"/>
      <Field name="market_data_symbol" label="Symbole Twelve Data" defaultValue={company?.market_data_symbol ?? ''} hint="ex. RMS:EPA"/>
      <Field name="market_data_multiplier" label="Multiplicateur marché" type="number" required defaultValue={company?.market_data_multiplier ?? 1} min="0.000001" step="0.000001"/>
      <Field name="country" label="Pays" defaultValue={company?.country ?? ''}/>
      <Field name="sector" label="Secteur" defaultValue={company?.sector ?? ''}/>
      <label className="text-sm text-slate-400">Statut de suivi<select name="active" defaultValue={company?.active === false ? 'false' : 'true'} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100"><option value="true">Active</option><option value="false">Inactive</option></select></label>
    </div>
    <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4 text-xs leading-5 text-slate-500">Auto Trader doit rester en <strong className="text-slate-300">GBP / MINOR / 0 décimale / multiplicateur 100</strong>. Modifier une société ne modifie aucun snapshot ni aucun cours historique.</div>
    <div className="flex justify-end"><button className="rounded-lg bg-cyan-500 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-400">{editing ? 'Enregistrer les métadonnées' : 'Créer la société'}</button></div>
  </form>;
}

function Field({ name, label, hint, type = 'text', required = false, defaultValue, min, max, step }: { name: string; label: string; hint?: string; type?: string; required?: boolean; defaultValue?: string | number; min?: string; max?: string; step?: string }) {
  return <label className="text-sm text-slate-400">{label}<input name={name} type={type} required={required} defaultValue={defaultValue} min={min} max={max} step={step} className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-slate-100 outline-none focus:border-cyan-500"/>{hint ? <span className="mt-1 block text-xs text-slate-600">{hint}</span> : null}</label>;
}
