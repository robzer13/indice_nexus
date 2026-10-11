import { redirect } from 'next/navigation';
import type { Metadata } from 'next';
import { AdminNav } from '@/components/admin/admin-nav';
import { isAdminAuthenticated } from '@/lib/auth/admin-session';
import { eurNumber, loadPaperDashboard } from '@/lib/data/paper-portfolio';
import { getFreshness } from '@/lib/domain/freshness';
import { createPaperPortfolioAction, createPaperPolicyAction, evaluatePaperSignalAction, executeVirtualFillAction } from './actions';

export const metadata: Metadata = { title: 'Portefeuille virtuel' };
export const dynamic = 'force-dynamic';
const eur = (n: number) => new Intl.NumberFormat('fr-FR',{style:'currency',currency:'EUR',maximumFractionDigits:2}).format(n);
const dateLabel = (v: string) => new Date(v).toLocaleString('fr-FR');
const formatQty = (v: number) => new Intl.NumberFormat('fr-FR',{maximumFractionDigits:6}).format(v);

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
    <h2 className="text-lg font-semibold text-white">{title}</h2><div className="mt-4">{children}</div>
  </section>;
}
function Input({ label, name, type = 'text', placeholder, required = false }: {
  label:string; name:string; type?:string; placeholder?:string; required?:boolean
}) {
  return <label className="block text-xs text-slate-400">{label}
    <input name={name} type={type} required={required} placeholder={placeholder}
      min={type==='number'?'0':undefined} step={type==='number'?'any':undefined}
      className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white"/>
  </label>;
}
const selectClass = 'mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white';
const btn = 'rounded-lg bg-cyan-400 px-4 py-2.5 text-sm font-semibold text-slate-950 hover:bg-cyan-300';

export default async function PaperPage({ searchParams }: { searchParams: Promise<Record<string,string|string[]|undefined>> }) {
  if (!(await isAdminAuthenticated())) redirect('/admin');
  const [data, qs] = await Promise.all([loadPaperDashboard(),searchParams]);
  const names = new Map(data.companies.map(c => [c.id,c]));
  const filledIds = new Set(data.fills.map(f => f.signal_id));
  const cash = data.portfolio ? eurNumber(data.portfolio.cash_eur) : 0;
  const capital = data.portfolio ? eurNumber(data.portfolio.initial_cash_eur) : 0;
  const cost = data.positions.reduce((n,p) => n+eurNumber(p.quantity)*eurNumber(p.avg_cost_eur),0);
  const pnl = data.equityEur === null ? null : data.equityEur-capital;
  return <main className="space-y-6">
    <div className="border-b border-slate-800 pb-6">
      <p className="text-xs font-bold uppercase tracking-[.2em] text-cyan-400">OroTitan · Investment Lab · SIMULATION</p>
      <h1 className="mt-2 text-3xl font-semibold text-white">Portefeuille virtuel</h1>
      <p className="mt-2 text-sm text-slate-400">CORE : conserver la qualité. VALUE : cibler la revalorisation.
        Transactions fictives, après le signal, sans ordre réel. Univers actuel : actions cotées directement en EUR.</p>
    </div>
    <AdminNav/>
    {typeof qs.error==='string' && <p className="rounded-xl border border-rose-800 bg-rose-950/30 p-4 text-sm text-rose-200">{qs.error}</p>}
    {typeof qs.success==='string' && <p className="rounded-xl border border-emerald-900 bg-emerald-950/30 p-4 text-sm text-emerald-200">{qs.success}</p>}
    {data.migrationMissing ? <Panel title="Migration non appliquée">
      <p className="text-sm text-amber-200">La base ne contient pas encore les tables de simulation. Aucune migration n’est appliquée automatiquement.</p>
      <p className="mt-3 font-mono text-xs text-slate-400">migrations/20261011_paper_long_term_portfolio_v1.sql</p>
    </Panel> : !data.portfolio ? <Panel title="Initialisation du laboratoire">
      <p className="mb-4 text-sm text-slate-400">Démarrer avec 100 000 € fictifs et zéro position. Aucun accès à un courtier.</p>
      <form action={createPaperPortfolioAction}><button className={btn}>Initialiser le portefeuille</button></form>
    </Panel> : <>
      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          ['Valeur',data.equityEur===null?'Indisponible':eur(data.equityEur)],
          ['Liquidités',eur(cash)],
          ['Positions au coût',eur(cost)],
          ['P&L total',pnl===null?'—':eur(pnl)],
        ].map(([a,b]) => <div key={a} className="rounded-2xl border border-slate-800 bg-slate-900/65 p-5">
          <p className="text-xs text-slate-500">{a}</p><p className="mt-2 text-2xl font-semibold text-white">{b}</p>
        </div>)}
      </section>
      {data.unpricedPositions>0 && <p className="rounded-xl border border-amber-800 bg-amber-950/20 p-3 text-sm text-amber-200">
        {data.unpricedPositions} position(s) sans cotation EUR fraîche : performance globale volontairement masquée.</p>}
      <Panel title="Positions">
        {data.positions.length===0 ? <p className="text-sm text-slate-500">Aucune position virtuelle ouverte.</p> :
          <div className="overflow-x-auto"><table className="min-w-[650px] w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500"><tr><th>Société</th><th>Stratégie</th><th>Quantité</th><th>PRU</th><th>Cours</th><th>Valeur</th></tr></thead>
            <tbody className="divide-y divide-slate-800">
              {data.positions.map(p => {const c=names.get(p.company_id),q=data.quotes.get(p.company_id);
                const valid=!!q && c?.currency==='EUR' && c?.quote_unit==='MAJOR' && !getFreshness(q.as_of).stale;
                return <tr key={p.company_id} className="text-slate-200">
                  <td className="py-3 font-medium">{c?.name??p.company_id.slice(0,8)}</td>
                  <td>{data.policies.find(x=>x.company_id===p.company_id)?.profile??'—'}</td>
                  <td>{formatQty(eurNumber(p.quantity))}</td><td>{eur(eurNumber(p.avg_cost_eur))}</td>
                  <td>{valid&&q?eur(eurNumber(q.price)):'—'}</td>
                  <td>{valid&&q?eur(eurNumber(q.price)*eurNumber(p.quantity)):'—'}</td>
                </tr>;})}
            </tbody></table></div>}
      </Panel>
      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Nouvelle stratégie ou révision">
          <p className="mb-3 text-xs leading-5 text-slate-400">Chaque version conserve une justification datée et le snapshot publié correspondant. Une nouvelle version n’efface pas les précédentes.</p>
          <form action={createPaperPolicyAction} className="space-y-3">
            <label className="block text-xs text-slate-400">Société
              <select name="company_id" required className={selectClass}><option value="">Choisir une société</option>
                {data.companies.filter(c=>c.currency==='EUR'&&c.quote_unit==='MAJOR').map(c=><option key={c.id} value={c.id}>{c.name} ({c.ticker})</option>)}
              </select></label>
            <div className="grid grid-cols-2 gap-3">
              <label className="block text-xs text-slate-400">Profil
                <select name="profile" defaultValue="VALUE" className={selectClass}>
                  <option value="CORE">CORE · Buy & Hold</option><option value="VALUE">VALUE</option>
                  <option value="CYCLICAL_VALUE">CYCLICAL VALUE</option><option value="TURNAROUND">TURNAROUND</option>
                </select></label>
              <label className="block text-xs text-slate-400">État de thèse
                <select name="thesis_state" defaultValue="ACTIVE" className={selectClass}>
                  <option value="ACTIVE">Active</option><option value="UNKNOWN">À vérifier</option><option value="INVALIDATED">Invalidée</option>
                </select></label>
              <Input label="Seuil d'achat € (facultatif)" name="entry_price_eur" type="number" placeholder="4.50"/>
              <Input label="Objectif de vente €" name="exit_target_eur" type="number" placeholder="12.00"/>
            </div>
            <label className="block text-xs text-slate-400">Jalon économique (CYCLICAL / TURNAROUND)
              <select name="milestone_state" defaultValue="UNCONFIRMED" className={selectClass}>
                <option value="UNCONFIRMED">Non confirmé</option><option value="CONFIRMED">Confirmé</option>
                <option value="NOT_APPLICABLE">Sans objet</option></select></label>
            <label className="block text-xs text-slate-400">Thèse et justification de la cible
              <textarea name="rationale" rows={3} required minLength={12} maxLength={2000}
                className={selectClass} placeholder="Pourquoi acheter, conserver ou vendre cette société ?"/></label>
            <button className={btn}>Enregistrer la stratégie</button>
          </form>
        </Panel>
        <Panel title="Stratégies suivies">
          {data.policies.length===0?<p className="text-sm text-slate-500">Aucune politique définie.</p>:
            <div className="space-y-3">{data.policies.map(p=><div key={p.id}
              className="rounded-xl border border-slate-800 bg-slate-950/70 p-4">
              <div className="flex justify-between gap-3">
                <div><p className="font-medium text-white">{names.get(p.company_id)?.name??p.company_id.slice(0,8)}</p>
                  <p className="mt-1 text-xs text-cyan-300">{p.profile} · V{p.version} · {p.thesis_state}</p></div>
                <span className="text-xs text-slate-400">{data.quotes.get(p.company_id)?eur(eurNumber(data.quotes.get(p.company_id)!.price)):'Cours indisponible'}</span>
              </div>
              <p className="mt-3 text-xs text-slate-400">{p.rationale}</p>
              {p.exit_target_eur!==null&&<p className="mt-2 text-xs text-slate-300">Cible : {eur(eurNumber(p.exit_target_eur))}</p>}
              <form action={evaluatePaperSignalAction} className="mt-3">
                <input type="hidden" name="policy_id" value={p.id}/>
                <button className="rounded-lg border border-cyan-700 px-3 py-2 text-xs text-cyan-200">Évaluer et historiser un signal</button>
              </form>
            </div>)}</div>}
        </Panel>
      </div>
      <Panel title="Signaux horodatés et exécution fictive">
        <p className="mb-4 text-xs text-slate-400">Le prix du signal n’est jamais un prix d’exécution. Il faut une nouvelle cotation enregistrée après le signal.</p>
        {data.signals.length===0?<p className="text-sm text-slate-500">Aucun signal enregistré.</p>:
          <div className="space-y-3">{data.signals.map(s=>{
            const q=data.quotes.get(s.company_id),done=filledIds.has(s.id);
            const newer=!!q&&Date.parse(q.as_of)>Date.parse(s.created_at)
              &&Date.parse(q.created_at)>Date.parse(s.created_at)&&!getFreshness(q.as_of).stale;
            const actionable=s.action==='BUY_CANDIDATE'||s.action==='SELL_CANDIDATE';
            return <div key={s.id} className="flex flex-col justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/50 p-4 lg:flex-row lg:items-center">
              <div><p className="text-sm font-semibold text-white">{names.get(s.company_id)?.name??s.company_id.slice(0,8)}
                <span className="ml-2 text-xs text-cyan-300">{s.action}</span></p>
                <p className="mt-2 text-xs text-slate-400">{s.reason} · {dateLabel(s.created_at)} · {eur(eurNumber(s.quote_price_eur))}</p></div>
              {done?<span className="text-xs text-emerald-300">Transaction simulée enregistrée</span>:
                actionable&&newer?<form action={executeVirtualFillAction} className="flex flex-wrap items-end gap-2">
                  <input type="hidden" name="signal_id" value={s.id}/>
                  <Input label="Titres virtuels" name="quantity" type="number" placeholder="10" required/>
                  <Input label="Frais €" name="fee_eur" type="number" placeholder="1"/>
                  <button className={btn}>Confirmer la simulation</button>
                </form>:<span className="text-xs text-amber-300">{actionable?'Attendre une nouvelle cotation':'Signal sans transaction'}</span>}
            </div>;})}</div>}
      </Panel>
      <Panel title="Transactions fictives">
        {data.fills.length===0?<p className="text-sm text-slate-500">Aucun achat ou vente simulé.</p>:
          <div className="overflow-x-auto"><table className="min-w-[600px] w-full text-left text-sm">
            <thead className="text-xs uppercase text-slate-500"><tr><th>Date</th><th>Société</th><th>Sens</th><th>Quantité</th><th>Prix</th><th>Frais</th><th>P&L réalisé</th></tr></thead>
            <tbody className="divide-y divide-slate-800">{data.fills.map(f=><tr key={f.id} className="text-slate-300">
              <td className="py-3">{dateLabel(f.filled_at)}</td><td>{names.get(f.company_id)?.name??f.company_id.slice(0,8)}</td>
              <td>{f.side}</td><td>{formatQty(eurNumber(f.quantity))}</td><td>{eur(eurNumber(f.price_eur))}</td>
              <td>{eur(eurNumber(f.fee_eur))}</td><td>{eur(eurNumber(f.realized_pnl_eur))}</td>
            </tr>)}</tbody></table></div>}
      </Panel>
    </>}
  </main>;
}
