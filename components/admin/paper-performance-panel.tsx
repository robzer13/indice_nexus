import { createServerSupabaseClient } from '@/lib/supabase/server';
import { maximumDrawdown } from '@/lib/domain/paper-performance';

type Point={valuation_day:string;nav_eur:number|string;};
const money=(n:number)=>new Intl.NumberFormat('fr-FR',{style:'currency',currency:'EUR',maximumFractionDigits:2}).format(n);
export async function PaperPerformancePanel({portfolioId}:{portfolioId:string}){
  const {data,error}=await createServerSupabaseClient().from('paper_daily_nav')
    .select('valuation_day,nav_eur').eq('portfolio_id',portfolioId)
    .order('valuation_day',{ascending:false}).limit(365);
  if(error?.code==='42P01'||error?.code==='PGRST205')return <p className="text-xs text-amber-200">Historique NAV en attente de migration V1.2.</p>;
  if(error)throw new Error('Historique de valeur impossible : '+error.message);
  const rows=((data??[]) as Point[]).reverse();
  if(rows.length===0)return <p className="text-sm text-slate-500">Premier relevé de valeur en attente de clôture admissible.</p>;
  const values=rows.map(x=>Number(x.nav_eur));
  const dd=maximumDrawdown(values);
  const start=values[0],end=values[values.length-1];
  return <div className="grid gap-3 sm:grid-cols-3">
    <div><p className="text-xs text-slate-500">Première valeur enregistrée</p>
      <p className="mt-1 font-semibold text-white">{money(start)}</p><p className="text-xs text-slate-500">{rows[0].valuation_day}</p></div>
    <div><p className="text-xs text-slate-500">Dernière valeur</p>
      <p className="mt-1 font-semibold text-white">{money(end)}</p><p className="text-xs text-slate-500">{rows[rows.length-1].valuation_day}</p></div>
    <div><p className="text-xs text-slate-500">Drawdown observé</p>
      <p className="mt-1 font-semibold text-white">{dd===null?'—':(dd*100).toFixed(2)+' %'}</p>
      <p className="text-xs text-slate-500">{rows.length} séance(s) documentée(s)</p></div>
    <p className="text-xs text-slate-500 sm:col-span-3">Valeur de marché en euros, sans dividendes ni benchmark tant que leurs séries vérifiées ne sont pas disponibles. Le drawdown sur peu de séances est peu représentatif.</p>
  </div>;
}
