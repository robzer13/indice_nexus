import { readPaperAutomationSettings } from '@/lib/automation/paper-daily';
import { configurePaperAutopilotAction } from '@/app/admin/paper/automation-actions';

export async function PaperAutopilotSettings({portfolioId}:{portfolioId:string}){
  const setting=await readPaperAutomationSettings(portfolioId);
  if(setting==='MIGRATION_NOT_INSTALLED')return <section className="rounded-xl border border-amber-900 bg-slate-900/60 p-4">
    <h2 className="text-lg font-semibold text-white">Autopilote V1.2 en attente</h2>
    <p className="mt-2 text-sm text-amber-200">Migration V1.2 absente : aucune automatisation possible.</p>
  </section>;
  const enabled=setting?.enabled===true;
  return <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
    <div className="flex flex-wrap items-center justify-between gap-4">
      <div><h2 className="text-lg font-semibold text-white">Autopilote fictif V1.2</h2>
        <p className="mt-1 text-sm text-slate-400">Cours de clôture vérifiés uniquement · 5 % du capital initial maximum par nouveau titre · 1 € de frais fictifs.</p>
        <p className="mt-2 text-xs text-slate-500">Activé seulement avec ton consentement. Sans transactions réelles ni accès courtier.
          Les cours non vérifiés, USD, fractions de titres et thèses périmées sont bloqués.</p>
      </div>
      <div><span className={enabled?'text-emerald-300':'text-amber-300'}>{enabled?'ACTIF':'DÉSACTIVÉ'}</span>
        <form action={configurePaperAutopilotAction} className="mt-2">
          <input type="hidden" name="enabled" value={enabled?'false':'true'}/>
          <button className="rounded-lg border border-cyan-700 px-4 py-2 text-xs text-cyan-200">
            {enabled?'Désactiver':'Autoriser les opérations virtuelles automatiques'}
          </button>
        </form>
      </div>
    </div>
  </section>;
}
