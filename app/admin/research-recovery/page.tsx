import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {researchInboxPilotMode} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getRecoverySnapshot,listRecoveryJournal} from '@/lib/orotitan-equity/product/research-recovery-shadow-server';
import {ASML_Q2_2026_PUBLIC_RELEASE} from '@/lib/orotitan-equity/product/research-recovery-official-source-v1';
import {runResearchRecoveryAction} from './actions';
export const dynamic='force-dynamic';
export const metadata:Metadata={title:'Research Recovery Loop V1 — SHADOW'};
type Params=Promise<Record<string,string|string[]|undefined>>;
function single(x:string|string[]|undefined):string|null{return typeof x==='string'?x:null;}
export default async function RecoveryPage({searchParams}:{searchParams:Params}) {
  if(!(await isAdminAuthenticated()))redirect('/admin');
  const q=await searchParams;
  const source=single(q.source)??'asml-observation';
  const enabled=researchInboxPilotMode()==='shared-test' && isHostedShadowResearchEnabled(process.env);
  let snapshot:Awaited<ReturnType<typeof getRecoverySnapshot>>|null=null;
  let journal:Awaited<ReturnType<typeof listRecoveryJournal>>=[];
  let failed=false;
  if(enabled) {
    try {
      snapshot=await getRecoverySnapshot(source);
      journal=await listRecoveryJournal(snapshot.input);
    } catch {failed=true;}
  }
  const done=new Set(journal.map(a=>a.blockerCode));
  return <main className="space-y-6">
    <header className="space-y-3">
      <p className="text-xs font-semibold uppercase tracking-widest text-cyan-400">OroTitan · Product · SHADOW</p>
      <h1 className="text-3xl font-semibold text-white">Research Recovery Loop V1</h1>
      <p className="max-w-3xl text-sm leading-6 text-slate-400">
        Boucle de diagnostic, recherche de pièces déjà disponibles et journal de reprise durable.
        Elle n&apos;exécute aucun changement du Registry, ne certifie aucune source et ne publie aucune valorisation.
      </p>
    </header>
    <AdminNav/>
    {!enabled?<section role="alert" className="rounded-lg border border-amber-800 p-5 text-amber-200">
      Fonction désactivée. Uniquement disponible dans la Preview isolée du Supabase SHADOW.
    </section>:<>
      <div className="rounded-lg border border-amber-800/60 bg-amber-950/10 p-4 text-sm text-amber-100">
        <strong>Périmètre strict :</strong> journal de pré-résolution non autoritatif.
        L&apos;observation ASML est une copie datée en lecture seule de la production, pas un run SHADOW.
        Aucune tentative enregistrée ici ne lève les blocages officiels.
      </div>
      <section className="rounded-lg border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="text-lg font-semibold text-white">Sélectionner le dossier</h2>
        <p className="mt-2 text-sm text-slate-400">
          <Link className="text-cyan-300 underline" href="/admin/research-recovery?source=asml-observation">
            Observation ASML — quatre blocages réels du 30/10
          </Link>. Vous pouvez aussi inspecter un UUID de run existant dans le projet SHADOW.
        </p>
        <form action="/admin/research-recovery" method="GET" className="mt-4 flex flex-wrap gap-3">
          <input type="text" name="source" aria-label="UUID de run SHADOW"
            placeholder="UUID du run SHADOW" className="min-w-[240px] flex-1 rounded-lg border border-slate-700 bg-slate-950 px-4 py-2 text-sm text-white"/>
          <button className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-white">Ouvrir</button>
        </form>
      </section>
      {(failed||single(q.error))&&<p role="alert" className="rounded-lg border border-rose-800 p-4 text-sm text-rose-200">
        Lecture ou sauvegarde du journal impossible. Aucune conclusion ni autorisation n&apos;a été créée.
      </p>}
      {snapshot&&<section className="space-y-5 rounded-lg border border-slate-800 bg-slate-900/55 p-5">
        <div>
          <h2 className="text-xl font-semibold text-white">
            {snapshot.origin==='ASML_READ_ONLY_MIRROR'?'ASML · observation miroir':'Run SHADOW'}
          </h2>
          <p className="mt-2 break-all font-mono text-xs text-slate-400">{snapshot.input.runId}</p>
          <p className="mt-1 text-xs text-slate-400">
            {snapshot.input.stageCode} · {snapshot.input.runStatus} · version run {snapshot.input.runVersion}
            · version étape {snapshot.input.stageVersion}
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          {[
            ['Blocages observés',snapshot.input.blockers.length],
            ['Tentatives persistées',journal.length],
            ['Blocages avec tentative',done.size],
          ].map(([label,value])=><div key={String(label)} className="rounded-md bg-slate-950/70 p-4">
            <p className="text-xs text-slate-400">{label}</p>
            <p className="mt-1 text-xl font-semibold text-white">{value}</p>
          </div>)}
        </div>
        <form action={runResearchRecoveryAction} className="flex flex-wrap items-center gap-4">
          <input type="hidden" name="source" value={source}/>
          <button className="rounded-lg bg-cyan-400 px-5 py-3 font-semibold text-slate-950 hover:bg-cyan-300">
            Lancer / reprendre un cycle borné
          </button>
          <p className="text-xs text-slate-400">Maximum quatre nouvelles tentatives par clic. Reprise sans recommencer les mêmes.</p>
        </form>
        {single(q.result)&&<p role="status" className="rounded-lg border border-cyan-900 p-3 text-sm text-cyan-100">
          Cycle terminé : {single(q.result)}. Les blocages officiels restent inchangés.
        </p>}
        <div className="space-y-3">
          {snapshot.input.blockers.map(b=>{
            const attempts=journal.filter(a=>a.blockerCode===b.code);
            return <article key={b.code} className="rounded-lg border border-slate-800 bg-slate-950/50 p-4">
              <p className="font-mono text-sm text-amber-300">{b.code}</p>
              <p className="mt-1 text-sm text-slate-200">{b.reason}</p>
              {attempts.length===0?<p className="mt-2 text-xs text-slate-500">Non encore étudié par le loop.</p>:
                attempts.map(a=><div key={a.strategy} className="mt-3 border-t border-slate-800 pt-3">
                  <p className="text-xs font-medium text-cyan-200">{a.strategy} · {a.outcome}</p>
                  <p className="mt-1 text-sm text-slate-300">{a.detail}</p>
                  {a.evidenceRefs.length>0&&<div className="mt-2 break-all font-mono text-xs text-slate-400">
                    <span>Candidats non admis : </span>
                    {a.evidenceRefs.map((ref,i)=><span key={ref}>
                      {i>0?' · ':''}
                      {ref===ASML_Q2_2026_PUBLIC_RELEASE.url
                        ? <a className="text-cyan-300 underline" href={ref} target="_blank" rel="noopener noreferrer">Communiqué officiel ASML Q2 2026</a>
                        : ref}
                    </span>)}
                  </div>}
                </div>)}
            </article>;
          })}
        </div>
      </section>}
    </>}
  </main>;
}
