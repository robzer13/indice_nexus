import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from
 '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {J0RecordClient} from './j0-record-client';
import {J0C5SecurityTestClient} from './j0-c5-security-test-client';
import {J0C8GatePanel} from './j0-c8-gate-panel';
import {unresolvedJ0Issues,j0DisplayedStatus} from
 '@/lib/orotitan-equity/product/runfactory-j0-projections-v1';

export const metadata:Metadata={title:'OroTitan · J0 · Control Tower SHADOW'};
export const dynamic='force-dynamic';

type Milestone={milestone_code:string;ordinal:number;title:string;
 objective:string;initial_status:string;gate_criteria:unknown;};
type Event={event_id:string;event_seq:number;milestone_code:string;event_type:string;
 title:string;evidence_ref:string|null;run_id:string|null;
 verification_state:string;operator_details:Record<string,unknown>;created_at:string;};
type Run={run_id:string;issuer_id:string|null;security_id:string|null;
 dossier_id:string|null;run_status:string;current_stage:string|null;
 run_type:string|null;state_version:number;data_cutoff:string;
 contract_set_sha256:string;created_at:string;};
type Issuer={issuer_id:string;display_name:string};

const label:Record<string,string>={
 NON_DEMARRE:'NON DÉMARRÉ',EN_COURS:'EN COURS',
 BLOQUE:'BLOQUÉ',TESTE:'TESTÉ',ACCEPTE:'ACCEPTÉ',
};
export default async function J0ControlTowerPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedShadowResearchEnabled(process.env))
  return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 const client=getHostedShadowRecoveryClient();
 const [milestonesQ,eventsQ,runsQ,issuersQ]=await Promise.all([
  client.from('orotitan_j0_milestones_v1')
   .select('milestone_code,ordinal,title,objective,initial_status,gate_criteria')
   .order('ordinal',{ascending:true}).limit(10),
  client.from('orotitan_j0_events_v1')
   .select('event_id,event_seq,milestone_code,event_type,title,evidence_ref,run_id,verification_state,operator_details,created_at')
   .order('event_seq',{ascending:false}).limit(250),
  client.from('orotitan_runs')
   .select('run_id,issuer_id,security_id,dossier_id,run_status,current_stage,run_type,state_version,data_cutoff,contract_set_sha256,created_at')
   .order('created_at',{ascending:false}).limit(200),
  client.from('issuers').select('issuer_id,display_name').limit(200),
 ]);
 if(milestonesQ.error||eventsQ.error||runsQ.error||issuersQ.error||
    !Array.isArray(milestonesQ.data)||!Array.isArray(eventsQ.data)||
    !Array.isArray(runsQ.data)||!Array.isArray(issuersQ.data)||
    milestonesQ.data.length!==9||
    eventsQ.data.length>=250||runsQ.data.length>=200||
    issuersQ.data.length>=200)
  return <main className="p-6 space-y-4">
   <h1 className="text-2xl text-white">J0 Control Tower — accès incomplet</h1>
   <p className="text-rose-300">
    Registry SHADOW indisponible, structure incomplète ou seuil de lecture atteint.
    Aucun état ne sera présenté comme confirmé.
   </p>
   <AdminNav/>
  </main>;
 const milestones=milestonesQ.data as Milestone[];
 const events=eventsQ.data as Event[];
 const runs=runsQ.data as Run[];
 const issuerMap=new Map((issuersQ.data as Issuer[])
  .map(i=>[i.issuer_id,i.display_name]));
 const verified=events.filter(e=>
  e.verification_state==='VERIFIED_DIRECT_REGISTRY_READBACK').length;
 const unresolved=milestones.flatMap(m=>unresolvedJ0Issues(events,m.milestone_code));
 const accepted=milestones.filter(m=>m.initial_status==='ACCEPTE').length;
 return <main className="mx-auto max-w-7xl space-y-6 p-5 text-slate-100">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan Run Factory · J0 · SHADOW ONLY
   </p>
   <h1 className="text-3xl font-bold">Control Tower — état réel et preuves</h1>
   <p className="text-slate-300">
    Vue relue directement dans le Registry. Les preuves déclarées sont distinctes
    des relectures indépendantes. Aucun jalon n’est automatiquement accepté.
   </p>
  </header>
  <AdminNav/>
  <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
   {[
    ['Jalons',String(milestones.length)],
    ['Jalons acceptés',String(accepted)],
    ['Runs Registry',String(runs.length)],
    ['Preuves directes / blocages',verified+' / '+unresolved.length],
   ].map(([name,value])=><div key={name}
     className="rounded-lg border border-slate-700 bg-slate-900 p-4">
     <p className="text-sm text-slate-400">{name}</p>
     <p className="mt-1 text-2xl font-bold">{value}</p>
    </div>)}
  </section>
  <section className="rounded-lg border border-amber-700 bg-slate-900 p-4 space-y-2">
   <h2 className="font-bold text-amber-200">
    ACTION : J0 requiert une démonstration et une acceptation distinctes
   </h2>
   <p className="text-sm text-slate-300">
    Le GO J0 autorise le pilotage, pas l’acceptation du gate ni la transition
    vers J1/J2. J1 est partiellement développé mais reste non accepté.
    Ni stage Research, ni snapshot, ni publication ne sont déclenchés ici.
   </p>
  </section>
  <section className="space-y-3">
   <h2 className="text-xl font-bold">Registre des jalons et critères de passage</h2>
   <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
    {milestones.map(m=>{
     const blockers=unresolvedJ0Issues(events,m.milestone_code);
     const observations=events.filter(e=>e.milestone_code===m.milestone_code);
     const status=j0DisplayedStatus({initialStatus:m.initial_status,unresolvedBlockerCount:blockers.length});
     const criteria=Array.isArray(m.gate_criteria)?
      m.gate_criteria.filter((x):x is string=>typeof x==='string'):[];
     return <article key={m.milestone_code}
      className="rounded-xl border border-slate-700 bg-slate-900 p-4 space-y-3">
      <div className="flex flex-wrap justify-between gap-2">
       <h3 className="font-semibold text-white">{m.milestone_code} · {m.title}</h3>
       <span className={'rounded px-2 py-1 text-xs font-bold '+
        (status==='BLOQUE'?'bg-rose-950 text-rose-300':
         status==='ACCEPTE'?'bg-emerald-950 text-emerald-300':
         'bg-slate-800 text-amber-200')}>{label[status]??status}</span>
      </div>
      <p className="text-sm text-slate-300">{m.objective}</p>
      <p className="text-xs text-slate-400">
       {observations.length} événements · {blockers.length} blocages ouverts
      </p>
      <details className="text-sm text-slate-300">
       <summary className="cursor-pointer text-cyan-300">Critères d’acceptation ({criteria.length})</summary>
       <ol className="mt-2 list-decimal space-y-2 pl-5">
        {criteria.map((item,i)=>{
         const proofs=observations.filter(e=>
          e.event_type==='EVIDENCE_RECORDED'&&
          e.operator_details?.criterionIndex===i+1);
         const direct=proofs.filter(e=>
          e.verification_state==='VERIFIED_DIRECT_REGISTRY_READBACK').length;
         return <li key={i}>
          <span>{item}</span>
          <span className="ml-2 text-xs text-slate-400">
           {proofs.length===0?'Aucune preuve rattachée':
            proofs.length+' preuves, '+direct+' relectures directes'}
          </span>
         </li>;
        })}
       </ol>
      </details>
      {blockers.map(issue=><p key={issue.event_id}
       className="rounded border border-rose-800 p-2 text-xs text-rose-200">
       BLOCAGE : {issue.title} · {issue.evidence_ref}
      </p>)}
     </article>;
    })}
   </div>
  </section>
  <section className="space-y-3">
   <h2 className="text-xl font-bold">Runs — lecture Registry en direct</h2>
   <div className="overflow-x-auto rounded-xl border border-slate-700">
    <table className="w-full min-w-[800px] text-left text-sm">
     <thead className="bg-slate-900 text-slate-300">
      <tr><th className="p-3">Émetteur / Run</th>
       <th className="p-3">Statut</th><th className="p-3">Stage</th>
       <th className="p-3">Cutoff</th><th className="p-3">Contrats</th></tr>
     </thead>
     <tbody>{runs.map(r=><tr key={r.run_id} className="border-t border-slate-800">
      <td className="p-3">
       <p className="font-semibold">{issuerMap.get(r.issuer_id??'')??'Émetteur non résolu'}</p>
       <code className="text-xs text-slate-400">{r.run_id}</code>
      </td>
      <td className="p-3">{r.run_status} · v{r.state_version}</td>
      <td className="p-3">{r.current_stage??'AUCUN'}</td>
      <td className="p-3">{r.data_cutoff}</td>
      <td className="p-3"><code className="text-xs">{r.contract_set_sha256.slice(0,12)}…</code></td>
     </tr>)}</tbody>
    </table>
   </div>
  </section>
  <section className="space-y-3">
   <h2 className="text-xl font-bold">Journal horodaté, preuves et décisions</h2>
   <div className="overflow-x-auto rounded-xl border border-slate-700">
    <table className="w-full min-w-[760px] text-left text-sm">
     <thead className="bg-slate-900 text-slate-300">
      <tr><th className="p-3">Date UTC</th><th className="p-3">Jalon</th>
       <th className="p-3">Événement</th><th className="p-3">Référence</th>
       <th className="p-3">Preuve</th></tr>
     </thead>
     <tbody>{events.map(e=><tr key={e.event_id} className="border-t border-slate-800">
      <td className="p-3 whitespace-nowrap">{new Date(e.created_at).toISOString().slice(0,19)}</td>
      <td className="p-3">{e.milestone_code}</td>
      <td className="p-3"><p className="font-medium">{e.title}</p>
       <p className="text-xs text-slate-400">{e.event_type}</p></td>
      <td className="p-3 break-all text-xs">{e.evidence_ref??'—'}</td>
      <td className="p-3 text-xs">{e.verification_state==='VERIFIED_DIRECT_REGISTRY_READBACK'?
       'RELECTURE REGISTRY':'DÉCLARÉE'}</td>
     </tr>)}</tbody>
    </table>
   </div>
  </section>
  <J0RecordClient runIds={runs.map(r=>r.run_id)}
   criteria={Object.fromEntries(milestones.map(m=>[
    m.milestone_code,Array.isArray(m.gate_criteria)?
     m.gate_criteria.filter((x):x is string=>typeof x==='string'):[],
   ]))}/>
  <J0C5SecurityTestClient/>
  <J0C8GatePanel proofs={events.filter(e=>e.milestone_code==='J0'&&
   e.event_type==='EVIDENCE_RECORDED'&&
   typeof e.operator_details?.criterionIndex==='number').map(e=>({
    eventId:e.event_id,title:e.title,
    criterionIndex:e.operator_details.criterionIndex as number,
    verified:e.verification_state==='VERIFIED_DIRECT_REGISTRY_READBACK',
   }))}/>
  <p className="text-sm text-slate-400">
   Portée : SHADOW uniquement. Le champ « initial_status » est la position
   d’ouverture du programme, non une certification d’avancement calculée.
   Les preuves enregistrées sont append-only et les portes d’acceptation
   restent explicitement séparées.
  </p>
  <footer className="flex gap-6 text-sm text-cyan-300">
   <Link href="/admin/research-factory/run-preflight">
    Run Factory P0-B/P0-E
   </Link>
   <Link href="/admin/research-factory/contracts">Contrats figés</Link>
  </footer>
 </main>;
}
