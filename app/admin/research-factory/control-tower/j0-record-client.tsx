'use client';
import {useState} from 'react';
import {useRouter} from 'next/navigation';

type EventType='EVIDENCE_RECORDED'|'BLOCKER_OPENED'|
 'BLOCKER_RESOLVED'|'GATE_REVIEW_REQUESTED';
type Receipt={ok:true;eventId:string;status:'CREATED'|'REPLAYED';
 verificationState:string;persistedReadbackVerified:true;accepted:false};
const types:{value:EventType;label:string}[]=[
 {value:'EVIDENCE_RECORDED',label:'Déposer une preuve (non acceptante)'},
 {value:'BLOCKER_OPENED',label:'Ouvrir un blocage'},
 {value:'BLOCKER_RESOLVED',label:'Résoudre un blocage'},
 {value:'GATE_REVIEW_REQUESTED',label:'Demander la revue du gate'},
];
export function J0RecordClient({runIds}:{runIds:readonly string[]}){
 const router=useRouter();
 const [milestone,setMilestone]=useState('J0');
 const [type,setType]=useState<EventType>('EVIDENCE_RECORDED');
 const [title,setTitle]=useState('');
 const [reference,setReference]=useState('');
 const [runId,setRunId]=useState('');
 const [details,setDetails]=useState('');
 const [busy,setBusy]=useState(false);
 const [receipt,setReceipt]=useState<Receipt|null>(null);
 const [error,setError]=useState('');
 async function record(){
  if(busy)return;
  setBusy(true);setError('');setReceipt(null);
  try{
   const eventKey='orotitan:j0:ui-'+crypto.randomUUID();
   const res=await fetch('/admin/research-factory/control-tower/record',{
    method:'POST',cache:'no-store',credentials:'same-origin',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
     eventKey,milestoneCode:milestone,eventType:type,
     title:title.trim(),evidenceRef:reference.trim(),
     runId:runId||null,details:{comment:details.trim(),
       provenance:'ADMIN_MANUAL_REPORT_ONLY'},
    }),
   });
   const json:unknown=await res.json();
   if(!res.ok||!json||typeof json!=='object'||
      !('ok' in json)||json.ok!==true)
    throw Error(json&&typeof json==='object'&&'error' in json?
     String(json.error):'J0 : écriture non confirmée.');
   const parsed=json as Receipt;
   if(parsed.accepted!==false||parsed.persistedReadbackVerified!==true)
    throw Error('Reçu incohérent, vérifier le journal dans le Registry.');
   setReceipt(parsed);router.refresh();
  }catch(e){setError(e instanceof Error?e.message:'Échec du journal.');}
  finally{setBusy(false);}
 }
 return <section className="rounded-xl border border-cyan-700 bg-slate-950 p-5 space-y-4">
  <div>
   <h2 className="text-xl font-semibold text-white">Journal opérationnel J0</h2>
   <p className="text-sm text-slate-300 mt-1">
    Enregistre des faits et exceptions, pas une approbation automatique.
    Un lien transmis sans relecture du Registry reste une preuve déclarée.
   </p>
  </div>
  <div className="grid gap-3 md:grid-cols-2">
   <label className="space-y-1 text-sm">
    <span>Jalon</span>
    <select value={milestone} disabled={busy}
     onChange={e=>setMilestone(e.target.value)}
     className="w-full rounded border border-slate-600 bg-slate-900 p-2">
     {Array.from({length:9},(_,i)=><option key={i} value={'J'+i}>J{i}</option>)}
    </select>
   </label>
   <label className="space-y-1 text-sm">
    <span>Type d’événement</span>
    <select value={type} disabled={busy}
     onChange={e=>setType(e.target.value as EventType)}
     className="w-full rounded border border-slate-600 bg-slate-900 p-2">
     {types.map(item=><option key={item.value} value={item.value}>
      {item.label}</option>)}
    </select>
   </label>
   <label className="space-y-1 text-sm">
    <span>Titre descriptif</span>
    <input value={title} disabled={busy} maxLength={200}
     onChange={e=>setTitle(e.target.value)}
     placeholder="Relecture Registry validée"
     className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
   </label>
   <label className="space-y-1 text-sm">
    <span>Référence de preuve / identifiant d’incident</span>
    <input value={reference} disabled={busy} maxLength={400}
     onChange={e=>setReference(e.target.value)}
     placeholder={type.startsWith('BLOCKER')?'issue:j1-http-e2e':
      'registry:run:<UUID> ou URL de commit'}
     className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
   </label>
   <label className="space-y-1 text-sm">
    <span>Run (facultatif)</span>
    <select value={runId} disabled={busy}
     onChange={e=>setRunId(e.target.value)}
     className="w-full rounded border border-slate-600 bg-slate-900 p-2">
     <option value="">Aucun</option>
     {runIds.map(id=><option key={id} value={id}>{id}</option>)}
    </select>
   </label>
   <label className="space-y-1 text-sm">
    <span>Commentaire / limitation</span>
    <input value={details} disabled={busy} maxLength={1000}
     onChange={e=>setDetails(e.target.value)}
     placeholder="Nature exacte de la preuve ou anomalie"
     className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
   </label>
  </div>
  <button type="button" disabled={busy||title.trim().length<5||
     reference.trim().length<8} onClick={()=>void record()}
   className="rounded bg-cyan-400 px-4 py-2 font-semibold text-slate-950 disabled:opacity-50">
   {busy?'Écriture et relecture en cours…':'Consigner et relire le reçu'}
  </button>
  {error&&<p role="alert" className="text-rose-300 text-sm">{error}</p>}
  {receipt&&<p className="text-emerald-300 text-sm break-all">
   {receipt.status} — {receipt.eventId} — {receipt.verificationState}.
   Relecture persistante confirmée. Aucune acceptation de jalon.
  </p>}
 </section>;
}
