'use client';
import {useState} from 'react';
import {useRouter} from 'next/navigation';
type Case={name:string;status:number;error:string};
type Success=Readonly<{ok:true;applicationHttpMatrixVerified:true;
 cases:readonly Case[];eventId:string;persistedReadbackVerified:true;
 evidenceState:'REPORTED';accepted:false;stageStarted:false;
 published:false;blockerClosure:'REQUIRES_INDEPENDENT_REVIEW';
 scope:'SHADOW_ONLY';}>;
export function J0C5SecurityTestClient(){
 const router=useRouter();
 const [running,setRunning]=useState(false);
 const [error,setError]=useState('');
 const [result,setResult]=useState<Success|null>(null);
 async function execute(){
  if(running)return;
  setRunning(true);setError('');setResult(null);
  try{
   const response=await fetch(
    '/admin/research-factory/control-tower/c5-security-test',{
     method:'POST',credentials:'same-origin',cache:'no-store',
     headers:{'Content-Type':'application/json'},body:'{}',
    });
   const raw:unknown=await response.json();
   if(!response.ok||!raw||typeof raw!=='object'||Array.isArray(raw)||
      !('ok' in raw)||raw.ok!==true){
    const message=raw&&typeof raw==='object'&&'error' in raw?
     String(raw.error):'C5 : aucune vérification concluante.';
    throw Error(message);
   }
   const proof=raw as Success;
   if(proof.applicationHttpMatrixVerified!==true||
      proof.persistedReadbackVerified!==true||
      proof.cases?.length!==4||proof.accepted!==false||
      proof.stageStarted!==false||proof.published!==false||
      proof.blockerClosure!=='REQUIRES_INDEPENDENT_REVIEW'||
      proof.scope!=='SHADOW_ONLY')
    throw Error('J0_C5_INCOMPLETE_RECEIPT');
   setResult(proof);
   router.refresh();
  }catch(e){setError(e instanceof Error?e.message:'C5 : erreur de contrôle.');}
  finally{setRunning(false);}
 }
 return <section className="space-y-3 rounded-xl border border-amber-600 bg-slate-950 p-5">
  <h2 className="text-xl font-bold uppercase text-amber-200">
   ACTION ADMINISTRATEUR — CONTRÔLE C5
  </h2>
  <p className="text-sm text-slate-200">
   Depuis cette session authentifiée, exécuter quatre refus HTTP sur
   la même Preview protégée par Vercel SSO. Le système vérifie qu’aucun
   événement imprévu, stage, artefact ou snapshot n’a été créé.
   Aucun mot de passe ou jeton n’est affiché.
  </p>
  <p className="text-sm text-slate-400">
   Le contrôle crée uniquement un reçu d’essai dans le journal J0
   si les quatre refus sont confirmés. Il n’accepte pas J0 et ne clôt
   pas automatiquement l’incident C5.
  </p>
  <button type="button" disabled={running} onClick={()=>void execute()}
   className="rounded bg-amber-400 px-4 py-3 font-bold text-slate-950 disabled:opacity-50">
   {running?'Tests de sécurité et relecture en cours…':
    'EXÉCUTER LE CONTRÔLE HTTP C5'}
  </button>
  {error&&<p role="alert" className="rounded border border-rose-700 p-3 text-sm text-rose-200">
   NON VÉRIFIÉ — {error}. Le blocage reste ouvert.
  </p>}
  {result&&<div className="space-y-2 rounded border border-emerald-700 p-3 text-sm">
   <p className="font-bold text-emerald-200">
    QUATRE REFUS HTTP APPLICATION CONFIRMÉS — REÇU PERSISTANT
   </p>
   <p className="break-all font-mono text-emerald-200">{result.eventId}</p>
   <ul className="space-y-1 text-slate-200">
    {result.cases.map(item=><li key={item.name}>
     {item.name} : {item.status} — {item.error}
    </li>)}
   </ul>
   <p className="text-slate-300">
    J0 non accepté. Relecture indépendante nécessaire pour clôturer C5.
   </p>
  </div>}
 </section>;
}
