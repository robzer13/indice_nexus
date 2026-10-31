'use client';
import {useRef,useState} from 'react';
import {useRouter} from 'next/navigation';

type Proof=Readonly<{eventId:string;title:string;criterionIndex:number;
 verified:boolean}>;
type Gate=Readonly<{
 eligibleForReview:boolean;missingEvidenceIndices:readonly number[];
 openBlockers:number;journalWatermark:number;alreadyAccepted:boolean;
 acceptanceEnabled:boolean;acceptanceAllowedNow:boolean;
 preparedReviews:number;
}>;
type GateResponse=Readonly<{
 ok:true;scope:'SHADOW_ONLY';gate:Gate;
 activationLocked:boolean;preparedReviews:number;
 persistedDecisions:number;accepted:boolean;humanGoRequired:true;
}>;
type ReviewReceipt=Readonly<{ok:true;status:string;reviewId:string;
 reviewSha256:string;persistedReadbackVerified:true;accepted:false}>;
export function J0C8GatePanel({proofs}:{proofs:readonly Proof[]}){
 const router=useRouter();
 const reviewKeyRef=useRef<string|null>(null);
 const decisionKeyRef=useRef<string|null>(null);
 const [loading,setLoading]=useState(false);
 const [gate,setGate]=useState<GateResponse|null>(null);
 const [error,setError]=useState('');
 const [evidenceIds,setEvidenceIds]=useState<string[]>(Array(7).fill(''));
 const [auditRef,setAuditRef]=useState('');
 const [auditSha,setAuditSha]=useState('');
 const [reviewer,setReviewer]=useState('');
 const [receipt,setReceipt]=useState<ReviewReceipt|null>(null);
 const [approvalPhrase,setApprovalPhrase]=useState('');
 const [approvalRef,setApprovalRef]=useState('');
 const [approver,setApprover]=useState('');
 const [decisionResult,setDecisionResult]=useState('');
 async function inspect(){
  setLoading(true);setError('');
  try{
   const response=await fetch(
    '/admin/research-factory/control-tower/c8-gate?milestone=J0',
    {cache:'no-store',credentials:'same-origin'});
   const v:unknown=await response.json();
   if(!response.ok||!v||typeof v!=='object'||!('ok' in v)||v.ok!==true)
    throw Error(v&&typeof v==='object'&&'error' in v?String(v.error):
     'C8 : preflight indisponible');
   setGate(v as GateResponse);
  }catch(e){setGate(null);
   setError(e instanceof Error?e.message:'C8 : échec réseau');}
  finally{setLoading(false);}
 }
 async function prepare(){
  if(loading||!gate?.gate.eligibleForReview)return;
  if(!reviewKeyRef.current)reviewKeyRef.current=
    'orotitan:c8:review:'+crypto.randomUUID();
  setLoading(true);setError('');setReceipt(null);
  try{
   const r=await fetch('/admin/research-factory/control-tower/c8-gate/prepare',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
     reviewKey:reviewKeyRef.current,
     milestone:'J0',auditRef:auditRef.trim(),auditSha256:auditSha.trim(),
     reviewerLabel:reviewer.trim(),evidenceEventIds:evidenceIds,
    }),
   });
   const v:unknown=await r.json();
   if(!r.ok||!v||typeof v!=='object'||!('ok' in v)||v.ok!==true)
    throw Error(v&&typeof v==='object'&&'error' in v?String(v.error):
     'Préparation refusée');
   const entry=v as ReviewReceipt;
   if(entry.accepted!==false||entry.persistedReadbackVerified!==true)
    throw Error('Reçu de préparation invalide');
   setReceipt(entry);router.refresh();
  }catch(e){setError(e instanceof Error?e.message:'Préparation impossible');}
  finally{setLoading(false);}
 }
 async function accept(){
  // This button is absent unless the live database gate reports activation.
  if(!receipt||!gate?.gate.acceptanceAllowedNow||
     approvalPhrase!=='ACCEPTER J0 SHADOW'||loading)return;
  if(!decisionKeyRef.current)decisionKeyRef.current=
    'orotitan:c8:decision:'+crypto.randomUUID();
  setLoading(true);setError('');setDecisionResult('');
  try{
   const r=await fetch('/admin/research-factory/control-tower/c8-gate/accept',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
     decisionKey:decisionKeyRef.current,
     reviewId:receipt.reviewId,reviewSha256:receipt.reviewSha256,
     approverLabel:approver.trim(),consentPhrase:approvalPhrase,
     authorityRef:approvalRef.trim(),
    }),
   });
   const v:unknown=await r.json();
   if(!r.ok||!v||typeof v!=='object'||!('ok' in v)||v.ok!==true)
    throw Error(v&&typeof v==='object'&&'error' in v?String(v.error):
     'Acceptation refusée');
   const t=v as {decisionId?:string;accepted?:boolean;persistedReadbackVerified?:boolean};
   if(t.accepted!==true||t.persistedReadbackVerified!==true)
    throw Error('Décision sans reçu persistant valide');
   setDecisionResult('Décision persistée : '+t.decisionId);
   router.refresh();
  }catch(e){setError(e instanceof Error?e.message:'Acceptation impossible');}
  finally{setLoading(false);}
 }
 return <section className="rounded-xl border border-amber-700 bg-slate-950 p-5 space-y-4">
  <h2 className="text-xl font-bold text-white">
   C8 · Acceptation humaine des jalons — SHADOW
  </h2>
  <p className="text-sm text-slate-300">
   Circuit en deux actes : dossier de revue figé puis décision distincte.
   Le code d’acceptation est délibérément désactivé en l’absence
   d’une nouvelle autorisation explicite. Aucun stage n’est démarré.
  </p>
  <button type="button" onClick={()=>void inspect()} disabled={loading}
   className="rounded border border-cyan-500 px-4 py-2 text-cyan-200 disabled:opacity-50">
   {loading?'Contrôle en cours…':'Vérifier les 8 critères et le verrou'}
  </button>
  {gate&&<div className="space-y-1 rounded border border-slate-700 p-3 text-sm">
   <p>J0 — préparation : <strong>{gate.gate.eligibleForReview?'ADMISSIBLE':'BLOQUÉE'}</strong></p>
   <p>Incidents ouverts : {gate.gate.openBlockers}</p>
   <p>Critères 1 à 7 sans preuve conforme : {gate.gate.missingEvidenceIndices.length?
    gate.gate.missingEvidenceIndices.join(', '):'aucun'}</p>
   <p>Verrou d’acceptation : <strong>{gate.activationLocked?'FERMÉ':'ACTIVÉ PAR GO DISTINCT'}</strong></p>
   <p>Revues persistées : {gate.preparedReviews} · Décisions acceptées : {gate.persistedDecisions}</p>
   <p>Dernier événement journal : #{gate.gate.journalWatermark}</p>
  </div>}
  {gate?.gate.eligibleForReview&&!gate.accepted&&<div className="space-y-3">
   <h3 className="font-bold">Préparer un dossier de revue (aucune acceptation)</h3>
   <p className="text-xs text-slate-400">
    Sélectionne une preuve par critère C1–C7. Les preuves C1/C6/C7 de J0
    doivent avoir été relues directement dans le Registry.
   </p>
   <div className="grid gap-2 md:grid-cols-2">
    {Array.from({length:7},(_,i)=>{
     const options=proofs.filter(p=>p.criterionIndex===i+1 &&
      (![1,6,7].includes(i+1)||p.verified));
     return <label key={i} className="space-y-1 text-sm">
      <span>C{i+1} · Preuve archivée</span>
      <select value={evidenceIds[i]} disabled={loading}
       onChange={e=>setEvidenceIds(old=>old.map((v,j)=>j===i?e.target.value:v))}
       className="w-full rounded border border-slate-600 bg-slate-900 p-2">
       <option value="">Sélectionner</option>
       {options.map(p=><option key={p.eventId} value={p.eventId}>
        {p.title.slice(0,65)} · {p.eventId.slice(0,8)}</option>)}
      </select>
     </label>;
    })}
    <label className="space-y-1 text-sm">
     <span>Rapport d’audit figé — URL GitHub sur commit</span>
     <input value={auditRef} onChange={e=>setAuditRef(e.target.value)}
      className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
    </label>
    <label className="space-y-1 text-sm">
     <span>SHA-256 du rapport</span>
     <input value={auditSha} onChange={e=>setAuditSha(e.target.value)}
      className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
    </label>
    <label className="space-y-1 text-sm">
     <span>Identifiant déclaré du réviseur</span>
     <input value={reviewer} onChange={e=>setReviewer(e.target.value)}
      className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
    </label>
   </div>
   <button type="button" onClick={()=>void prepare()} disabled={loading||
    evidenceIds.some(x=>!x)||!auditRef.trim()||!/^[a-f0-9]{64}$/.test(auditSha)||
    reviewer.trim().length<4}
    className="rounded bg-cyan-300 px-4 py-2 font-semibold text-slate-950 disabled:opacity-50">
    Sceller et relire le dossier de revue
   </button>
  </div>}
  {receipt&&<p className="text-sm text-emerald-300 break-all">
   Revue {receipt.status} : {receipt.reviewId} ·
   SHA-256 {receipt.reviewSha256}. Aucune décision acceptante.
  </p>}
  {receipt&&gate?.gate.acceptanceAllowedNow&&<div className="space-y-2 border-t border-amber-800 pt-3">
   <h3 className="font-bold text-amber-200">Décision humaine distincte</h3>
   <p className="text-xs text-amber-100">
    Actif uniquement après GO distinct, activation serveur ET DB.
    Le libellé saisi est une déclaration, pas une signature personnelle forte.
   </p>
   <input placeholder="Identifiant déclaré de l’approbateur" value={approver}
    onChange={e=>setApprover(e.target.value)}
    className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
   <input placeholder="Référence du GO d’activation" value={approvalRef}
    onChange={e=>setApprovalRef(e.target.value)}
    className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
   <input placeholder="Recopier : ACCEPTER J0 SHADOW" value={approvalPhrase}
    onChange={e=>setApprovalPhrase(e.target.value)}
    className="w-full rounded border border-slate-600 bg-slate-900 p-2"/>
   <button type="button" onClick={()=>void accept()} disabled={loading||
    approver.trim().length<4||approvalPhrase!=='ACCEPTER J0 SHADOW'}
    className="rounded bg-rose-300 px-4 py-2 font-bold text-slate-950 disabled:opacity-50">
    Accepter explicitement J0 (uniquement après GO distinct)
   </button>
  </div>}
  {decisionResult&&<p className="text-emerald-300">{decisionResult}</p>}
  {error&&<p role="alert" className="text-rose-300">NON VALIDÉ — {error}</p>}
 </section>;
}
