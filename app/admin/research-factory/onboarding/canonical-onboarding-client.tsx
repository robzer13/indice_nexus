'use client';

import {useState} from 'react';
import type {P0CDecision} from
 '@/lib/orotitan-equity/product/runfactory-canonical-onboarding-candidate-v1';

type InspectReply=Readonly<{
 ok:true;mode:'PRE_ADMISSION_CANDIDATE_ONLY';
 checkedAt:string;queryDurationMs:number;result:P0CDecision;
 sourceVerified:false;legacyCrosswalkAuthorized:false;
 rpcExecuted:false;registryMutated:false;productionAccessed:false;
}>;
export function CanonicalOnboardingClient(){
 const [isin,setIsin]=useState('');
 const [ticker,setTicker]=useState('');
 const [exchange,setExchange]=useState('');
 const [issuerName,setIssuerName]=useState('');
 const [reference,setReference]=useState('');
 const [busy,setBusy]=useState(false);
 const [error,setError]=useState('');
 const [report,setReport]=useState<InspectReply|null>(null);
 function changed(fn:(s:string)=>void){return (value:string)=>{fn(value);setReport(null);setError('');};}
 async function inspect(){
  if(busy||!isin.trim()||!ticker.trim()||!exchange.trim())return;
  setBusy(true);setError('');setReport(null);
  try{
   const res=await fetch('/admin/research-factory/onboarding/inspect',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({isin,ticker,exchange,
     issuerDisplayName:issuerName.trim()||null,
     evidenceUrl:reference.trim()||null}),
   });
   const raw:unknown=await res.json();
   if(!res.ok||!raw||typeof raw!=='object'||!('ok' in raw)||raw.ok!==true)
    throw Error(raw&&typeof raw==='object'&&'error' in raw?
     String(raw.error):'Erreur lors du contrôle d’identité.');
   const answer=raw as InspectReply;
   if(answer.mode!=='PRE_ADMISSION_CANDIDATE_ONLY'||
      answer.rpcExecuted!==false||answer.registryMutated!==false||
      answer.productionAccessed!==false||answer.sourceVerified!==false||
      answer.result.operationAuthorized!==false||
      answer.result.registryMutated!==false)
    throw Error('Réponse inattendue : intégrité du préflight non établie.');
   setReport(answer);
  }catch(e){setError(e instanceof Error?e.message:'Erreur de lecture.');}
  finally{setBusy(false);}
 }
 function exportReport(){
  if(!report)return;
  const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const anchor=document.createElement('a');
  anchor.href=url;anchor.download='OROTITAN_ONBOARDING_CANDIDATE_'+report.result.isin+'.json';
  document.body.appendChild(anchor);anchor.click();anchor.remove();URL.revokeObjectURL(url);
 }
 const data=[
  ['ISIN',isin,setIsin,'FR0000120321'],
  ['Ticker',ticker,setTicker,'OR'],
  ['Exchange',exchange,setExchange,'XPAR'],
 ];
 return <div className="space-y-5">
  <section className="rounded-lg border border-cyan-700 p-5 space-y-4">
   <h2 className="text-xl font-semibold text-white">Détection canonique et rapprochement du Screener</h2>
   <p className="text-sm text-slate-300">
    Saisissez une seule fois le titre à enregistrer. Le système recherche
    les identités existantes et propose les données déjà présentes dans
    le Screener, sans les considérer comme preuves juridiques.
   </p>
   <div className="grid gap-3 md:grid-cols-3">
    {data.map(([label,value,setValue,placeholder])=>
     <label key={label as string} className="space-y-1 text-sm text-slate-200">
      <span>{label as string}</span>
      <input value={value as string} disabled={busy}
       onChange={e=>changed(setValue as (v:string)=>void)(e.target.value)}
       placeholder={placeholder as string} maxLength={36}
       className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
     </label>)}
   </div>
   <label className="block space-y-1 text-sm text-slate-200">
    <span>Nom d’émetteur (facultatif — si non déjà retrouvé)</span>
    <input value={issuerName} maxLength={180} disabled={busy}
     onChange={e=>changed(setIssuerName)(e.target.value)}
     placeholder="Dénomination à rapprocher des sources officielles"
     className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
   </label>
   <label className="block space-y-1 text-sm text-slate-200">
    <span>Référence officielle (facultative — non vérifiée automatiquement)</span>
    <input value={reference} maxLength={2048} disabled={busy}
     onChange={e=>changed(setReference)(e.target.value)}
     placeholder="https://..."
     className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
   </label>
   <button type="button" disabled={busy||!isin.trim()||!ticker.trim()||!exchange.trim()}
    onClick={()=>void inspect()}
    className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
    {busy?'Contrôle en cours…':'Préparer l’onboarding — lecture seule'}
   </button>
   {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>
  {report&&<section className="rounded-lg border border-slate-700 p-5 space-y-4">
   <h2 className="text-xl font-semibold text-white">Proposition vérifiable, non admise</h2>
   <p className="font-mono text-sm text-amber-200">{report.result.status}</p>
   <dl className="grid gap-3 md:grid-cols-2 text-sm text-slate-200">
    <div><dt className="text-slate-400">Émetteur proposé</dt><dd>
     {report.result.proposedDisplayName??'INCONNU'}
    </dd></div>
    <div><dt className="text-slate-400">Origine du nom</dt><dd>
     {report.result.metadataSource}
    </dd></div>
    <div><dt className="text-slate-400">Fiche Screener héritée</dt><dd className="break-all">
     {report.result.legacyCompanyId??'AUCUNE'}
    </dd></div>
    <div><dt className="text-slate-400">Émetteur canonique existant</dt>
     <dd className="break-all">{report.result.existingIssuerId??'AUCUN CONFIRMÉ'}</dd></div>
    <div><dt className="text-slate-400">Titre canonique existant</dt>
     <dd className="break-all">{report.result.existingSecurityId??'AUCUN CONFIRMÉ'}</dd></div>
    <div><dt className="text-slate-400">Devise suggérée (non authentifiée)</dt>
     <dd>{report.result.tradingCurrencyCandidate??'À RÉSOUDRE'}</dd></div>
   </dl>
   <div className="space-y-2">
    <h3 className="font-semibold text-white">Contrôles et actions restantes</h3>
    <ul className="list-disc space-y-1 pl-5 text-sm text-slate-300">
     {report.result.blockerReasons.map(x=><li key={x}>{x}</li>)}
     <li>Confirmer les sources officielles et la relation entre personne morale,
      ISIN, marché, ticker, devise, unité de cotation et dossier.</li>
     <li>Créer une RPC transactionnelle gouvernée avec verrou et contrôles
      de doublons avant toute persistance.</li>
    </ul>
   </div>
   <p className="text-xs text-amber-100">
    Le lien fourni est seulement un pointeur, non une preuve consultée.
    Aucune source externe n’a été récupérée par ce module.
    Aucune émission d’UUID, aucune écriture dans Supabase,
    aucun appel RPC, aucune publication ni ouverture de Research.
   </p>
   <p className="text-xs text-slate-400">
    Inspection {report.queryDurationMs} ms · empreinte diagnostic
    <code className="break-all"> {report.result.candidateFingerprintSha256}</code>
   </p>
   <button type="button" onClick={exportReport}
    className="rounded-md border border-cyan-400 px-4 py-2 text-cyan-300">
    Exporter le dossier de pré-admission JSON
   </button>
  </section>}
 </div>;
}
