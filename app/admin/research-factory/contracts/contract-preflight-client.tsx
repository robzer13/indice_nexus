'use client';

import {useMemo,useRef,useState} from 'react';
import type {ContractProfileId,ContractProfileSummary} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';

type VerificationRow={
 profileId:ContractProfileId;logicalName:string;
 expectedSha256:string;resolvedSha256:string;resolvedBytes:number;
 status:'ORIGINAL_CONTRACT_BYTES_VERIFIED';registryAdmitted:false;
 checkedAt:string;
};
const ENDPOINT='/admin/research-factory/contracts/verify';

export function ContractPreflightClient({
 profiles,
}:{profiles:readonly ContractProfileSummary[]}){
 const [chosen,setChosen]=useState<ContractProfileId|''>('');
 const [rows,setRows]=useState<VerificationRow[]>([]);
 const rowsRef=useRef<VerificationRow[]>([]);
 const stopRef=useRef(false);
 const abortRef=useRef<AbortController|null>(null);
 const [running,setRunning]=useState(false);
 const [error,setError]=useState('');
 const [status,setStatus]=useState('Choisir explicitement un profil contractuel.');
 const active=useMemo(()=>profiles.find(x=>x.profileId===chosen),[profiles,chosen]);
 const complete=Boolean(active&&rows.length===active.pinCount&&
  active.pins.every(pin=>rows.some(r=>r.logicalName===pin.logicalName&&
   r.status==='ORIGINAL_CONTRACT_BYTES_VERIFIED'&&
   r.expectedSha256===pin.contentSha256)));
 const next=active?.pins.find(pin=>!rows.some(r=>r.logicalName===pin.logicalName));

 function selectProfile(value:string){
  if(running)return;
  const valid=profiles.some(x=>x.profileId===value);
  setChosen(valid?value as ContractProfileId:'');
  setRows([]);rowsRef.current=[];setError('');
  setStatus(valid?'Profil sélectionné — les octets des contrats ne sont pas encore vérifiés.':
   'Choisir explicitement un profil contractuel.');
 }
 async function start(){
  if(!active||running||complete)return;
  stopRef.current=false;setRunning(true);setError('');
  try{
   for(const pin of active.pins){
    if(stopRef.current)break;
    if(rowsRef.current.some(r=>r.logicalName===pin.logicalName))continue;
    setStatus('Vérification Git blob + SHA-256 canonique : '+pin.logicalName);
    const controller=new AbortController();
    abortRef.current=controller;
    const response=await fetch(ENDPOINT,{
     method:'POST',credentials:'same-origin',cache:'no-store',
     headers:{'Content-Type':'application/json'},
     body:JSON.stringify({profileId:active.profileId,logicalName:pin.logicalName}),
     signal:controller.signal,
    });
    const result:unknown=await response.json();
    if(!response.ok||!result||typeof result!=='object'||
       !('ok' in result)||result.ok!==true)
     throw Error('Vérification impossible pour '+pin.logicalName+' ('+
      (result&&typeof result==='object'&&'error' in result?
       String(result.error):'HTTP '+response.status)+')');
    const verified=result as Record<string,unknown>;
    if(verified.profileId!==active.profileId||
       verified.logicalName!==pin.logicalName||
       verified.contractSetSha256!==active.contractSetSha256||
       verified.expectedSha256!==pin.contentSha256||
       verified.resolvedSha256!==pin.contentSha256||
       verified.status!=='ORIGINAL_CONTRACT_BYTES_VERIFIED'||
       verified.registryAdmitted!==false||
       verified.issuerResearchAuthorized!==false||
       !Number.isSafeInteger(verified.resolvedBytes)||
       typeof verified.checkedAt!=='string')
     throw Error('Réponse de vérification incohérente : '+pin.logicalName);
    const row:VerificationRow={
     profileId:active.profileId,logicalName:pin.logicalName,
     expectedSha256:pin.contentSha256,resolvedSha256:pin.contentSha256,
     resolvedBytes:verified.resolvedBytes as number,
     status:'ORIGINAL_CONTRACT_BYTES_VERIFIED',
     registryAdmitted:false,checkedAt:verified.checkedAt,
    };
    rowsRef.current=[...rowsRef.current,row];
    setRows(rowsRef.current);
   }
   if(stopRef.current)setStatus('Interrompu. Reprise dans cet onglet possible.');
   else setStatus('Vérification des octets terminée — aucun run ni stage créé.');
  }catch(e){
   if(stopRef.current)setStatus('Interrompu. Reprise possible.');
   else{
    setError(e instanceof Error?e.message:'Erreur de vérification.');
    setStatus('BLOQUÉ — conserver les résultats partiels, aucune admission.');
   }
  }finally{abortRef.current=null;setRunning(false);}
 }
 function stop(){stopRef.current=true;abortRef.current?.abort();}
 function reset(){
  if(running)return;
  rowsRef.current=[];setRows([]);setError('');
  setStatus('Contrôles réinitialisés. Aucun état serveur persistant.');
 }
 function exportResult(){
  if(!active)return;
  const report={
   schema:'OROTITAN_CONTRACT_PROFILE_PREFLIGHT_V1',
   checkedAt:new Date().toISOString(),
   profileSelectedExplicitly:active.profileId,
   contractSetSha256:active.contractSetSha256,
   expectedPinCount:active.pinCount,
   independentlyVerifiedPinCount:rowsRef.current.length,
   allPinnedContractBytesVerified:complete,
   scope:'PRE_ADMISSION_ONLY',
   runCreated:false,issuerResearchAuthorized:false,
   registryAdmitted:false,stageAdvanced:false,
   checkedPins:rowsRef.current,
   missingPins:active.pins.filter(p=>!rowsRef.current.some(
    r=>r.logicalName===p.logicalName)).map(x=>x.logicalName),
  };
  const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;
  a.download='OROTITAN_CONTRACT_PREFLIGHT_'+active.profileId+'.json';
  document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
 }
 return <div className="space-y-6">
  <section className="space-y-3 rounded-lg border border-cyan-700 p-5">
   <h2 className="text-xl font-semibold text-white">P0-A — sélection de l’autorité contractuelle</h2>
   <p className="text-sm text-slate-300">
    Les trois profils proviennent des packs existants du dépôt. Le choix n’est
    jamais déduit de la date, du ticker ou d’une version supposée plus récente.
    Aucun nouveau contrat n’est créé ici.
   </p>
   <label htmlFor="contract-profile" className="block text-sm text-slate-200">
    Profil méthodologique autorisé pour le run projeté
   </label>
   <select id="contract-profile" value={chosen} disabled={running}
    onChange={e=>selectProfile(e.target.value)}
    className="w-full rounded-md border border-slate-500 bg-slate-900 p-3 text-white">
    <option value="">— Sélection explicite requise —</option>
    {profiles.map(x=><option value={x.profileId} key={x.profileId}>
     {x.label} — {x.pinCount} pins
    </option>)}
   </select>
   {active&&<p className="break-all text-xs text-slate-400">
    Contract Set SHA-256 : {active.contractSetSha256}
   </p>}
   <div className="flex flex-wrap gap-3">
    <button type="button" onClick={()=>void start()}
     disabled={!active||running||complete}
     className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
     {rows.length?'Reprendre la vérification':'Vérifier tous les contrats'}
    </button>
    <button type="button" onClick={stop} disabled={!running}
     className="rounded-md border border-slate-500 px-4 py-3 text-white disabled:opacity-50">
     Interrompre
    </button>
    <button type="button" onClick={reset} disabled={running||rows.length===0}
     className="rounded-md border border-slate-500 px-4 py-3 text-white disabled:opacity-50">
     Réinitialiser
    </button>
   </div>
   <p role="status" aria-live="polite" className="text-sm text-cyan-100">{status}</p>
   {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>
  <section className="space-y-3 rounded-lg border border-slate-700 p-5">
   <h2 className="text-xl font-semibold text-white">Résultat de vérification</h2>
   <p className="text-sm text-slate-300">
    {rows.length}/{active?.pinCount??0} contrats canoniques vérifiés.
    {complete?' Pack entièrement vérifié en octets.':' État incomplet — non admissible.'}
   </p>
   {next&&<p className="text-xs text-slate-400">
    Prochaine autorité : {next.logicalName}
   </p>}
   {active&&<div className="overflow-x-auto">
    <table className="w-full text-left text-sm text-slate-200">
     <thead><tr className="border-b border-slate-700">
      <th className="p-2">Pin</th><th className="p-2">Version</th>
      <th className="p-2">Format</th><th className="p-2">Octets</th>
     </tr></thead>
     <tbody>{active.pins.map(pin=>{
      const match=rows.find(r=>r.logicalName===pin.logicalName);
      return <tr key={pin.logicalName} className="border-b border-slate-800">
       <td className="p-2 font-mono">{pin.logicalName}</td>
       <td className="p-2">{pin.version}</td>
       <td className="p-2">{pin.locatorFormat}</td>
       <td className="p-2">{match?'VÉRIFIÉS':'NON VÉRIFIÉS'}</td>
      </tr>;
     })}</tbody>
    </table>
   </div>}
   <p className="text-xs text-amber-100">
    Même 13/13 ou 15/15 ne constitue ni une sélection méthodologique
    autorisée pour un issuer, ni une création de run.
    Le contrôle des identités, du cutoff et des gates reste distinct.
    Les résultats partiels sont conservés seulement dans cet onglet.
   </p>
   <button type="button" disabled={!active||rows.length===0} onClick={exportResult}
    className="rounded-md border border-cyan-400 px-4 py-2 text-cyan-300 disabled:opacity-50">
    Exporter le rapport JSON de préflight
   </button>
  </section>
 </div>;
}
