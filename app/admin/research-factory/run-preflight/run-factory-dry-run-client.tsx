'use client';

import {useState} from 'react';
import type {ContractProfileSummary} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';
import type {RunFactoryDecision} from
 '@/lib/orotitan-equity/product/runfactory-identity-preflight-v1';
import type {RunPreflightSeed} from
 '@/lib/orotitan-equity/product/runfactory-run-preflight-handoff-v1';

type PreflightReply={ok:true;checkedAt:string;result:RunFactoryDecision;
 mode:'DRY_RUN_ONLY';rpcExecuted:false;registryAdmitted:false;
 humanGOAttached:false;contractOriginalBytesVerifiedForThisSession:false;
 authoritativeProductionEnvironmentVerified:false;};

type CommitReply={ok:true;status:'CREATED'|'REPLAYED';runId:string;
 profileId:'OROTITAN_V3_1';persistedReadbackVerified:true;
 receiptPersisted:true;stageStarted:false;snapshotCreated:false;
 published:false;environment:'SHADOW_ONLY';productionTouched:false;};

export function RunFactoryDryRunClient({
 profiles,initialValues,
}:{profiles:readonly ContractProfileSummary[];initialValues:RunPreflightSeed}){
 const [isin,setIsin]=useState(initialValues.isin);
 const [exchange,setExchange]=useState(initialValues.exchange);
 const [cutoff,setCutoff]=useState(initialValues.dataCutoff);
 const [profile,setProfile]=useState<string>(initialValues.profileId);
 const [running,setRunning]=useState(false);
 const [error,setError]=useState('');
 const [answer,setAnswer]=useState<PreflightReply|null>(null);
 const [approval,setApproval]=useState('');
 const [committing,setCommitting]=useState(false);
 const [commitAnswer,setCommitAnswer]=useState<CommitReply|null>(null);

 async function inspect(){
  if(running||!profile||!isin.trim()||!cutoff)return;
  setRunning(true);setError('');setAnswer(null);setApproval('');setCommitAnswer(null);
  try{
   const response=await fetch('/admin/research-factory/run-preflight/inspect',{
    method:'POST',cache:'no-store',credentials:'same-origin',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({isin,exchange:exchange.trim()||null,
     dataCutoff:cutoff,profileId:profile}),
   });
   const v:unknown=await response.json();
   if(!response.ok||!v||typeof v!=='object'||!('ok' in v)||v.ok!==true)
    throw Error(v&&typeof v==='object'&&'error' in v?
     String(v.error):'Échec du préflight de lecture.');
   const parsed=v as PreflightReply;
   if(parsed.mode!=='DRY_RUN_ONLY'||parsed.rpcExecuted!==false||
      parsed.registryAdmitted!==false||parsed.humanGOAttached!==false||
      parsed.result.registryAdmitted!==false||
      parsed.result.rpcExecuted!==false||
      parsed.result.goAuthorized!==false||
      parsed.result.profileId!==profile)
    throw Error('Réponse incohérente ou non conforme au mode préflight.');
   setAnswer(parsed);
  }catch(e){setError(e instanceof Error?e.message:'Échec de lecture.');}
  finally{setRunning(false);}
 }
 async function createRun(){
  if(!answer||committing||running||
     answer.result.status!=='NEW_RUN_PLAN_CANDIDATE'||
     answer.result.profileId!=='OROTITAN_V3_1'||
     !answer.result.issuerId||!answer.result.securityId||
     !answer.result.dossierId||
     approval!=='GO P0-E SHADOW '+answer.result.isin+' '+answer.result.dataCutoff)
   return;
  setCommitting(true);setError('');setCommitAnswer(null);
  try{
   const r=await fetch('/admin/research-factory/run-preflight/commit',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
     profileId:'OROTITAN_V3_1',intent:'INITIAL_RUN_SHADOW_ONLY',
     issuerId:answer.result.issuerId,
     securityId:answer.result.securityId,
     dossierId:answer.result.dossierId,
     isin:answer.result.isin,
     exchange:answer.result.exchange,
     dataCutoff:answer.result.dataCutoff,
     confirmation:approval,
    }),
   });
   const raw:unknown=await r.json();
   if(!r.ok||!raw||typeof raw!=='object'||!('ok' in raw)||raw.ok!==true)
    throw Error(raw&&typeof raw==='object'&&'error' in raw?
     String(raw.error):'P0-E : création non confirmée.');
   const report=raw as CommitReply;
   if(report.profileId!=='OROTITAN_V3_1'||
      report.persistedReadbackVerified!==true||
      report.receiptPersisted!==true||report.stageStarted!==false||
      report.snapshotCreated!==false||report.published!==false||
      report.environment!=='SHADOW_ONLY'||report.productionTouched!==false)
    throw Error('P0-E : reçu incomplet ; contrôle manuel du Registry requis.');
   setCommitAnswer(report);
  }catch(e){
   setError(e instanceof Error?e.message:'P0-E : erreur de transaction.');
  }finally{setCommitting(false);}
 }
 function download(){
  if(!answer)return;
  const bytes=JSON.stringify({
   schema:'OROTITAN_RUN_FACTORY_IDENTITY_DRYRUN_V1',
   ...answer,
  },null,2);
  const blob=new Blob([bytes],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;
  link.download='OROTITAN_RUN_FACTORY_IDENTITY_PREFLIGHT.json';
  document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);
 }
 const selected=profiles.find(p=>p.profileId===profile);
 return <div className="space-y-5">
  <section className="rounded-lg border border-cyan-700 p-5 space-y-4">
   <h2 className="text-xl font-semibold text-white">Contrôle d’identité et de doublon</h2>
   <p className="text-sm text-slate-300">
    Un titre est identifié par son ISIN, puis son marché si nécessaire.
    L’identité doit déjà exister dans le Registry.
    La génération reste explicitement sélectionnée ; aucun GO n’est envoyé.
   </p>
   <div className="grid gap-3 md:grid-cols-2">
    <label className="space-y-1 text-sm text-slate-200">
     <span>ISIN (obligatoire)</span>
     <input value={isin} onChange={e=>setIsin(e.target.value)}
      placeholder="FR0000120321" maxLength={16} disabled={running}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
    </label>
    <label className="space-y-1 text-sm text-slate-200">
     <span>Marché (facultatif)</span>
     <input value={exchange} onChange={e=>setExchange(e.target.value)}
      placeholder="XPAR" maxLength={30} disabled={running}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
    </label>
    <label className="space-y-1 text-sm text-slate-200">
     <span>Data cutoff</span>
     <input type="date" value={cutoff} onChange={e=>setCutoff(e.target.value)}
      disabled={running}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
    </label>
    <label className="space-y-1 text-sm text-slate-200">
     <span>Profil contractuel approuvé</span>
     <select value={profile} disabled={running}
      onChange={e=>setProfile(e.target.value)}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white">
      <option value="">— Choix explicite requis —</option>
      {profiles.map(p=><option value={p.profileId} key={p.profileId}>
       {p.label}
      </option>)}
     </select>
    </label>
   </div>
   {selected&&<p className="break-all text-xs text-slate-400">
    Contract Set SHA-256 : {selected.contractSetSha256}
   </p>}
   <button type="button" onClick={()=>void inspect()}
    disabled={running||!profile||!isin.trim()||!cutoff}
    className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
    {running?'Lecture du Registry SHADOW…':'Exécuter le préflight (lecture seule)'}
   </button>
   {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>
  {answer&&<section className="rounded-lg border border-slate-700 p-5 space-y-3">
   <h2 className="text-xl font-semibold text-white">Décision de préflight</h2>
   <p className="font-mono text-amber-200">{answer.result.status}</p>
   <p className="text-sm text-slate-200">{answer.result.nextAction}</p>
   <dl className="grid grid-cols-1 gap-2 text-sm text-slate-300 md:grid-cols-2">
    <div><dt>Émetteur</dt><dd className="font-semibold">
     {answer.result.issuerName??'NON RÉSOLU'}</dd></div>
    <div><dt>Type de run candidat</dt><dd>
     {answer.result.proposedRunType??'NON DÉTERMINÉ'}</dd></div>
    <div><dt>Run existant</dt><dd className="break-all">
     {answer.result.existingRunId??'AUCUN CONFIRMÉ'}</dd></div>
    <div><dt>Snapshot de référence</dt><dd className="break-all">
     {answer.result.baselineSnapshotId??'AUCUN'}</dd></div>
   </dl>
   <p className="text-xs text-amber-100">
    Pas d’autorisation GO, pas de validation des octets contractuels dans
    cette session, pas de vérification de la base de production.
    Aucune RPC exécutée, aucun run créé, aucune admission.
   </p>
   {answer.result.status==='NEW_RUN_PLAN_CANDIDATE'&&
    answer.result.profileId==='OROTITAN_V3_1'&&
    answer.result.issuerId&&answer.result.securityId&&answer.result.dossierId&&
    <div className="rounded-lg border border-amber-600 p-4 space-y-3">
     <h3 className="font-bold uppercase text-amber-200">
      ACTION ADMINISTRATEUR REQUISE — GO P0-E SHADOW
     </h3>
     <p className="text-sm text-slate-200">
      Crée un run INITIAL et rattache son identité dans UNE transaction.
      Vérifie à nouveau les 15 contrats d’origine. Aucun stage, analyse,
      snapshot ni publication ne sera lancé.
     </p>
     <label className="block space-y-1 text-sm text-slate-200">
      <span>Recopier exactement : GO P0-E SHADOW {answer.result.isin} {answer.result.dataCutoff}</span>
      <input value={approval} disabled={committing||commitAnswer!==null}
       onChange={e=>setApproval(e.target.value)}
       className="w-full rounded border border-amber-500 bg-slate-900 p-3 text-white"/>
     </label>
     <button type="button" onClick={()=>void createRun()}
      disabled={committing||commitAnswer!==null||
       approval!=='GO P0-E SHADOW '+answer.result.isin+' '+answer.result.dataCutoff}
      className="rounded bg-amber-400 px-4 py-3 font-bold text-slate-950 disabled:opacity-50">
      {committing?'Création et contrôle d’intégrité…':'Créer le run INITIAL SHADOW — V3.1'}
     </button>
     {commitAnswer&&<div className="space-y-1 rounded border border-emerald-700 p-3 text-sm text-emerald-200">
      <p className="font-bold">RUN {commitAnswer.status} — LECTURE PERSISTANTE VÉRIFIÉE</p>
      <p className="break-all font-mono">RUN_ID : {commitAnswer.runId}</p>
      <p>Aucun stage, snapshot ou publication.</p>
     </div>}
    </div>}
   <button type="button" onClick={download}
    className="rounded-md border border-cyan-400 px-4 py-2 text-cyan-300">
    Exporter la décision JSON
   </button>
  </section>}
 </div>;
}
