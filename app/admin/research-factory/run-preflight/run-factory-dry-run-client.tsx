'use client';

import {useState} from 'react';
import type {ContractProfileSummary} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';
import {isJ1InitialAdmissionCandidate,type RunFactoryDecision} from
 '@/lib/orotitan-equity/product/runfactory-identity-preflight-v1';
import type {RunPreflightSeed} from
 '@/lib/orotitan-equity/product/runfactory-run-preflight-handoff-v1';
import {buildJ1ResearchHandoff} from
 '@/lib/orotitan-equity/product/runfactory-j1-research-handoff-v1';

type PreflightReply={ok:true;checkedAt:string;result:RunFactoryDecision;
 mode:'DRY_RUN_ONLY';rpcExecuted:false;registryAdmitted:false;
 humanGOAttached:false;contractOriginalBytesVerifiedForThisSession:false;
 authoritativeProductionEnvironmentVerified:false;};
type Identity={isin:string;exchange:string;issuerId:string;
 issuerName:string;securityId:string;ticker:string};
type ResolveReply={ok:true;status:'RESOLVED'|'AMBIGUOUS'|'NOT_FOUND';
 identity:Identity|null;readOnly:true;environment:'SHADOW_ONLY'};
type CommitReply={ok:true;status:'CREATED'|'REPLAYED';runId:string;
 issuerId:string;securityId:string;dossierId:string;
 dataCutoff:string;runType:'INITIAL';runStatus:'CREATED';
 profileId:'OROTITAN_V3_1';contractSetSha256:string;
 all15OriginalContractBytesVerified:true;persistedReadbackVerified:true;
 receiptPersisted:true;runIdentityBound:true;
 stageStarted:false;snapshotCreated:false;
 published:false;environment:'SHADOW_ONLY';productionTouched:false;};
const ONBOARDING_SHADOW_URL=
 'https://orotitan-j1-identity-shadow-preview-e6xopn7xc-orotitan.vercel.app/admin/research-factory/onboarding/execute';

function todayLocal():string{
 const now=new Date();
 const p=(x:number)=>String(x).padStart(2,'0');
 return now.getFullYear()+'-'+p(now.getMonth()+1)+'-'+p(now.getDate());
}
function downloadJSON(filename:string,value:unknown):void{
 const blob=new Blob([JSON.stringify(value,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob);
 const link=document.createElement('a');link.href=url;link.download=filename;
 document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);
}
async function postJSON(path:string,body:unknown):Promise<unknown>{
 const response=await fetch(path,{
  method:'POST',cache:'no-store',credentials:'same-origin',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body),
 });
 const parsed:unknown=await response.json();
 if(!response.ok||!parsed||typeof parsed!=='object'||
    !('ok' in parsed)||parsed.ok!==true)
  throw Error(parsed&&typeof parsed==='object'&&'error' in parsed?
   String(parsed.error):'Requête SHADOW indisponible.');
 return parsed;
}

/**
 * J1: one user request -> unique canonical identity -> read-only preflight ->
 * one explicit GO -> existing atomic RPC -> verified Research routing packet.
 * The initial GO is mandatory. Neither stages nor publication are triggered.
 */
export function RunFactoryDryRunClient({
 profiles,initialValues,
}:{profiles:readonly ContractProfileSummary[];initialValues:RunPreflightSeed}){
 const [company,setCompany]=useState(initialValues.isin);
 const [cutoff,setCutoff]=useState(initialValues.dataCutoff||todayLocal());
 const [profile,setProfile]=useState('OROTITAN_V3_1');
 const [busy,setBusy]=useState<'prepare'|'commit'|null>(null);
 const [error,setError]=useState('');
 const [resolution,setResolution]=useState<'NOT_FOUND'|'AMBIGUOUS'|null>(null);
 const [identity,setIdentity]=useState<Identity|null>(null);
 const [preflight,setPreflight]=useState<PreflightReply|null>(null);
 const [approved,setApproved]=useState(false);
 const [commitAttempted,setCommitAttempted]=useState(false);
 const [commitAnswer,setCommitAnswer]=useState<CommitReply|null>(null);
 const [handoff,setHandoff]=useState('');
 const [copied,setCopied]=useState(false);
 const selected=profiles.find(p=>p.profileId===profile);
 const decision=preflight?.result??null;
 const canCommit=!!decision&&isJ1InitialAdmissionCandidate(decision);
 function reset(){
  setError('');setResolution(null);setIdentity(null);setPreflight(null);
  setApproved(false);setCommitAttempted(false);setCommitAnswer(null);
  setHandoff('');setCopied(false);
 }
 async function prepare(){
  if(busy||!company.trim()||!cutoff)return;
  setBusy('prepare');reset();
  try{
   if(profile!=='OROTITAN_V3_1'||!selected)
    throw Error('J1_PROFILE_NOT_SUPPORTED');
   const raw=await postJSON(
    '/admin/research-factory/run-preflight/resolve',{query:company.trim()},
   ) as ResolveReply;
   if(raw.readOnly!==true||raw.environment!=='SHADOW_ONLY')
    throw Error('J1_RESOLUTION_NOT_TRUSTED');
   if(raw.status!=='RESOLVED'||!raw.identity){
    setResolution(raw.status==='AMBIGUOUS'?'AMBIGUOUS':'NOT_FOUND');
    return;
   }
   const found=raw.identity;
   const p=await postJSON(
    '/admin/research-factory/run-preflight/inspect',{
     isin:found.isin,exchange:found.exchange,
     dataCutoff:cutoff,profileId:profile,
    },
   ) as PreflightReply;
   if(p.mode!=='DRY_RUN_ONLY'||p.rpcExecuted!==false||
      p.registryAdmitted!==false||p.humanGOAttached!==false||
      p.result.registryAdmitted!==false||p.result.rpcExecuted!==false||
      p.result.goAuthorized!==false||
      p.result.profileId!==profile||
      p.result.isin!==found.isin||
      p.result.exchange!==found.exchange||
      p.result.issuerId!==found.issuerId||
      p.result.securityId!==found.securityId||
      p.result.contractSetSha256!==selected.contractSetSha256)
    throw Error('J1_IDENTITY_PREFLIGHT_MISMATCH');
   setIdentity(found);setPreflight(p);
  }catch(e){setError(e instanceof Error?e.message:'J1_PREPARE_FAILED');}
  finally{setBusy(null);}
 }
 async function go(){
  if(!canCommit||!approved||busy||commitAttempted||!decision||!identity)return;
  // One HTTP write attempt per human GO. A transport failure requires a
  // new read-only preflight, never a blind second POST.
  setCommitAttempted(true);setBusy('commit');setError('');
  try{
   const report=await postJSON(
    '/admin/research-factory/run-preflight/commit',{
     profileId:'OROTITAN_V3_1',intent:'INITIAL_RUN_SHADOW_ONLY',
     issuerId:decision.issuerId,securityId:decision.securityId,
     dossierId:decision.dossierId,isin:decision.isin,
     exchange:decision.exchange,dataCutoff:decision.dataCutoff,
     expectedExistingRunId:decision.existingRunId,
     confirmation:'GO P0-E SHADOW '+decision.isin+' '+decision.dataCutoff,
    },
   ) as CommitReply;
   if(report.profileId!=='OROTITAN_V3_1'||
      report.runType!=='INITIAL'||report.runStatus!=='CREATED'||
      report.issuerId!==decision.issuerId||
      report.securityId!==decision.securityId||
      report.dossierId!==decision.dossierId||
      report.dataCutoff!==decision.dataCutoff||
      report.contractSetSha256!==decision.contractSetSha256||
      (decision.existingRunId===null?report.status!=='CREATED':
       report.status!=='REPLAYED'||report.runId!==decision.existingRunId)||
      report.all15OriginalContractBytesVerified!==true||
      report.runIdentityBound!==true)
    throw Error('J1_COMMIT_RECEIPT_MISMATCH');
   const packet=buildJ1ResearchHandoff({
    runId:report.runId,issuerId:report.issuerId,
    securityId:report.securityId,dossierId:report.dossierId,
    issuerName:identity.issuerName,
    isin:identity.isin,exchange:identity.exchange,
    dataCutoff:decision.dataCutoff,profileId:report.profileId,
    contractSetSha256:report.contractSetSha256,status:report.status,
    persistedReadbackVerified:report.persistedReadbackVerified,
    receiptPersisted:report.receiptPersisted,stageStarted:report.stageStarted,
    snapshotCreated:report.snapshotCreated,published:report.published,
    environment:report.environment,productionTouched:report.productionTouched,
   });
   setCommitAnswer(report);setHandoff(packet);
  }catch(e){
   setError((e instanceof Error?e.message:'J1_COMMIT_FAILED')+
    ' — Ne réessayez pas la transaction. Relancez uniquement le contrôle en lecture.');
  }finally{setBusy(null);}
 }
 async function copyHandoff(){
  if(!handoff)return;
  try{await navigator.clipboard.writeText(handoff);setCopied(true);}
  catch{setError('Copie indisponible : sélectionnez le texte du handoff.');}
 }
 return <div className="space-y-5">
  <section className="rounded-lg border border-cyan-700 p-5 space-y-4">
   <h2 className="text-xl font-semibold text-white">Demander une analyse — J1 SHADOW</h2>
   <p className="text-sm text-slate-300">
    Saisissez le nom, le ticker ou l’ISIN. OroTitan résout l’identité et
    détermine automatiquement s’il faut créer ou réutiliser le Run.
    Aucune transaction avant votre GO explicite.
   </p>
   <label className="block space-y-1 text-sm text-slate-200">
    <span>Entreprise, ticker ou ISIN</span>
    <input value={company} disabled={busy!==null||commitAnswer!==null}
     onChange={e=>{setCompany(e.target.value);reset();}}
     placeholder="Schneider Electric, SU ou FR0000121972"
     maxLength={100}
     className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
   </label>
   <details className="rounded border border-slate-700 p-3 text-sm text-slate-300">
    <summary className="cursor-pointer font-semibold">Paramètres d’exécution</summary>
    <div className="mt-3 grid gap-3 md:grid-cols-2">
     <label className="space-y-1"><span>Data cutoff</span>
      <input type="date" value={cutoff}
       disabled={busy!==null||commitAnswer!==null}
       onChange={e=>{setCutoff(e.target.value);reset();}}
       className="w-full rounded bg-slate-900 p-2"/></label>
     <label className="space-y-1"><span>Profil figé</span>
      <select value={profile} disabled={busy!==null||commitAnswer!==null}
       onChange={e=>{setProfile(e.target.value);reset();}}
       className="w-full rounded bg-slate-900 p-2">
       <option value="OROTITAN_V3_1">OROTITAN V3.1</option>
      </select></label>
    </div>
    {selected&&<p className="mt-2 break-all text-xs">
     Contract Set SHA-256 : {selected.contractSetSha256}</p>}
   </details>
   <button type="button" onClick={()=>void prepare()}
    disabled={busy!==null||!company.trim()||!cutoff||commitAnswer!==null}
    className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
    {busy==='prepare'?'Résolution et préflight…':'Préparer l’analyse (lecture seule)'}
   </button>
   {resolution==='NOT_FOUND'&&<div className="space-y-2 text-sm text-amber-100">
    <p>Identité canonique absente. Onboarding avec revue humaine des sources
     officielles requis avant création d’un Run.</p>
    <a href={ONBOARDING_SHADOW_URL} target="_blank" rel="noopener noreferrer"
     className="text-cyan-300 underline">Ouvrir l’onboarding SHADOW contrôlé</a>
    <p>Après admission, revenez ici et relancez « Préparer l’analyse ».</p>
   </div>}
   {resolution==='AMBIGUOUS'&&<p className="text-sm text-amber-100">
    Plusieurs titres correspondent. Utilisez l’ISIN exact. Aucun titre n’a été
    choisi automatiquement.</p>}
   {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>
  {decision&&identity&&<section className="rounded-lg border border-slate-700 p-5 space-y-3">
   <h2 className="text-xl font-semibold text-white">Plan automatiquement déterminé</h2>
   <p className="font-mono text-sm text-amber-200">{decision.status}</p>
   <p className="text-sm text-slate-200">{identity.issuerName} · {decision.isin} ·
    {decision.exchange} · Cutoff {decision.dataCutoff}</p>
   <p className="text-sm text-slate-300">Type : {decision.proposedRunType??'Non déterminé'} ·
    Run existant : {decision.existingRunId??'aucun'}</p>
   {!canCommit&&<p className="text-sm text-amber-100">
    Ce cas n’est pas admissible à J1 INITIAL. Aucun GO ni RPC disponible.
    {decision.nextAction}
   </p>}
   {canCommit&&!commitAnswer&&<div className="space-y-3 rounded-md border border-amber-600 p-4">
    <label className="flex items-start gap-3 text-sm text-amber-100">
     <input type="checkbox" checked={approved}
      disabled={busy!==null||commitAttempted}
      onChange={e=>setApproved(e.target.checked)}/>
     <span>Je donne le GO d’admission J1 SHADOW pour ce Run
      {decision.existingRunId?' (réutilisation)':' (création INITIAL)'}.
      Les 15 contrats seront revérifiés avant la transaction.
      Aucun stage, snapshot ni publication n’est autorisé.</span>
    </label>
    <button type="button" disabled={!approved||busy!==null||commitAttempted}
     onClick={()=>void go()}
     className="rounded bg-amber-400 px-4 py-3 font-bold text-slate-950 disabled:opacity-50">
     {busy==='commit'?'Admission et relecture persistante…':
      decision.existingRunId?'GO — Réutiliser le Run existant':
      'GO — Créer le Run INITIAL'}
    </button>
    {commitAttempted&&!commitAnswer&&<p className="text-sm text-amber-100">
     Aucune répétition automatique de RPC. Pour connaître l’état réel,
     utilisez « Préparer l’analyse » (lecture seule).</p>}
   </div>}
   <button type="button" onClick={()=>downloadJSON(
    'OROTITAN_J1_PREFLIGHT.json',preflight)}
    className="rounded-md border border-cyan-400 px-3 py-2 text-sm text-cyan-300">
    Exporter le préflight</button>
  </section>}
  {commitAnswer&&<section className="space-y-3 rounded-lg border border-emerald-700 p-5">
   <h2 className="text-xl font-semibold text-emerald-200">
    RUN {commitAnswer.status} — J1 persisté et relu</h2>
   <p className="break-all font-mono text-xs text-slate-200">
    RUN_ID = {commitAnswer.runId}</p>
   <p className="text-sm text-slate-300">
    15/15 contrats vérifiés. Reçu persisté. Aucun stage, snapshot ni publication.
   </p>
   <h3 className="font-semibold text-white">Handoff vers Research</h3>
   <p className="text-xs text-amber-100">Paquet d’orientation, pas admission
    automatique du stage Research. Le stage devra revérifier le Registry et
    les contrats avant exécution.</p>
   <textarea aria-label="Handoff Research" readOnly rows={17}
    value={handoff} className="w-full rounded bg-slate-900 p-3 font-mono text-xs text-slate-200"/>
   <div className="flex flex-wrap gap-3">
    <button type="button" onClick={()=>void copyHandoff()}
     className="rounded bg-cyan-400 px-4 py-2 font-semibold text-slate-950">
     {copied?'Handoff copié':'Copier pour Research'}</button>
    <button type="button" onClick={()=>downloadJSON('OROTITAN_J1_RUN_RECEIPT.json',{
      ...commitAnswer,handoff})}
     className="rounded border border-cyan-500 px-4 py-2 text-cyan-200">
     Exporter reçu + handoff</button>
   </div>
  </section>}
 </div>;
}
