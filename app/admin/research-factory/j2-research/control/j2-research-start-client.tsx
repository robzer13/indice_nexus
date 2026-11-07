'use client';
import {useRef,useState} from 'react';

type J2Check={
 runId:string;expectedRunStateVersion:number;
 stageContractSha256:string;contractSetSha256:string;
};

export function J2ResearchStartClient({runId}:{runId:string}){
 const [preflight,setPreflight]=useState<J2Check|null>(null);
 const [approved,setApproved]=useState(false);
 const [loading,setLoading]=useState(false);
 const [attempted,setAttempted]=useState(false);
 const [message,setMessage]=useState('');
 const [started,setStarted]=useState(false);
 const lock=useRef(false);
 async function inspect(){
  if(loading||attempted)return;
  setLoading(true);setMessage('');setPreflight(null);setApproved(false);
  try{
   const response=await fetch(
    '/admin/research-factory/j2-research/preflight?runId='+
    encodeURIComponent(runId),{cache:'no-store',credentials:'same-origin'});
   const parsed:unknown=await response.json();
   if(!response.ok||!parsed||typeof parsed!=='object')
    throw Error('J2_PREFLIGHT_BLOCKED');
   const p=parsed as Record<string,unknown>;
   if(p.ok!==true||p.eligible!==true||
      p.reasonCode!=='J2_RESEARCH_START_PREFLIGHT_PASS'||
      p.runId!==runId||typeof p.expectedRunStateVersion!=='number'||
      !Number.isSafeInteger(p.expectedRunStateVersion)||
      p.stageStartAuthorized!==false||p.rpcExecuted!==false||
      p.researchExecuted!==false||p.publicationAuthorized!==false||
      p.readOnly!==true||p.environment!=='SHADOW_ONLY'||
      typeof p.contractSetSha256!=='string'||
      typeof p.stageContractSha256!=='string')
    throw Error('J2_PREFLIGHT_INVARIANTS_FAILED');
   setPreflight({
    runId,expectedRunStateVersion:p.expectedRunStateVersion,
    stageContractSha256:p.stageContractSha256,
    contractSetSha256:p.contractSetSha256,
   });
  }catch(e){setMessage(e instanceof Error?e.message:'J2_PREFLIGHT_FAILED');}
  finally{setLoading(false);}
 }
 async function start(){
  if(!preflight||!approved||attempted||loading||lock.current)return;
  lock.current=true;setAttempted(true);setLoading(true);setMessage('');
  try{
   const response=await fetch('/admin/research-factory/j2-research/start',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
     runId,expectedRunStateVersion:preflight.expectedRunStateVersion,
     contractSetSha256:preflight.contractSetSha256,
     stageContractSha256:preflight.stageContractSha256,
     confirmation:'GO J2 RESEARCH SHADOW '+runId,
    }),
   });
   const body:unknown=await response.json();
   if(!response.ok||!body||typeof body!=='object')
    throw Error('J2_STAGE_START_NOT_CONFIRMED');
   const v=body as Record<string,unknown>;
   if(v.ok!==true||v.status!=='RESEARCH_STARTED'||
      v.runId!==runId||v.stage!=='RESEARCH'||
      v.stageLifecycle!=='IN_PROGRESS'||
      v.registryReadbackVerified!==true||
      v.researchExecuted!==false||v.snapshotCreated!==false||
      v.publicationAuthorized!==false||
      v.environment!=='SHADOW_ONLY'||v.productionTouched!==false)
    throw Error('J2_STAGE_START_READBACK_NOT_CONFIRMED');
   setStarted(true);
  }catch(e){
   setMessage((e instanceof Error?e.message:'J2_STAGE_START_UNVERIFIED')+
    ' — Ne relancez pas le POST. Contrôlez le Registry en lecture seule.');
  }finally{setLoading(false);}
 }
 return <section className="space-y-4 rounded-lg border border-slate-700 p-5">
  <h2 className="text-xl font-semibold">Research V2.0 — Démarrage contrôlé</h2>
  <p className="break-all font-mono text-xs">RUN_ID = {runId}</p>
  <p className="text-sm text-amber-200">SHADOW uniquement.
   Le préflight ne démarre aucun stage. Une autorisation explicite
   déclenche au maximum une RPC canonique, sans publication.</p>
  {!attempted&&<button onClick={()=>void inspect()} type="button"
   disabled={loading} className="rounded bg-cyan-400 px-4 py-2 text-slate-950">
   Vérifier l’admission (lecture seule)
  </button>}
  {preflight&&!attempted&&<div className="space-y-3">
   <p className="text-sm text-emerald-200">Admission technique conforme.
    Version du Run : {preflight.expectedRunStateVersion}.</p>
   <label className="flex items-start gap-3 text-sm">
    <input type="checkbox" checked={approved}
     onChange={e=>setApproved(e.target.checked)}/>
    <span>J’autorise une seule transaction de démarrage Research sur SHADOW.
     Le Run passera à ACTIVE / RESEARCH IN_PROGRESS, sans produire
     encore d’analyse financière.</span>
   </label>
   <button type="button" onClick={()=>void start()}
    disabled={!approved||loading}
    className="rounded bg-amber-400 px-4 py-3 text-slate-950 disabled:opacity-50">
    GO — Démarrer Research (SHADOW)
   </button>
  </div>}
  {message&&<p role="alert" className="text-sm text-rose-300">{message}</p>}
  {started&&<p className="font-semibold text-emerald-200">
   RESEARCH_STARTED — Registry relu et vérifié. Collecte encore à exécuter.
  </p>}
 </section>;
}
