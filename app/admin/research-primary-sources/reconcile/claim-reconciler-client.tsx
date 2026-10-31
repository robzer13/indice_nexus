'use client';

import {useMemo,useRef,useState} from 'react';
import {buildV35Matrix,validateV35Chunk,type V35Chunk,type V35Source}
 from '@/lib/orotitan-equity/product/asml-research-claim-matrix-v35';

const ENDPOINT='/admin/research-primary-sources/audit/run';
const RUN_ID='a1e59126-1ca9-4e55-8764-54a1e3d4dc24';
const CUTOFF='2026-10-30';
const BATCH_SIZE=20;

function createExport(sources:readonly V35Source[],chunks:readonly V35Chunk[]){
 return {
  schema:'OROTITAN_ASML_PRIVATE_PDF_RECONCILIATION_OBSERVATIONS_V3_5',
  runId:RUN_ID,dataCutoff:CUTOFF,generatedAt:new Date().toISOString(),
  provenance:'Private SHADOW stored bytes, v3.4 rehash + bounded PDF text extraction',
  validation:'PRE_ADMISSION_OBSERVATION_ONLY',
  sourcePublisherCryptographicallyAuthenticated:false,
  noteOnPostCutoffCapture:'A PDF captured after 2026-10-30 is not automatically an authenticated historical byte record.',
  historicalBytesAtCutoffCertified:false,
  pdfPagesVisuallyReviewed:false,
  evidenceClaimsValidated:0,registryAdmitted:false,
  readyForDeepDive:'NOT_EVALUATED',
  note:'Context hints only triage numeric candidates. Absence in extracted text is not evidence of absence in the PDF.',
  sources:sources.map(s=>({id:s.id,expectedSha256:s.sourceSha256,
   title:s.title,claimIds:s.probes.map(p=>p.claimId)})),
  matrix:buildV35Matrix(sources,chunks),
  scanReceipts:chunks.map(c=>({
   id:c.sourceId,sourceSha256:c.sourceSha256,receiptSha256:c.receiptSha256,
   checkedAt:c.checkedAt,sourcePublicationDate:c.sourceDate,
   storageCaptureTimestamp:c.captureTimestamp,
   capturedAfterCutoff:c.captureTimestamp.slice(0,10)>CUTOFF,
   startPage:c.startPage,endPage:c.endPage,
   pdfPageCount:c.pdfPageCount,pageTextHashes:c.pages.map(p=>({
    pdfPage:p.pdfPage,textSha256:p.textSha256,textLength:p.textLength,
   })),
  })),
 };
}
export function ClaimReconcilerClient({sources}:{sources:V35Source[]}){
 const [chunks,setChunks]=useState<V35Chunk[]>([]);
 const chunksRef=useRef<V35Chunk[]>([]);
 const cursorRef=useRef({sourceIndex:0,startPage:1});
 const abortRef=useRef<AbortController|null>(null);
 const stopRef=useRef(false);
 const [running,setRunning]=useState(false);
 const [status,setStatus]=useState('Prêt — aucun scan automatisé exécuté dans cette session.');
 const [error,setError]=useState('');
 const matrix=useMemo(()=>buildV35Matrix(sources,chunks),[sources,chunks]);

 async function start(){
  if(running||cursorRef.current.sourceIndex>=sources.length)return;
  stopRef.current=false;setRunning(true);setError('');
  try{
   while(cursorRef.current.sourceIndex<sources.length&&!stopRef.current){
    const {sourceIndex,startPage}=cursorRef.current;
    const source=sources[sourceIndex];
    if(!source)throw Error('Source inconnue.');
    setStatus(source.id+' — examen à partir de la page PDF '+startPage);
    const controller=new AbortController();
    abortRef.current=controller;
    const response=await fetch(ENDPOINT,{
     method:'POST',credentials:'same-origin',cache:'no-store',
     headers:{'Content-Type':'application/json'},
     body:JSON.stringify({
      mode:'scan_pages',sourceId:source.id,startPage,pageCount:BATCH_SIZE,
     }),signal:controller.signal,
    });
    const payload:unknown=await response.json();
    if(!response.ok){
     const err=typeof payload==='object'&&payload!==null&&'error' in payload?
      String(payload.error):'HTTP '+response.status;
     throw Error('Endpoint v3.4 : '+err);
    }
    const nextChunk=validateV35Chunk(payload,source,startPage);
    const next=[...chunksRef.current,nextChunk];
    // Guarantees contiguous, in-order coverage before recording this batch.
    buildV35Matrix(sources,next);
    chunksRef.current=next;
    setChunks(next);
    cursorRef.current=nextChunk.endPage===nextChunk.pdfPageCount?
     {sourceIndex:sourceIndex+1,startPage:1}:
     {sourceIndex,startPage:nextChunk.endPage+1};
   }
   if(stopRef.current)setStatus('Examen interrompu — résultats partiels conservés ici ; reprise possible.');
   else setStatus('Balayage terminé — revue humaine des contextes et des pages toujours requise.');
  }catch(e){
   if(stopRef.current){
    setStatus('Examen interrompu — reprise possible depuis la dernière page vérifiée.');
   }else{
    setError(e instanceof Error?e.message:'Échec du scan automatisé.');
    setStatus('Bloqué — aucune admission. Relancer pour reprendre au dernier lot validé.');
   }
  }finally{
   abortRef.current=null;
   setRunning(false);
  }
 }
 function stop(){
  if(!running)return;
  stopRef.current=true;
  abortRef.current?.abort();
 }
 function reset(){
  if(running)return;
  chunksRef.current=[];setChunks([]);
  cursorRef.current={sourceIndex:0,startPage:1};
  setError('');setStatus('Réinitialisé — aucun scan automatisé enregistré dans cette session.');
 }
 function getJson(){return JSON.stringify(createExport(sources,chunksRef.current),null,2);}
 async function copy(){
  try{await navigator.clipboard.writeText(getJson());setStatus('Rapport JSON copié — observations uniquement.');}
  catch{setError('Copie impossible : utilise l’export JSON.');}
 }
 function download(){
  const blob=new Blob([getJson()],{type:'application/json'});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;
  a.download='ASML_2026-10-30_a1e59126_RESEARCH_PRIVATE_PDF_RECONCILIATION_V3_5.json';
  document.body.appendChild(a);a.click();a.remove();URL.revokeObjectURL(url);
 }
 const nextSource=sources[cursorRef.current.sourceIndex];
 return <div className="space-y-6">
  <section className="space-y-3 rounded-lg border border-cyan-700 p-5">
   <h2 className="text-xl font-semibold text-white">Réconciliateur documentaire v3.5</h2>
   <p className="text-sm text-slate-300">
    Parcourt automatiquement les trois PDF privés par blocs de 20 pages, en réutilisant
    uniquement l’endpoint v3.4 authentifié, en lecture seule. Chaque appel recalcule
    l’empreinte du fichier avant extraction. Le calcul se déroule dans cette page ;
    fermer l’onglet interrompt l’exécution et perd les résultats non exportés.
   </p>
   <p className="text-sm text-amber-100">
    Extraction lexicale non autoritative : aucune validation de claim, aucune
    preuve d’authenticité de l’émetteur, aucun fait post-cutoff ajouté, aucune
    écriture Registry, aucune transition de stage.
   </p>
   <div className="flex flex-wrap gap-3">
    <button type="button" onClick={()=>void start()}
     disabled={running||matrix.complete}
     className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
     {chunks.length?'Reprendre le balayage':'Balayer les 3 PDF'}
    </button>
    <button type="button" onClick={stop} disabled={!running}
     className="rounded-md border border-slate-500 px-4 py-3 text-white disabled:opacity-50">
     Interrompre
    </button>
    <button type="button" onClick={reset} disabled={running}
     className="rounded-md border border-slate-500 px-4 py-3 text-white disabled:opacity-50">
     Réinitialiser
    </button>
   </div>
   <p role="status" aria-live="polite" className="text-sm text-cyan-100">{status}</p>
   {nextSource&&<p className="text-xs text-slate-400">
    Prochaine position : {nextSource.id} — page PDF {cursorRef.current.startPage}
   </p>}
   {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>
  <section className="space-y-3 rounded-lg border border-slate-700 p-5">
   <h2 className="text-lg font-semibold text-white">Couverture et matrice — non admises</h2>
   <p className="text-sm text-slate-300">
    Documents complètement parcourus : {matrix.coveredDocuments}/{matrix.totalDocuments}.
    Claims dans le périmètre : {matrix.claimCount}.
    Claims validés : {matrix.validatedClaimCount}.
   </p>
   <div className="space-y-2">
    {matrix.coverage.map(c=><p className="text-sm text-slate-300" key={c.sourceId}>
     <strong>{c.sourceId}</strong> — {c.scannedPages}/{c.pdfPageCount??'?'} pages ;
     {c.unextractableTextPages} page(s) sans texte extractible ;
     {c.complete?' couverture complète':' couverture partielle'}
    </p>)}
   </div>
   <div className="overflow-x-auto">
    <table className="w-full text-left text-sm text-slate-200">
     <thead><tr className="border-b border-slate-600">
      <th className="px-2 py-2">Claim</th><th className="px-2 py-2">Source</th>
      <th className="px-2 py-2">Observation de recherche</th>
      <th className="px-2 py-2">Pages candidates</th>
     </tr></thead>
     <tbody>{matrix.claims.map(row=><tr key={row.sourceId+row.claimId}
       className="border-b border-slate-700 align-top">
      <td className="px-2 py-2 font-mono">{row.claimId}</td>
      <td className="px-2 py-2 text-xs">{row.sourceId}</td>
      <td className="px-2 py-2">{row.searchStatus}</td>
      <td className="px-2 py-2">
       {row.observations.length?
        <details><summary className="cursor-pointer">
         {row.observations.length} occurrence(s)
        </summary>
         {row.observations.map((o,i)=><div key={o.pdfPage+'-'+i}
           className="my-2 border-t border-slate-700 py-2 text-xs">
          <strong>PDF page {o.pdfPage} · {o.contextHintObservation}</strong>
          <p>{o.excerpt}</p>
          <p className="break-all text-slate-500">Texte SHA-256 : {o.pageTextSha256}</p>
         </div>)}
        </details>:'—'}
      </td>
     </tr>)}</tbody>
    </table>
   </div>
   <p className="text-xs text-amber-100">
    Les indices de contexte ne sont que des filtres lexicaux : HINT_PRESENT ne
    prouve pas la validité du claim ; HINT_ABSENT n’établit pas qu’il est faux.
    Une absence de nombre peut refléter une limite d’extraction PDF.
    Les métriques dérivées requièrent une réconciliation de calcul distincte.
   </p>
   <div className="flex flex-wrap gap-3">
    <button type="button" onClick={()=>void copy()} disabled={!chunks.length}
     className="rounded-md border border-cyan-400 px-4 py-2 text-cyan-300 disabled:opacity-50">
     Copier le rapport JSON
    </button>
    <button type="button" onClick={download} disabled={!chunks.length}
     className="rounded-md border border-cyan-400 px-4 py-2 text-cyan-300 disabled:opacity-50">
     Exporter le JSON de réconciliation
    </button>
   </div>
  </section>
 </div>;
}
