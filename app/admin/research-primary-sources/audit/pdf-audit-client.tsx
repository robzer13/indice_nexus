'use client';

import {useState} from 'react';
type SourceOption={id:string;title:string;linkedClaims:string[]};
type AuditResponse={
 ok?:boolean;mode?:string;checkedAt?:string;error?:string;
 independentByteRehashCount?:number;registryAdmitted?:false;
 sources?:Array<{id:string;sourceOriginalHash:string;receiptHash:string;byteSize:number;
  integrity:string;sourceDate:string;captureTimestamp:string;evidenceClaims:string[];
  independentlyAuthenticatedPublisher:false;historicalBytesAtCutoffCertified:false}>;
 source?:{id:string;sourceOriginalHash:string;integrity:string};
 scan?:{pdfPageCount:number;startPage:number;endPage:number;pagesNotScanned:number;
  pageObservations:Array<{pdfPage:number;textLength:number;textSha256:string;
  candidateMatches:Array<{claimId:string;excerpt:string;match:string}>}>};
};
export function PdfAuditClient({sources}:{sources:SourceOption[]}){
 const [selected,setSelected]=useState(sources[0]?.id||'ASML_2025_US_GAAP');
 const [startPage,setStartPage]=useState('1');
 const [pages,setPages]=useState('12');
 const [pending,setPending]=useState(false);
 const [result,setResult]=useState<AuditResponse|null>(null);
 const [message,setMessage]=useState('');
 async function run(mode:'rehash_all'|'scan_pages'){
  if(pending)return;
  setPending(true);setResult(null);setMessage('');
  try{
   const payload=mode==='rehash_all'?{mode}:
     {mode,sourceId:selected,startPage:Number(startPage),pageCount:Number(pages)};
   const response=await fetch('/admin/research-primary-sources/audit',{
    method:'POST',credentials:'same-origin',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),cache:'no-store',
   });
   const obj=await response.json() as AuditResponse;
   setResult(obj);
   if(!response.ok||!obj.ok)setMessage('Audit non validé : '+(obj.error||'réponse invalide'));
  }catch{
   setMessage('Audit indisponible : la requête ou le serveur a échoué.');
  }finally{setPending(false);}
 }
 return <div className="space-y-6">
   <section className="space-y-3 rounded-lg border border-slate-700 p-5">
     <h2 className="text-xl font-semibold text-white">1. Recalcul indépendant des trois SHA-256</h2>
     <p className="text-sm text-slate-300">
       Lecture privée des trois PDF et de leurs reçus depuis SHADOW, puis recalcul SHA-256,
       sans utiliser le code de capture. Aucun octet ni reçu n&apos;est modifié.
     </p>
     <button type="button" disabled={pending} onClick={()=>void run('rehash_all')}
       className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
       {pending?'Vérification en cours':'Recalculer les 3 empreintes en lecture seule'}
     </button>
   </section>
   <section className="space-y-3 rounded-lg border border-slate-700 p-5">
     <h2 className="text-xl font-semibold text-white">2. Rapprochement exploratoire par pages</h2>
     <p className="text-sm text-slate-300">
       Numérotation des pages PDF à partir de 1. Maximum 20 pages par requête.
       Les nombres repérés sont seulement des candidats lexicaux, jamais des affirmations admises.
     </p>
     <label className="block text-sm text-slate-200" htmlFor="sourceId">Document archivé</label>
     <select id="sourceId" value={selected} onChange={e=>setSelected(e.target.value)}
       className="w-full rounded-md bg-slate-800 px-3 py-3 text-white">
       {sources.map(s=><option key={s.id} value={s.id}>{s.title} — {s.linkedClaims.length} affirmations liées</option>)}
     </select>
     <div className="flex flex-wrap gap-4">
       <label className="space-y-1 text-sm text-slate-200">
         Première page PDF
         <input className="block w-36 rounded-md bg-slate-800 px-3 py-2 text-white"
           type="number" min={1} max={2000} value={startPage}
           onChange={e=>setStartPage(e.target.value)}/>
       </label>
       <label className="space-y-1 text-sm text-slate-200">
         Nombre de pages
         <input className="block w-36 rounded-md bg-slate-800 px-3 py-2 text-white"
           type="number" min={1} max={20} value={pages}
           onChange={e=>setPages(e.target.value)}/>
       </label>
     </div>
     <button type="button" disabled={pending} onClick={()=>void run('scan_pages')}
       className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
       Examiner ces pages et leurs valeurs candidates
     </button>
   </section>
   {message&&<p role="alert" className="text-sm text-rose-300">{message}</p>}
   {result?.ok&&<section className="space-y-3 rounded-lg border border-slate-700 p-5">
     <h2 className="text-lg font-semibold text-white">Résultats techniques — non admis</h2>
     <p className="text-sm text-amber-100">
       Relecture des octets conservés uniquement. Aucune authentification cryptographique de
       l&apos;émetteur ni certification historique au cutoff. Aucune admission au Registry.
     </p>
     {result.sources?.map(s=><article key={s.id} className="space-y-1 border-t border-slate-700 py-3">
       <p className="font-semibold">{s.id}</p>
       <p className="text-sm text-emerald-300">{s.integrity} — {s.byteSize.toLocaleString('fr-FR')} octets</p>
       <p className="break-all font-mono text-xs">SHA256: {s.sourceOriginalHash}</p>
       <p className="text-xs text-slate-400">Capture : {s.captureTimestamp}, reçu intègre, non admis</p>
     </article>)}
     {result.scan&&<div className="space-y-3">
       <p className="text-sm text-slate-300">
         Document : {result.source?.id} — {result.scan.pdfPageCount} pages au total.
         Pages examinées : {result.scan.startPage}–{result.scan.endPage}.
       </p>
       {result.scan.pageObservations.map(p=><article key={p.pdfPage}
          className="space-y-1 rounded-md border border-slate-700 p-3">
         <p className="font-semibold">Page PDF {p.pdfPage} · {p.textLength} caractères · {p.candidateMatches.length} nombre(s) candidat(s)</p>
         <p className="break-all font-mono text-xs text-slate-500">Empreinte du texte : {p.textSha256}</p>
         {p.candidateMatches.map(m=><p key={m.claimId} className="text-xs text-slate-300">
           <strong>{m.claimId} (correspondance numérique seulement)</strong> — {m.excerpt}
         </p>)}
       </article>)}
     </div>}
     <button type="button"
       className="rounded-md border border-slate-500 px-4 py-2 text-sm text-cyan-300"
       onClick={()=>void navigator.clipboard.writeText(JSON.stringify(result,null,2))}>
       Copier le rapport JSON de vérification
     </button>
   </section>}
 </div>;
}
