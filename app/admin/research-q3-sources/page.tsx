import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {getHostedShadowRecoveryClient} from '@/lib/orotitan-equity/product/research-inbox-pilot-server';
import {ASML_Q3_ARCHIVE_ROOT,ASML_Q3_PRIVATE_BUCKET}
 from '@/lib/orotitan-equity/product/asml-q3-private-source-archive-v1';
import {ASML_Q3_US_GAAP_PDF,type UserSuppliedQ3PdfReceipt}
 from '@/lib/orotitan-equity/product/asml-q3-browser-upload-v1';
export const dynamic='force-dynamic';
export const metadata:Metadata={title:'ASML Q3 2026 · Capture privée SHADOW'};
type Query=Promise<{result?:string;error?:string}>;

async function receipts():Promise<UserSuppliedQ3PdfReceipt[]> {
 const client=getHostedShadowRecoveryClient();
 const {data:b,error:be}=await client.storage.getBucket(ASML_Q3_PRIVATE_BUCKET);
 if(be||!b||b.public!==false)throw Error('UNSAFE_BUCKET');
 const directory=ASML_Q3_ARCHIVE_ROOT+'/browser-receipts';
 const {data,error}=await client.storage.from(ASML_Q3_PRIVATE_BUCKET)
   .list(directory,{limit:30});
 if(error||!data||data.length>=30)throw Error('CAPTURE_LIST_UNAVAILABLE');
 const items:UserSuppliedQ3PdfReceipt[]=[];
 for(const item of data){
  if(!/^[a-f0-9]{64}\.json$/.test(item.name))throw Error('BAD_RECEIPT_NAME');
  const {data:blob,error:dlError}=await client.storage.from(ASML_Q3_PRIVATE_BUCKET)
    .download(directory+'/'+item.name);
  if(dlError||!blob)throw Error('RECEIPT_UNAVAILABLE');
  const content=Buffer.from(await blob.arrayBuffer());
  if(content.length>4000)throw Error('RECEIPT_TOO_LARGE');
  const record=JSON.parse(content.toString('utf8')) as UserSuppliedQ3PdfReceipt;
  const expectedSha=item.name.slice(0,64);
  if(record.schema!=='ASML_Q3_BROWSER_SUPPLIED_PDF_RECEIPT_V1' ||
     record.documentId!==ASML_Q3_US_GAAP_PDF.id ||
     record.sourceLink!==ASML_Q3_US_GAAP_PDF.officialLink ||
     record.sourcePath!==ASML_Q3_ARCHIVE_ROOT+'/browser-supplied/'+expectedSha+'.pdf' ||
     record.receiptPath!==directory+'/'+item.name ||
     record.sha256!==expectedSha ||
     record.publicationDate!=='2026-10-14'||record.cutoff!=='2026-10-30' ||
     record.storageReadbackVerified!==true ||
     record.sourceOriginAuthenticated!==false ||
     record.sourceDocumentAuthenticityVerified!==false ||
     record.registryAdmitted!==false ||
     !Number.isFinite(record.byteSize)||record.byteSize<10000||record.byteSize>24000000)
     throw Error('RECEIPT_INTEGRITY_UNVERIFIED');
  items.push(record);
 }
 return items.sort((a,b)=>b.uploadedAt.localeCompare(a.uploadedAt));
}

export default async function AsmlQ3Sources({searchParams}:{searchParams:Query}) {
 if(!await isAdminAuthenticated())redirect('/admin');
 const enabled=isHostedShadowResearchEnabled(process.env);
 const params=await searchParams;
 let list:UserSuppliedQ3PdfReceipt[]=[];
 let readFailed=false;
 if(enabled){try{list=await receipts();}catch{readFailed=true;}}
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs font-semibold uppercase tracking-widest text-cyan-400">OroTitan · Research · SHADOW</p>
   <h1 className="text-3xl font-semibold text-white">ASML Q3 2026 — Capture privée des sources</h1>
   <p className="text-sm text-slate-400">Pré-admission du RUN <code>a1e59126-1ca9-4e55-8764-54a1e3d4dc24</code>, cutoff 30 octobre 2026.</p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-800 bg-amber-950/10 p-4 text-sm text-amber-100">
   La SEC bloque les requêtes automatisées (HTTP 403). Ce formulaire transporte uniquement un PDF
   que vous avez obtenu vous-même depuis ASML. L’authenticité et la provenance du fichier transmis
   restent <strong>NON VÉRIFIÉES</strong> : l’archivage garantit les octets reçus et leur conservation,
   pas l’origine du document. Aucun artifact Registry n’est créé. Aucun passage en Deep Dive.
  </section>
  {!enabled?<div role="alert" className="text-amber-200">Capture désactivée hors SHADOW Preview.</div>:<>
   <section className="space-y-4 rounded-lg border border-slate-800 bg-slate-900/60 p-5">
    <h2 className="text-lg font-semibold text-white">1. Obtenir le document officiel</h2>
    <p className="text-sm text-slate-300">Sur le site investisseurs ASML, rubrique <strong>Q3 2026 financial results</strong>, choisir <strong>Financial statements US GAAP PDF</strong>.</p>
    <div className="space-x-4 text-sm text-cyan-300">
     <a className="underline" href="https://www.investor.asml.com/quarterly-results" target="_blank" rel="noopener noreferrer">Résultats trimestriels ASML</a>
     <a className="underline" href={ASML_Q3_US_GAAP_PDF.officialLink} target="_blank" rel="noopener noreferrer">Lien PDF officiel</a>
    </div>
    <h2 className="text-lg font-semibold text-white">2. Archiver la copie téléchargée</h2>
    <form action="/admin/research-q3-sources/upload" method="post" encType="multipart/form-data" className="space-y-4">
     <label className="block text-sm text-slate-200" htmlFor="q3pdf">PDF officiel ASML, taille maximale 24 Mo</label>
     <input id="q3pdf" name="document" type="file" accept=".pdf,application/pdf" required
       className="block w-full text-sm text-slate-300"/>
     <button type="submit" className="rounded-lg bg-cyan-400 px-5 py-3 font-semibold text-slate-950">
       Archiver la source en privé et vérifier SHA-256
     </button>
    </form>
    {params.result==='stored_sha_verified'&&<p role="status" className="text-emerald-200">Fichier archivé : relecture et SHA-256 conformes, provenance et admission non certifiées.</p>}
    {(params.error||readFailed)&&<p role="alert" className="text-rose-200">Archivage ou lecture refusé. Aucune preuve n’est admise. {params.error??''}</p>}
   </section>
   <section className="space-y-3 rounded-lg border border-slate-800 bg-slate-900/60 p-5">
    <h2 className="text-lg font-semibold text-white">Reçus d’archivage privés</h2>
    <p className="text-xs text-slate-400">{list.length} copie(s) enregistrée(s), non admise(s). Chaque reçu pointe vers les octets réels conservés dans le bucket privé SHADOW.</p>
    {list.map(item=><article key={item.sha256} className="rounded-lg border border-slate-700 p-3 text-sm text-slate-300">
      <p>PDF reçu le {item.uploadedAt} · {item.byteSize.toLocaleString('fr-FR')} octets</p>
      <p className="break-all font-mono text-xs">sha256:{item.sha256}</p>
      <p className="text-xs text-amber-200">Stockage intègre · Origine non attestée · Registry non admis</p>
     </article>)}
   </section>
   <Link href="/admin/research-recovery" className="text-sm text-cyan-300 underline">Retour au Recovery Loop V1</Link>
  </>}
 </main>;
}
