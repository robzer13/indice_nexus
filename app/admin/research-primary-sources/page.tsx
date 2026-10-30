import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {ASML_V29_SOURCES,ASML_V29_RUN,ASML_V29_CUTOFF,type AsmlV29SourceId}
 from '@/lib/orotitan-equity/product/asml-official-pdf-capture-v29';
import {asmlCaptureErrorLabel}
 from '@/lib/orotitan-equity/product/asml-capture-error-diagnostics-v291';
export const metadata:Metadata={title:'ASML / TSMC · Capture primaire v2.9.1 · SHADOW'};
export const dynamic='force-dynamic';
type Params=Promise<{result?:string;error?:string;source?:string}>;

export default async function PrimarySourcesPage({searchParams}:{searchParams:Params}){
  if(!await isAdminAuthenticated())redirect('/admin');
  const enabled=isHostedShadowResearchEnabled(process.env);
  const q=await searchParams;
  const activeSource:AsmlV29SourceId=q.source&&Object.prototype.hasOwnProperty.call(ASML_V29_SOURCES,q.source)
    ?q.source as AsmlV29SourceId:'ASML_2025_US_GAAP';
  const reported=(q.result??'').split(':');
  const verifiedResult=reported.length===2&&
    Object.hasOwn(ASML_V29_SOURCES,reported[0])&&/^[a-f0-9]{64}$/.test(reported[1]);
  return <main className="space-y-6">
    <header className="space-y-2">
      <p className="text-xs uppercase tracking-widest text-cyan-300">OroTitan · Research · SHADOW</p>
      <h1 className="text-3xl font-semibold text-white">Capture primaire des documents ASML / TSMC</h1>
      <p className="text-sm text-slate-400">RUN {ASML_V29_RUN} · cutoff {ASML_V29_CUTOFF}</p>
    </header>
    <AdminNav/>
    <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
      Télécharger uniquement des documents autorisés sur les sites émetteurs.
      Le téléchargement par HTTPS et la relecture SHA-256 assurent la traçabilité de la copie,
      mais ne constituent pas une signature numérique du producteur du document.
      <strong> Aucun Registry, aucune transition de stage, aucune publication.</strong>
    </section>
    {!enabled?<p role="alert" className="text-rose-300">Fonction désactivée hors Preview SHADOW isolée.</p>:<>
      <section className="rounded-lg border border-slate-700 p-5 space-y-4">
        <h2 className="text-lg font-semibold text-white">Sources sur liste blanche</h2>
        <form action="/admin/research-primary-sources/capture" method="post" className="space-y-4">
          <label htmlFor="sourceId" className="block text-sm text-slate-200">Document officiel</label>
          <select id="sourceId" name="sourceId" required defaultValue={activeSource}
            className="block w-full rounded-md bg-slate-800 px-3 py-3 text-white">
            {Object.values(ASML_V29_SOURCES).map(s=><option key={s.id} value={s.id}>
              {s.title}
            </option>)}
          </select>
          <button type="submit" className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950">
            Capturer et vérifier en privé
          </button>
        </form>
        {verifiedResult&&<p role="status" className="break-all font-mono text-xs text-emerald-200">
          Reçu SHA-256 vérifié en SHADOW : {q.result} (non admis)
        </p>}
        {q.error&&<div role="alert" className="space-y-1 text-sm text-rose-300">
          <p><strong>Capture non confirmée :</strong> {asmlCaptureErrorLabel(q.error)}</p>
          <p>Document concerné : {ASML_V29_SOURCES[activeSource].title}</p>
          <p>Aucun fichier ne doit être tenu pour admis au Registry. Le détail brut de l'erreur n’est pas exposé.</p>
        </div>
      </section>
      <p className="text-sm text-slate-400">Les liens sont codés en dur dans la liste blanche.
        Si un émetteur redirige vers une autre origine, la capture est bloquée et l&apos;origine doit être examinée.
      </p>
    </>}
    <Link href="/admin/research-q3-sources" className="text-sm text-cyan-300 underline">
      Capture manuelle existante du PDF ASML T3 2026
    </Link>
  </main>;
}
