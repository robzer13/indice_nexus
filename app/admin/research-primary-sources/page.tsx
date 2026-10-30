import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {ASML_V29_SOURCES,ASML_V29_RUN,ASML_V29_CUTOFF}
 from '@/lib/orotitan-equity/product/asml-official-pdf-capture-v29';
export const metadata:Metadata={title:'ASML · Capture primaire v2.9 · SHADOW'};
export const dynamic='force-dynamic';
type Params=Promise<{result?:string;error?:string}>;

export default async function PrimarySourcesPage({searchParams}:{searchParams:Params}){
  if(!await isAdminAuthenticated())redirect('/admin');
  const enabled=isHostedShadowResearchEnabled(process.env);
  const q=await searchParams;
  const reported=(q.result??'').split(':');
  const verifiedResult=reported.length===2&&
    Object.hasOwn(ASML_V29_SOURCES,reported[0])&&/^[a-f0-9]{64}$/.test(reported[1]);
  return <main className="space-y-6">
    <header className="space-y-2">
      <p className="text-xs uppercase tracking-widest text-cyan-300">OroTitan · Research · SHADOW</p>
      <h1 className="text-3xl font-semibold text-white">Capture primaire des documents ASML</h1>
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
          <select id="sourceId" name="sourceId" required
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
        {q.error&&<p role="alert" className="text-sm text-rose-300">
          Capture non confirmée. Aucun fichier ne doit être tenu pour admis au Registry.
        </p>}
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
