import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {V34_SOURCES}
 from '@/lib/orotitan-equity/product/asml-research-private-pdf-reaudit-v34';
import {PdfAuditClient} from './pdf-audit-client';

export const metadata:Metadata={title:'OroTitan · Private PDF Research re-audit v3.4 · SHADOW'};
export const dynamic='force-dynamic';

export default async function PrivatePdfAuditPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 const enabled=isHostedShadowResearchEnabled(process.env);
 const options=Object.entries(V34_SOURCES).map(([id,s])=>({
  id,title:s.display,linkedClaims:[...s.evidenceClaims],
 }));
 return <main className="space-y-6">
   <header className="space-y-2">
     <p className="text-xs uppercase tracking-widest text-cyan-300">OroTitan · Research · SHADOW</p>
     <h1 className="text-3xl font-semibold text-white">Réaudit indépendant des PDF privés v3.4</h1>
     <p className="text-sm text-slate-400">
       RUN a1e59126-1ca9-4e55-8764-54a1e3d4dc24 · cutoff 2026-10-30
     </p>
   </header>
   <AdminNav/>
   <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
     <strong>Pré-admission uniquement.</strong> Recalcul de SHA-256 à partir des octets
     archivés ; le reçu initial ne tient pas lieu de preuve. Un SHA conforme ne
     démontre ni une signature de l&apos;émetteur ni l&apos;identité du fichier au cutoff.
     La recherche lexicale sur des pages ne constitue pas une validation du Evidence Ledger.
     Aucune écriture dans Supabase, aucun Registry, aucune transition de stage.
   </section>
   {enabled?<PdfAuditClient sources={options}/>:
     <p role="alert" className="text-rose-300">
       Audit désactivé hors de la Preview SHADOW isolée.
     </p>}
   <Link href="/admin/research-primary-sources" className="text-sm text-cyan-300 underline">
     Revenir à la capture primaire
   </Link>
 </main>;
}
