import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {V34_PROBES,V34_SOURCES,type V34SourceId}
 from '@/lib/orotitan-equity/product/asml-research-private-pdf-reaudit-v34';
import type {V35Source} from
 '@/lib/orotitan-equity/product/asml-research-claim-matrix-v35';
import {ClaimReconcilerClient} from './claim-reconciler-client';

export const metadata:Metadata={
 title:'OroTitan · ASML private PDF claim reconciliation v3.5 · SHADOW'
};
export const dynamic='force-dynamic';

const CONTEXT_HINT_OVERRIDES:Readonly<Record<string,string>>={
 'C-010':'factoring',
 'C-011':'down payments',
};
export default async function PrivateClaimReconciliationPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 const enabled=isHostedShadowResearchEnabled(process.env);
 const options:V35Source[]=Object.entries(V34_SOURCES).map(([id,s])=>({
  id,title:s.display,sourceSha256:s.sha256,
  probes:V34_PROBES[id as V34SourceId].map(p=>({
   claimId:p.claimId,numeric:p.numeric,derived:p.derived===true,
   contextHint:CONTEXT_HINT_OVERRIDES[p.claimId]??p.context??null,
  })),
 }));
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">OroTitan · Research · SHADOW</p>
   <h1 className="text-3xl font-semibold text-white">ASML — Réconciliateur documentaire v3.5</h1>
   <p className="text-sm text-slate-400">
    RUN a1e59126-1ca9-4e55-8764-54a1e3d4dc24 · cutoff 2026-10-30
   </p>
  </header>
  <AdminNav/>
  {enabled?<ClaimReconcilerClient sources={options}/>:
   <p role="alert" className="text-rose-300">
    Réconciliation désactivée hors de la Preview SHADOW isolée.
   </p>}
  <Link href="/admin/research-primary-sources/audit"
    className="text-sm text-cyan-300 underline">
    Retour à l’audit SHA-256 et pages PDF v3.4
  </Link>
 </main>;
}
