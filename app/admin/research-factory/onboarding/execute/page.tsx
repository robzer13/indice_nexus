import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {AtomicOnboardingClient} from '../atomic-onboarding-client';
import {parseUntrustedOnboardingDraft,type DraftQuery} from
 '@/lib/orotitan-equity/product/runfactory-onboarding-review-draft-v1';

export const metadata:Metadata={
 title:'OroTitan · Run Factory P0-D · Atomic SHADOW Onboarding',
};
export const dynamic='force-dynamic';

export default async function AtomicOnboardingPage({
 searchParams,
}:{searchParams:Promise<DraftQuery>}){

 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedShadowResearchEnabled(process.env))
  return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 const initialValues=parseUntrustedOnboardingDraft(await searchParams);
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan · Run Factory P0-D · SHADOW
   </p>
   <h1 className="text-3xl font-semibold text-white">
    Onboarding canonique transactionnel
   </h1>
   <p className="text-sm text-slate-400">
    Lecture préalable → sources revues → confirmation → RPC PostgreSQL
    atomique → reçu de persistance.
   </p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
   <strong>Action réelle sur le seul Registry SHADOW.</strong>
   Après confirmation, cette interface peut créer un émetteur, son titre coté,
   un dossier actif et une correspondance legacy si applicable.
   Elle ne crée aucun run, snapshot, score ni publication.
   Les URL de sources sont attestées manuellement, pas vérifiées en octets
   par cette fonction.
  </section>
  {initialValues&&<p className="rounded-lg border border-slate-700 p-3 text-sm text-slate-300">
    Dossier candidat prérempli. Les champs et liens sont des propositions
    non vérifiées par cette page. Confirmez leur exactitude avant
    toute autorisation. Aucun consentement n’est précoché.
   </p>}
  <AtomicOnboardingClient initialValues={initialValues}/>
  <div className="flex flex-wrap gap-5">
   <Link href="/admin/research-factory/onboarding"
    className="text-sm text-cyan-300 underline">
    Revenir au P0-C read-only
   </Link>
   <Link href="/admin/research-factory/contracts"
    className="text-sm text-cyan-300 underline">
    P0-A Contrats
   </Link>
  </div>
 </main>;
}
