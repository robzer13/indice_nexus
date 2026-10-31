import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {CanonicalOnboardingClient} from './canonical-onboarding-client';

export const metadata:Metadata={
 title:'OroTitan · Run Factory P0-C · Governed Onboarding SHADOW',
};
export const dynamic='force-dynamic';
export default async function CanonicalOnboardingPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedShadowResearchEnabled(process.env))
  return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan · Run Factory P0-C · SHADOW
   </p>
   <h1 className="text-3xl font-semibold text-white">
    Préparer l’identité canonique sans recréer l’entreprise
   </h1>
   <p className="text-sm text-slate-400">
    ISIN + titre + marché → détection des doublons → rapprochement legacy/canonique
    → candidat de revue.
   </p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
   <strong>Onboarding préparatoire, pas création de société.</strong>
   Le SHADOW contient les identités de test et ne constitue pas un miroir
   du Registry de production. Une absence constatée ici n’autorise pas
   l’insertion dans la base de production. Aucun lien URL n’est vérifié
   comme preuve tant que l’extracteur de sources officielles ne l’a pas
   effectivement consulté et authentifié.
  </section>
  <CanonicalOnboardingClient/>
  <div className="flex flex-wrap gap-6">
   <Link href="/admin/research-factory/onboarding/execute"
    className="text-sm text-cyan-300 underline">
    P0-D : onboarding transactionnel SHADOW
   </Link>
   <Link href="/admin/research-factory/run-preflight"
    className="text-sm text-cyan-300 underline">
    P0-B : préflight du run
   </Link>
   <Link href="/admin/research-factory/contracts"
    className="text-sm text-cyan-300 underline">
    P0-A : contrats
   </Link>
  </div>
 </main>;
}
