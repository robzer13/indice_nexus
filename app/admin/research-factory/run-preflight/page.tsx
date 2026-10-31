import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {listContractProfileSummaries} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';
import {RunFactoryDryRunClient} from './run-factory-dry-run-client';

export const metadata:Metadata={title:'OroTitan · Run Factory P0-B · Identity SHADOW dry-run'};
export const dynamic='force-dynamic';

export default async function RunFactoryIdentityPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedShadowResearchEnabled(process.env))
  return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 const profiles=listContractProfileSummaries();
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan · Run Factory P0-B · SHADOW
   </p>
   <h1 className="text-3xl font-semibold text-white">
    Préflight d’identité, de cutoff et de run existant
   </h1>
   <p className="text-sm text-slate-400">
    Préparation sans écriture. Réutilise le Registry et les packs existants.
   </p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
   <strong>Limite de sécurité :</strong> cette page ne crée jamais de run.
   Le bootstrap officiel V3.0.3 requiert une vérification de production ;
   il n’est pas exécuté ni simulé comme « production valide » depuis SHADOW.
   L’exactitude de l’identité, la résolution des contrats et l’autorisation
   GO doivent être établies avant l’appel contrôlé aux RPC existantes.
  </section>
  <RunFactoryDryRunClient profiles={profiles}/>
  <div className="flex flex-wrap gap-6">
   <Link href="/admin/research-factory/contracts"
    className="text-sm text-cyan-300 underline">
    Vérifier les contrats (P0-A)
   </Link>
   <Link href="/admin/research-portability/loreal"
    className="text-sm text-cyan-300 underline">
    Retour au pilote L’Oréal
   </Link>
  </div>
 </main>;
}
