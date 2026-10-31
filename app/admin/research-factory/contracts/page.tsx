import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {listContractProfileSummaries} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';
import {ContractPreflightClient} from './contract-preflight-client';

export const metadata:Metadata={
 title:'OroTitan · Run Factory P0-A · Contract Profile Preflight · SHADOW'
};
export const dynamic='force-dynamic';

export default async function ContractPreflightPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedShadowResearchEnabled(process.env))
  return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 const profiles=listContractProfileSummaries();
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan · Run Factory · SHADOW
   </p>
   <h1 className="text-3xl font-semibold text-white">
    Préflight contractuel P0-A — indépendant de l’entreprise
   </h1>
   <p className="text-sm text-slate-400">
    Réutilisation des packs V1 / V2 / V3.1 existants. Lecture seule.
   </p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
   <strong>Pré-admission seulement.</strong> Le contrôle structurel du pack
   et sa correspondance au SHA-256 global sont effectués au chargement.
   Le contrôle octet par octet des contrats nécessite le bouton explicite
   ci-dessous ; il est réalisé sur le commit Git immuable référencé.
   Ce service ne crée ni issuer, ni run, ni stage ni artefact Registry.
   Il n’autorise pas de sélection automatique de la méthodologie.
  </section>
  <ContractPreflightClient profiles={profiles}/>
  <Link href="/admin/research-portability/loreal"
   className="text-sm text-cyan-300 underline">
   Retour au pilote L’Oréal
  </Link>
 </main>;
}
