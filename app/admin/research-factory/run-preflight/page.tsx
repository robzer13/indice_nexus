import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled,isHostedJ1RunFactoryEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {listContractProfileSummaries} from
 '@/lib/orotitan-equity/product/runfactory-contract-profile-v1';
import {RunFactoryDryRunClient} from './run-factory-dry-run-client';
import {parseUntrustedRunPreflightQuery,type RunPreflightQuery} from
 '@/lib/orotitan-equity/product/runfactory-run-preflight-handoff-v1';

export const metadata:Metadata={title:'OroTitan · J1 SHADOW · Demander une analyse'};
export const dynamic='force-dynamic';

export default async function RunFactoryIdentityPage({
 searchParams,
}:{searchParams:Promise<RunPreflightQuery>}){
 if(!await isAdminAuthenticated())redirect('/admin');
 if(!isHostedShadowResearchEnabled(process.env)&&
    !isHostedJ1RunFactoryEnabled(process.env))
  return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 const profiles=listContractProfileSummaries();
 const seed=parseUntrustedRunPreflightQuery(await searchParams);
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan · Run Factory P0-B · SHADOW
   </p>
   <h1 className="text-3xl font-semibold text-white">
    Demander une analyse — Run Factory J1
   </h1>
   <p className="text-sm text-slate-400">
    Une seule demande : résolution d’identité, plan INITIAL ou réutilisation, GO explicite, transaction et handoff Research.
   </p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-700 p-4 text-sm text-amber-100">
   <strong>SHADOW uniquement :</strong> la résolution et le préflight
   sont en lecture seule. Une seule autorisation GO explicite est exigée
   avant la RPC atomique INITIAL. Aucun stage Research ni publication
   ne démarre automatiquement. PRODUCTION reste hors périmètre.
  </section>
  {seed.isin&&<section className="rounded-lg border border-cyan-700 p-4 text-sm text-slate-200">
    Identité transmise depuis l’onboarding P0-D. Les valeurs sont des
    suggestions ; elles seront revérifiées directement dans le Registry
    SHADOW après sélection explicite du profil contractuel.
   </section>}
  <RunFactoryDryRunClient profiles={profiles} initialValues={seed}/>
  <div className="flex flex-wrap gap-6">
   <Link href="/admin/research-factory/onboarding"
    className="text-sm text-cyan-300 underline">
    Cas d’exception : revue d’identité P0-C
   </Link>
   <Link href="/admin/research-factory/contracts"
    className="text-sm text-cyan-300 underline">
    Consulter les contrats épinglés (P0-A)
   </Link>
   <Link href="/admin/research-portability/loreal"
    className="text-sm text-cyan-300 underline">
    Retour au pilote L’Oréal
   </Link>
  </div>
 </main>;
}
