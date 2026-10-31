import Link from 'next/link';
import {redirect} from 'next/navigation';
import type {Metadata} from 'next';
import {AdminNav} from '@/components/admin/admin-nav';
import {isAdminAuthenticated} from '@/lib/auth/admin-session';
import {isHostedShadowResearchEnabled} from
 '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {runArithmeticChecks} from
 '@/lib/orotitan-equity/product/research-financial-arithmetic-qa-v1';
import {assessPrivatePdfSource} from
 '@/lib/orotitan-equity/product/research-private-source-readiness-v1';
import {LOREAL_PILOT2_CUTOFF,LOREAL_PILOT2_QA,LOREAL_PILOT2_SOURCES,
 LOREAL_PILOT2_PDF_PROBES,LOREAL_PILOT2_STATUS} from
 '@/lib/orotitan-equity/product/loreal-pilot2-candidate-config-v1';

export const metadata:Metadata={title:'OroTitan · L’Oréal pilot2 · portability SHADOW'};
export const dynamic='force-dynamic';
export default async function LorealPortabilityPreflightPage(){
 if(!await isAdminAuthenticated())redirect('/admin');
 const enabled=isHostedShadowResearchEnabled(process.env);
 if(!enabled)return <main className="p-6 text-rose-300">SHADOW_ONLY</main>;
 const qa=runArithmeticChecks(LOREAL_PILOT2_QA);
 const pdf=LOREAL_PILOT2_SOURCES.filter(s=>s.type==='PDF_CANDIDATE');
 const readiness=pdf.map(s=>assessPrivatePdfSource({
  id:s.id,title:s.title,publicationDate:s.date,expectedSha256:s.sha256,
  expectedByteSize:null,privateStoragePath:null,receiptSha256:null,
  storedBytesIndependentlyRehashed:false,privateReceiptIndependentlyVerified:false,
  captureTimestamp:null,
 },LOREAL_PILOT2_CUTOFF));
 const passed=qa.filter(q=>q.passed).length;
 const exact=qa.filter(q=>q.verdict==='EXACT_MATCH').length;
 const probes=Object.values(LOREAL_PILOT2_PDF_PROBES).reduce((s,p)=>s+p.length,0);
 return <main className="space-y-6">
  <header className="space-y-2">
   <p className="text-xs uppercase tracking-widest text-cyan-300">
    OroTitan · Research portability · SHADOW
   </p>
   <h1 className="text-3xl font-semibold text-white">
    L’Oréal — pilote n°2 : pré-admission et portabilité
   </h1>
   <p className="text-sm text-slate-400">
    OR · FR0000120321 · cutoff {LOREAL_PILOT2_CUTOFF} · RUN_ID NON PERSISTÉ
   </p>
  </header>
  <AdminNav/>
  <section className="rounded-lg border border-amber-600 p-4 text-sm text-amber-100 space-y-2">
   <p><strong>État : PILOTAGE_PRE_FLIGHT.</strong> Aucune donnée
    présentée sur cette page n’est une preuve admise.
    Les chiffres sont des transcriptions de communiqués officiels,
    pas des extractions de PDF privés scellés.</p>
   <p><strong>Bloqueurs :</strong> résolution de l’identité et du jeu
    exact de contrats V1 / V2-rc1 avant création de run ; capture,
    empreinte et reçu privé des PDF L’Oréal.</p>
   <p>Claims validés : {LOREAL_PILOT2_STATUS.evidenceClaimsValidated}.
    Registry : aucune admission. Deep Dive : NON ÉVALUÉ.
    Aucune écriture en base ou dans le stockage depuis cette page.</p>
  </section>
  <section className="rounded-lg border border-slate-700 p-5 space-y-3">
   <h2 className="text-xl font-semibold text-white">QA générique — contrôles déterministes</h2>
   <p className="text-sm text-slate-300">{passed}/{qa.length} rapprochements
    arithmétiques compatibles, dont {exact} exacts. Tolérance de précision
    calculée automatiquement à partir du nombre de termes ; aucune
    tolérance ad hoc propre à L’Oréal.</p>
   <div className="overflow-x-auto">
    <table className="w-full text-left text-sm text-slate-200">
     <thead><tr className="border-b border-slate-600">
      <th className="p-2">Contrôle</th><th className="p-2">Somme</th>
      <th className="p-2">Publié</th><th className="p-2">Écart</th>
      <th className="p-2">Tolérance</th><th className="p-2">Verdict</th>
     </tr></thead>
     <tbody>{qa.map(q=><tr key={q.id} className="border-b border-slate-700">
      <td className="p-2">{q.id}</td>
      <td className="p-2 font-mono">{q.calculatedSum}</td>
      <td className="p-2 font-mono">{q.reportedComparator}</td>
      <td className="p-2 font-mono">{q.difference}</td>
      <td className="p-2 font-mono">{q.roundingTolerance}</td>
      <td className="p-2">{q.verdict}</td>
     </tr>)}</tbody>
    </table>
   </div>
   <p className="text-xs text-amber-200">
    Un contrôle arithmétique réussi ne certifie pas la véracité d’une donnée
    transcrite ni la comparabilité de deux notions financières.
   </p>
  </section>
  <section className="rounded-lg border border-slate-700 p-5 space-y-3">
   <h2 className="text-xl font-semibold text-white">PDF primaires — éligibilité au scan</h2>
   <p className="text-sm text-slate-300">Moteur de sondes transposable :
    {probes} sondes configurées pour {pdf.length} documents,
    mais aucune extraction L’Oréal exécutée.</p>
   {readiness.map((r,i)=><div className="border-t border-slate-700 pt-3 text-sm"
    key={r.id}>
    <p className="font-semibold text-slate-100">{pdf[i].title}</p>
    <p className="text-amber-200">{r.state}</p>
    <p className="text-slate-400">SHA-256 original absent ;
     reçu privé absent ; scan bloqué.</p>
   </div>)}
  </section>
  <section className="rounded-lg border border-slate-700 p-5 space-y-2">
   <h2 className="text-xl font-semibold text-white">Portabilité vérifiable</h2>
   <p className="text-sm text-slate-300">Le moteur arithmétique
    et l’extracteur de nombres sont indépendants de l’émetteur.
    Le code ASML v3.4 réutilise désormais l’extracteur commun,
    sans relâcher les contrôles d’accès aux PDF privés.</p>
   <p className="text-sm text-slate-300">La capture PDF, les reçus privés,
    les preuves visuelles, les calculs IFRS de cash-flow
    et le run formel restent à développer ou à admettre.</p>
   <Link href="/admin/research-factory/contracts"
    className="mr-6 text-sm text-cyan-300 underline">
    Préflight des contrats Run Factory P0-A
   </Link>
   <Link href="/admin/research-primary-sources/reconcile"
    className="text-sm text-cyan-300 underline">
    Consulter le réconciliateur ASML v3.5
   </Link>
  </section>
 </main>;
}
