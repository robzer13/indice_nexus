import type {PreJalonResult} from
 '@/lib/orotitan-equity/product/pre-jalon-readiness-v1';

type Row=Readonly<{title:string;readiness:PreJalonResult}>;
const explanations:Record<string,string>={
 PREDECESSOR_C8_NOT_ACCEPTED:'Jalon précédent non accepté via C8',
 PREDECESSOR_OPEN_BLOCKER:'Incident bloquant sur le jalon précédent',
 PREDECESSOR_INCIDENT_STATE_UNVERIFIED:'État des incidents non vérifié',
 EXACT_EIGHT_CRITERIA_MISSING:'Huit critères exacts non vérifiés',
 IMMUTABLE_DESIGN_PLAN_MISSING:'Plan de conception versionné absent',
 PLAN_IMMUTABILITY_NOT_VERIFIED:'Référence ou empreinte du plan non vérifiée',
 DESIGN_PLAN_INCOMPLETE:'Plan fonctionnel, alternatives ou tests incomplets',
 INDEPENDENT_REVIEW_NOT_VERIFIED:'Relecture indépendante non prouvée',
 PLAN_REQUIRES_REVISION:'Plan à réviser',
 PLAN_REVIEW_NOT_APPROVED:'Plan non approuvé',
 DISTINCT_GO_BUILD_MISSING:'GO BUILD distinct absent',
 GO_BUILD_NOT_VERIFIED:'GO BUILD non vérifié',
};
export function PreJalonPanel({entries}:{entries:readonly Row[]}){
 return <section className="rounded-xl border border-cyan-800 bg-slate-950 p-5 space-y-3">
  <h2 className="text-xl font-semibold text-white">
   Functionality First · Pré-jalon J1–J8
  </h2>
  <p className="text-sm text-slate-300">
   Vue de lecture du contrôle de préparation. La recherche et la rédaction
   d’un brouillon sont possibles ; un plan approuvé n’est jamais une
   acceptation C8 ni un GO de démarrage. Aucun changement du Registry.
  </p>
  <div className="grid gap-3 md:grid-cols-2">
   {entries.map(({title,readiness:r})=><article key={r.milestoneCode}
    className="rounded-lg border border-slate-700 p-3 space-y-2">
    <h3 className="font-medium text-white">{r.milestoneCode} · {title}</h3>
    <p className="text-sm text-amber-200">
     {r.status} · Construction : {r.buildAuthorized?'AUTORISÉE':'INTERDITE'}
    </p>
    <p className="text-xs text-slate-400">
     Prédécesseur : {r.predecessor} · Plan :
     {r.planApproved?'vérifié':'non approuvé'}
    </p>
    <details className="text-xs text-slate-300">
     <summary className="cursor-pointer">Conditions non satisfaites ({r.blockers.length})</summary>
     <ul className="mt-2 list-disc pl-5 space-y-1">
      {r.blockers.map(code=><li key={code}>
       {explanations[code]??code}</li>)}
     </ul>
    </details>
   </article>)}
  </div>
  <p className="text-xs text-slate-400">
   Première tranche : pré-jalon en lecture seule ; tant qu’aucune vérification
   officielle de plan, de relecteur et de GO n’est raccordée au serveur,
   ces champs restent absents et la construction reste bloquée.
  </p>
 </section>;
}
