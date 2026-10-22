/**
 * P0 v3.1 recovery worklist. A diagnostic, not an evidence verifier or runner.
 * Registry blocker_summary is untrusted OBSERVATION. Never infer readiness,
 * mark blockers closed, create artifacts, or promote a stage from this module.
 */
import type { OperatorRun } from "./pilotage-operator-view-v3-1";

export type RecoveryKind =
  | "EVIDENCE_CONTRADICTION" | "INDEPENDENT_CORROBORATION"
  | "TECHNOLOGY_ADOPTION" | "FINANCIAL_RECONCILIATION"
  | "PRIMARY_FILING" | "SHARE_DENOMINATOR" | "UNCLASSIFIED_EVIDENCE"
  | "MISSING_CHECKPOINT" | "MISSING_BLOCKER_DETAIL";
export type RecoveryTask = {
  key: string;
  observedBlocker: string;
  observedCode: string | null;
  kind: RecoveryKind;
  workInstructions: readonly string[];
  acceptanceEvidence: string;
  humanEscalation: string;
  status: "OPEN_UNVERIFIED";
  trust: "REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION";
};
export type RecoveryWorklist = {
  runId: string;
  stage: string;
  mayAdvance: false;
  mayPublish: false;
  mayMutate: false;
  verifiedResearchPerformed: false;
  tasks: RecoveryTask[];
};

type Recipe = Pick<RecoveryTask, "kind" | "workInstructions" | "acceptanceEvidence" | "humanEscalation">;
const ask = "Après recherche ciblée, recherche contradictoire et épuisement raisonnable des sources accessibles, demander le document précis, sa finalité et un substitut acceptable.";
const recipes: ReadonlyArray<[RegExp, Recipe]> = [
  [/(?:\b(?:C003|contradiction|conflict)\b|\d+\s+vs\s+\d+)/i, {
    kind:"EVIDENCE_CONTRADICTION",
    workInstructions:[
      "Localiser les deux déclarations primaires, leurs dates, définitions, périmètres et unités.",
      "Rechercher une réconciliation officielle puis une corroboration indépendante et contradictoire; conserver les deux valeurs tant que le conflit subsiste.",
    ],
    acceptanceEvidence:"Deux citations traçables et une réconciliation sourcée, ou un conflit persistant explicitement enregistré.",
    humanEscalation:ask,
  }],
  [/(?:independent|second.sourc|qualification|yield|refutation|réfutation|moat)/i,{
    kind:"INDEPENDENT_CORROBORATION",
    workInstructions:[
      "Identifier les hypothèses critiques, puis chercher des sources indépendantes et des éléments contraires aux affirmations de l'émetteur.",
      "Documenter les limites de qualification commerciale, de second sourcing ou de rendement fab sans assimiler annonce et adoption vérifiée.",
    ],
    acceptanceEvidence:"Registre des sources distinctes, qualité de leur indépendance et statut de chaque hypothèse après recherche contradictoire.",
    humanEscalation:ask,
  }],
  [/(?:adoption curve|technology adoption|\bHBM\b|photonics)/i,{
    kind:"TECHNOLOGY_ADOPTION",
    workInstructions:[
      "Séparer les marchés HBM, logique et photonique; expliciter unités, dénominateurs et horizons de chaque courbe.",
      "Chercher des métriques externes contradictoires et ne pas extrapoler de TAM ou de part de marché sans preuve.",
    ],
    acceptanceEvidence:"Modèle d'adoption segmenté, sources datées et hypothèses/lacunes explicitement bornées.",
    humanEscalation:ask,
  }],
  [/(?:\bROIC\b|invested capital|capital allocation|owner cash|owner earnings|tax benefits|development capital|\bFCF\b|net debt)/i,{
    kind:"FINANCIAL_RECONCILIATION",
    workInstructions:[
      "Rassembler états financiers audités, annexes et ponts de calcul aux dates du data cutoff.",
      "Reconstituer le numérateur, le dénominateur et les retraitements; isoler impôts atypiques, capital de développement et différences de périmètre.",
    ],
    acceptanceEvidence:"Calculs reproductibles avec périodes, unités, sources primaires et écarts résiduels non masqués.",
    humanEscalation:ask,
  }],
  [/(?:ESEF|audited|audit[ée]|annual report|full.document|rapport annuel)/i,{
    kind:"PRIMARY_FILING",
    workInstructions:[
      "Retrouver les documents d'origine (rapport audité et ESEF) de chaque exercice indiqué.",
      "Contrôler tableaux et notes en document complet, identifier les pages absentes et tenter une deuxième source d'accès.",
    ],
    acceptanceEvidence:"Documents originaux identifiés et contrôle croisé des exercices, avec pages et versions vérifiables.",
    humanEscalation:ask,
  }],
  [/(?:share count|denominator|dilut|treasury|options|nombre d.actions)/i,{
    kind:"SHARE_DENOMINATOR",
    workInstructions:[
      "Retracer capital émis, autocontrôle, options, conversions et événements entre les dates de référence.",
      "Séparer dénombrements économiques et juridiques; ne pas transporter un chiffre ancien vers la date de valorisation sans preuve.",
    ],
    acceptanceEvidence:"Pont de mouvements daté, sources primaires et dénominateur admis ou explicitement inconnu.",
    humanEscalation:ask,
  }],
];
const fallback: Recipe = {
  kind:"UNCLASSIFIED_EVIDENCE",
  workInstructions:[
    "Définir la pièce, le calcul ou la contradiction exacts derrière ce blocage; rechercher les sources primaires accessibles.",
    "Diversifier les sources, rechercher une réfutation, puis consigner l'issue sans modifier le verdict.",
  ],
  acceptanceEvidence:"Preuves traçables associées au même run et réévaluation documentée du blocage.",
  humanEscalation:ask,
};
const missingCheckpoint: Recipe = {
  kind:"MISSING_CHECKPOINT",
  workInstructions:[
    "Vérifier la disponibilité et les coordonnées du manifeste de l'étape courante dans le Registry.",
    "Si un artefact est enregistré, contrôler son dépôt privé, ses empreintes et son lineage par D2 avant toute reprise.",
  ],
  acceptanceEvidence:"Manifeste effectivement disponible et vérifié sur les octets; aucune simple déclaration d'existence.",
  humanEscalation:"Faire corriger le chemin de dépôt ou les droits d'accès par l'opérateur habilité; ne pas créer un faux manifeste.",
};
const missingDetails: Recipe = {
  kind:"MISSING_BLOCKER_DETAIL",
  workInstructions:[
    "Récupérer les métadonnées du checkpoint et ses preuves immuables par l'identité autorisée.",
    "Distinguer absence de diagnostic et preuve d'absence de blocage; ne pas émettre de GO.",
  ],
  acceptanceEvidence:"Cause et critères de reprise enregistrés et vérifiables.",
  humanEscalation:"Escalader le défaut de lecture du manifeste ou du registre au mainteneur, sans contournement de droits.",
};
function task(key: string, text: string, code: string|null, recipe: Recipe): RecoveryTask {
  return {key, observedBlocker:text,observedCode:code,...recipe,
    status:"OPEN_UNVERIFIED",trust:"REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION"};
}

export function buildRecoveryWorklist(run: OperatorRun): RecoveryWorklist {
  const tasks = run.blockers.map((blocker,index)=>{
    // Description is never executed as an instruction. Fixed recipes only.
    const recipe = recipes.find(([pattern])=>pattern.test(blocker.text))?.[1] ?? fallback;
    return task(String(index+1),blocker.text,blocker.code,recipe);
  });
  if (!run.activeManifestKind && (run.status==="ACTIVE" || run.status==="BLOCKED")) {
    tasks.push(task("MANIFEST","Aucun manifeste d'étape actif observé",null,missingCheckpoint));
  }
  if (tasks.length===0 && (run.status==="BLOCKED" || run.contractStatusCode !== null)) {
    tasks.push(task("DIAGNOSTIC","Blocage déclaré sans détail exploitable dans le Registry",run.contractStatusCode,missingDetails));
  }
  return {runId:run.runId,stage:run.stage,mayAdvance:false,mayPublish:false,
    mayMutate:false,verifiedResearchPerformed:false,tasks};
}
