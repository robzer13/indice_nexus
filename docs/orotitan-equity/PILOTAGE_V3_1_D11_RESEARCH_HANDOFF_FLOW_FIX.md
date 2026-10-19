# OroTitan D11 — Research → Fundamentals : corriger la fluidité sans ouvrir les gates

## Défaut démontré

Le contrat effectivement épinglé pour Research V3.1 est
`OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0`, section 8, dans
`contracts/orotitan-equity/v2/`. Ses sept sorties normales sont :

- `RESEARCH_SOURCE_MANIFEST`
- `EVIDENCE_LEDGER`
- `CONFLICT_LEDGER` (ou référence à l'autorité exacte selon le contrat)
- `MATERIAL_RESEARCH_HYPOTHESIS_REGISTER`
- `RESEARCH_GAP_REGISTER`
- `INPUT_SUFFICIENCY_ASSESSMENT`
- `CLASSIFICATION_INPUT_RECORD`

Les couches D1 et D2 de Pilotage demandaient à tort deux types distincts :
`DD_INPUT_SUFFICIENCY_RECORD` et `ANALYSIS_INPUT_LOCK`. Un Research FINAL
complètement conforme aurait été rejeté lors d'un handoff vers Fundamentals
faute de ces deux aliases non spécifiés par l'autorité Research épinglée.

## Correction isolée, non déployée

Une constante commune reflète les sept noms exacts du contrat V2.
Le vérificateur D2 (artefacts signés, contenu référencé par le FINAL) et
le planificateur D1 partagent maintenant cette source de vérité.
Les tests refusent les aliases historiques incorrects. **Il ne s'agit ni
d'un changement du contrat gelé, ni d'une tolérance à l'absence de preuve.**

L'entrée en Fundamentals continue à exiger :
`RESEARCH = COMPLETE`, `FINAL Research Stage Manifest` vérifié,
`READY_FOR_DEEP_DIVE = YES`, `FINAL_OUTPUTS_VERIFIED`, sept sorties
résolues sans ambiguïté et SHA vérifiés, avec révision du stage non obsolète.
Aucune publication, transformation de CHECKPOINT en FINAL ou création de
preuve n'est introduite.

## Besi : état réel et prochaine action

Run `48b9426b-4d0d-4858-9d87-efcd13ac415d`, cutoff
`2026-10-12` : **ACTIVE / RESEARCH IN_PROGRESS / CHECKPOINT v4 /
READY_FOR_DEEP_DIVE = NO / BLOCKED_INSUFFICIENT_INPUT**.
Le Registry enregistre six blocages de Research et le
`INPUT_SUFFICIENCY_ASSESSMENT` actif ne déclare pas la suffisance.
Le correctif D11 **ne change pas** ces faits.

Ce run possède déjà ses registres Research versionnés et un ensemble de
modèles financiers externes non admis au Registry : ces modèles ne peuvent
pas remplacer des preuves scellées. Pour avancer réellement :

1. Reconciler C003 : 18 contre 15 clients Besi hybrid bonding fin 2025,
   ou maintenir un conflit matériel avec une limitation probante.
2. Vérifier si les données financières collectées permettent l'étude du
   ROIC et des Owner Earnings en **Fundamentals**, sans exiger l'achèvement
   analytique complet de ces calculs pendant Research.
3. Vérifier l'indépendance et les bornes des preuves commerciales (G003/G009)
   et les hypothèses d'adoption HBM, logique et photonique (G004).
4. Réévaluer la suffisance, la couverture et les conflits sous l'autorité
   Research V2, puis produire de nouvelles **versions** des registres,
   sans écraser les versions précédentes.
5. Seulement si l'audit rend `READY_FOR_DEEP_DIVE = YES` légitime :
   assembler les sept sorties FINAL, garantir leur hash et leur lignée,
   enregistrer le FINAL et faire passer le run à DEEP_DIVE par le mécanisme
   CAS officiellement autorisé. La PR D11 **ne réalise aucune de ces écritures**.

## Limites explicites

Les rapports Excel/PDF V2 et V3 produits hors des artefacts Research doivent
être qualifiés et rapprochés ; ils ne deviennent pas automatiquement une
preuve autoritative. Les mêmes chiffres ne doivent pas être utilisés comme
ROIC/Owner Earnings certifiés avant Fundamentals et ses contrôles.

Le diagnostic `CONTINUE / RESEARCH_ADMITTED` signifie autorisation de
**continuer Research** ; pas approbation de la suffisance, encore moins
passage à Fundamentals. La publication V3.1 reste `NO GO`.

## Validation attendue

CI lint/TypeScript/tests/PG/build sur le commit exact ; tests négatifs
d'absence d'artefacts V2, anciens aliases refusés, CHECKPOINT toujours
non admissible comme Research FINAL. PR en brouillon, ni fusion, ni
déploiement, ni écriture Supabase.
