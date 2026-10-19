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
`DD_INPUT_SUFFICIENCY_RECORD` et `ANALYSIS_INPUT_LOCK`. **Troisième couche
identifiée :** le validateur de manifeste V1 et la fonction SQL **actuellement
installée en production** `public.orotitan_assert_required_outputs(text,jsonb)`
imposent les mêmes anciens noms. Un Research FINAL conforme au contrat V2
serait donc rejeté dans le pipeline ET par le Registry historique.

## Correction isolée, non déployée

Une constante commune reflète les sept noms exacts du contrat V2.
Le vérificateur D2 (artefacts signés, contenu référencé par le FINAL) et
le planificateur D1 partagent maintenant cette source de vérité.
Pour ne pas modifier le validateur V1 épinglé, D2 lui transmet exclusivement
une **copie de validation de schéma** des deux noms V2 sous la désignation
historique V1. Les octets, les références Git, les types persistés, les hashes
et les gates sont exclusivement calculés sur les sept vrais types V2.
Les tests refusent les anciens noms substitués aux vrais artefacts V2.
**Il ne s'agit ni d'un changement du contrat gelé, ni d'une tolérance à
l'absence de preuve.**

**Blocage physique restant :** la fonction SQL de production est toujours
V1-only. Sa mise en conformité nécessite une migration **version-aware**
qui garde inchangé le cas V1 et exige les sept types V2 pour la version
`OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0`. Un script proposé,
**non installé**, est conservé à
`docs/orotitan-equity/proposals/D11_V31_RESEARCH_FINAL_REGISTRY_V2_SQL_PROPOSED.sql`.
Il exige revue SQL négative et autorisation distincte avant toute migration.
Tant que ce point n'est pas résolu, aucune preuve de handoff physique V3.1
n'est revendiquée.

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


## Validation D11 complémentaire : transaction complète en PostgreSQL 17 isolé

Le test `tests/postgres/research-v31-handoff-registry-e2e.sql`, exécuté par
`tests/postgres/run-research-v31-handoff-e2e-tests.sh` dans
`npm run test:postgres`, applique toutes les migrations Registry jusqu'à
V1.12, puis **uniquement dans la base éphémère** la fonction de compatibilité
SQL proposée. Il réutilise les contrôles de bundle-persistence du Registry
actuel (attestation d'un objet staged NON_AUTHORITATIVE, événement
`PERSISTENCE_ATTESTED`, receipt V1.1, vérification des bytes/base64,
hash SHA256, hash Git blob et référencement de l'artefact).

Il exerce les RPC opérationnelles réelles et les **15 pins authentiques
du Contract Set V3.1** : création du run, liaison issuer/security/dossier,
démarrage Research, enregistrement du CHECKPOINT NO, refus de Deep Dive
avant FINAL, refus de FINAL aux types V1 ou à classification manquante,
refus CAS obsolète sans promotion `AUTHORITATIVE`, enregistrement FINAL
avec les 7 vrais outputs Research V2, idempotence de la finalisation,
supersession du CHECKPOINT, puis `start_orotitan_stage(DEEP_DIVE)` admis.

Résultat CI intégralement réussi au commit `5eab339c782cb588ec2ca10823f5b7c1739982d8` :
https://github.com/robzer13/indice_nexus/actions/runs/39308914305

**Portée exacte** : test de transaction réel dans une base locale avec
fixtures et attestations simulées par l'API de persistance du Registry.
Il ne lit pas réellement le dépôt d'artefacts GitHub dans le CI, et il ne
valide pas la suffisance analytique du dossier Besi. Aucune fonction SQL
n'a été installée en production, aucune mutation d'un run réel, aucun
handoff effectif de Besi. La migration reste une autorisation distincte
après revue du SQL et protocole de déploiement/retour arrière.
