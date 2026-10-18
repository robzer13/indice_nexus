# OroTitan D8 — Revue et décision de gel de la provenance I2 et du writer physique V3.1

**Date de revue : 18 octobre 2026.** Dépôt unique : `robzer13/indice_nexus`, branche `feat/orotitan-pilotage-v31-adapter-20261018`, PR #383 (DRAFT). **Aucune migration, installation SQL, fusion, publication ou modification de données de production.**

## 1. Décision de gel, portée exacte

**GEL TECHNIQUE V1.0 DU DOSSIER DE REVUE : OUI.** Le fichier `docs/orotitan-equity/d8/OROTITAN_D8_REVIEW_BASELINE_V1_0.json` identifie sans ambiguïté neuf artefacts de revue par SHA-1 de **blob Git** (et non par SHA-1 brut du fichier). La CI recalcule les identifiants des octets et refuse toute dérive. Les contrats méthodologiques préexistants demeurent inchangés.

**GEL MÉTHODOLOGIQUE DU SCHÉMA DE PROVENANCE D6 : NON. GEL D'UN WRITER PHYSIQUE V3.1 DÉPLOYABLE : NON.** Le gel d'une référence de revue ne constitue ni une preuve d'origine des jugements fondamentaux ni une autorisation de publication. Cette distinction est normativement conservée par le manifeste : `methodology_freeze_approved=false`, `i2_end_to_end_certified=false`, `i3b_end_to_end_certified=false`, `physical_writer_v31_approved=false`, `production_migration_authorized=false`, `publication_authorized=false`.

## 2. Revue de la provenance des calculs

Le schéma D6 `D6_PROPOSED_0.1.0` prescrit run, cutoff, révision Deep Dive, Contract Set, identité de l'artefact, Research Evidence Ledger, canonical I2 input et liaison **une par valeur** à un artefact d'autorité (type, UUID, version, SHA-256, pointeur JSON, rôle de jugement). Le validateur vérifie la couverture exacte, la lignée cryptographique préalablement contrôlée par D2 et le recalcul du candidat sans arrondi.

Points positifs techniquement acquis : couverture des feuilles I2, unicité, égalité exacte, refus des sources/rôles erronés, lignée Research explicite, tests adversariaux.

**Blocages pour un vrai FREEZE de méthode :**

1. Les formats internes de `ANALYTICAL_BLOCK_OUTPUTS`, `CERTIFICATION_ARTIFACT`, `VALUATION_ARTIFACT` et `OROTITAN_TERMINAL_GATE_ARTIFACT` ne sont pas figés par un même schéma JSON machine-validable garantissant le sens des chemins `canonical`. La simple répétition exacte d'une valeur I2 ne prouve pas son origine analytique.
2. La correspondance `FUNDAMENTAL_JUDGMENT` / `CERTIFICATION_JUDGMENT` / `VALUATION_JUDGMENT` / `TERMINAL_GATE_JUDGMENT` du D6 est une **proposition de routage**, pas une nouvelle autorité méthodologique. Elle nécessite une revue par les propriétaires de chaque étape et une preuve de non-circularité.
3. Le red team, les bornes hard-proof des mouvements d'actions, la corrélation des bornes, les états d'Investment Class croisant plusieurs intervalles et la fiabilité économique ne sont pas certifiés par la seule comparaison exacte des entrées numériques.
4. Les hashes de contenu d'artefacts D2 concernent des **octets immuables bruts**. Le SQL D7 sandbox utilise `digest(p_canonical_payload::text)` sur la représentation JSONB normalisée ; ce hash **n'est pas** la preuve SHA-256 des octets source. Le futur contrat doit définir explicitement l'attestation des octets bruts, l'identifiant du candidat et la projection JSON canonique, et empêcher les conversions non traçables.
5. La convention de conteneur `canonical_contract_input` n'existe pas dans un contrat de format d'artefact amont gelé. Les sorties antérieures ne sont pas automatiquement rétrocompatibles ; une entrée manquante est un refus, jamais une reconstruction implicite.

**Conclusion provenance : `DESIGN_REFERENCE_PINNED`, `METHOD_FREEZE_BLOCKED`, `I2_PASS_NOT_GRANTED`.**

## 3. Revue du writer physique et de la transition historique

L'autorité figée `I3B_VALIDATED_SNAPSHOT_WRITER_V1.2` exige les validations V1+V2+V3.1, l'admission exacte/bornée du dénominateur à `VALUATION_DATE`, la séparation de `REFERENCE_PRICE_DATE`, le rapprochement FCFF/FCFE/Owner Earnings, le recalcul I2, les contraintes de plage et la protection des historiques. D7 documente l'ordre transactionnel mais son code SQL n'est qu'une **simulation de frontière CAS** dans `d7_sandbox` ; les certificats `D7_TEST_ONLY_APPROVED` n'attestent aucune analyse.

**Inspection Supabase en lecture seule le 18/10/2026 :** les fonctions canoniques installées sont `persist_orotitan_research_snapshot` (V1) et `persist_orotitan_research_snapshot_v2`. Les fonctions `record_orotitan_publish_authorization` et `record_orotitan_publish_result` ne sont pas des writers V3.1. Les contraintes sur `research_snapshots` autorisent uniquement `04_SCREENER_SCHEMA_V1/1.0.0` et `04_SCREENER_SCHEMA_V2/2.0.0`; 1 snapshot V1 et 34 V2, aucun V3.1.

**Blocages pour la recevabilité du writer V3.1 :**

1. Définir et faire approuver une autorité persistante d'admission **côté serveur**, avec secret/acteur indépendant, run exact, état Registry, Contract Set figé, manifestes FINAL, octets candidats, preuve de lignée D6 complète, date et qualité analytique. Elle ne doit pas provenir du JSON candidat ou d'un indicateur `PASS`.
2. En une **même transaction SQL**, verrouiller les lignes `orotitan_runs` et `research_dossiers` dans un ordre déterministe, vérifier le CAS de run (`state_version`) et le CAS du pointeur canonique, vérifier l'autorisation idempotente, lier le bon snapshot au run, puis insérer, pointer et enregistrer le résultat atomiquement ou dans un protocole d'outbox transactionnel robuste. La séquence n'existe pas dans le writer D7 de test.
3. Préparer une migration **nouvelle et distincte**, à autoriser explicitement plus tard, conservant toutes les lignes, contraintes, triggers et permissions V1/V2 et ajoutant le couple contract/schema V3.1 validé. Aucun downgrade V3.1 vers V2 ; aucun faux PASS.
4. Exécuter des **tests de concurrence parallèle réels** (deux sessions PostgreSQL simultanées sur le même dossier et sur deux dossiers) ; la matrice D7 réussie vérifie un CAS séquentiellement obsolète et un rollback, pas un conflit intersessions mesuré.
5. Tester l'exactitude du retour d'idempotence lorsque l'autorisation expirée/révoquée, la version du writer modifiée, le pointeur changé, la lignée ou le manifeste remplacé. Vérifier l'absence de double effet et les receipt receipts de publication.
6. Revoir le `SECURITY DEFINER`, l'absence de schémas modifiables dans `search_path`, les permissions effectives (non seulement les GRANT visibles), le rôle de service, les éventuelles politiques RLS et la surface d'exposition de la fonction.

**Conclusion writer : `TRANSACTION_BOUNDARY_REVIEWED`, `PHYSICAL_WRITER_FREEZE_BLOCKED`, `MIGRATION_NOT_AUTHORIZED`.**

## 4. Décisions contrôlées et prérequis à un prochain gel

Le manifeste D8 fixe une **base de revue immuable**. Les modifications ultérieures exigent une nouvelle version du manifeste et une nouvelle revue de tous les pins concernés. Les prérequis pour émettre un véritable `FREEZE` normatif sur le format de provenance : schémas amont officiels avec preuves de rôle, décisions de certification signées, tests indépendants de non-circularité/valeur/temps et explicitation des traitements BOUNDED. Pour le writer : interface de preuve persistée et vérifiable en transaction, tests SQL intersessions, contrat de réversibilité non destructive, attestation administrateur distincte et nouvelle revue formelle.

**Interdits maintenus :** ne pas promouvoir `I2_PASS`, `I3B_PASS`, `CANONICAL_CANDIDATE_ADMITTED`, `READY_TO_PUBLISH` ni `publicationAuthorized` au motif que le D8 est versionné. Ne pas déplacer le pointeur courant, créer un snapshot en production, appliquer de SQL D7, installer de writer public V3.1, fusionner la PR ou publier sans mandat séparé.

## 5. Validation

Le garde-fou de revue `tests/equity-v3-d8-freeze-review.test.ts` contrôle le manifeste, les SHA de blob Git, la non-promotion des autorisations et la non-existence d'un writer V3.1 public dans les fixtures. Les tests PostgreSQL D7 en base locale continuent de vérifier CAS, refus d'attestation, identités, représentations de dénominateur, dates, idempotence, versioning et rollback. Une CI verte est une preuve de qualité du code de revue, **pas** une recette de production.

**Décision finale D8 : GEL DU SOCLE DOCUMENTAIRE DE REVUE = OUI ; GEL NORMATIF DE PROVENANCE = NON ; GEL DU WRITER POUR PRODUCTION = NON ; MIGRATION = NO GO.**
