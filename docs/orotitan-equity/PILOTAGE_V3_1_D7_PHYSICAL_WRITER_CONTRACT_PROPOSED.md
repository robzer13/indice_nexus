# OroTitan D7 — contrat physique candidat du writer canonique V3.1

**Statut : PROPOSITION NON GELÉE, NON DÉPLOYABLE.** PR #383 de `robzer13/indice_nexus`. Ce document ne modifie ni les contrats FREEZE V1/V2/V3.1 ni les contraintes actuelles en base.

## 1. Périmètre et décision

La publication canonique V3.1 demeure **NO GO**. L'audit D6 a vérifié que les fonctions de persistance canoniques installées en production sont V1 et V2 et que `research_snapshots` n'admet physiquement que les couples `04_SCREENER_SCHEMA_V1/1.0.0` et `04_SCREENER_SCHEMA_V2/2.0.0`. Les RPC `record_orotitan_publish_authorization` et `record_orotitan_publish_result` sont des jalons de pilotage, non des writers validant un payload V3.1.

D7 définit une frontière transactionnelle **candidate** et un banc d'essai négatif. Le SQL du banc `tests/postgres/d7-sandbox-writer.sql` crée uniquement la fonction `d7_sandbox.persist_v31_candidate` sur une base locale jetable ; il ne définit **aucune** fonction `public.persist_orotitan_research_snapshot_v3_1` et ne figure pas dans `migrations/`.

## 2. Contrat requis du futur writer

### 2.1 Entrées

- Dossier UUID, run UUID et référence de l'artefact candidat (UUID/version/SHA-256), observation attendue du pointeur `current_snapshot_id`.
- Payload canonique validé selon les schémas et méthodes V1 (coeur I2), V2 (produit) et V3.1 (overlay), avec `snapshot_id`, `issuer_id`, `security_id`, `report_id`, verrou de dates, versions.
- Référence à une preuve **persistée et gouvernée côté serveur**, obtenue après D2–D6, liant sans équivoque run, contrat épinglé, manifeste FINAL actif, preuves d'entrées I2, dénominateur, date, exact bytes du candidat et snapshot ID ; cette preuve ne peut pas être un booléen transmis par le client ni un rapport `PASS` auto-déclaré.
- Autorisation `GO PUBLISH <COMPANY>` distincte, identité administrateur, CAS `orotitan_runs.state_version`, clé d'idempotence et journal d'événement.

### 2.2 Invariants physiques

1. `SECURITY DEFINER` (uniquement après revue) avec `search_path` épinglé ; permission `EXECUTE` seulement au rôle de service, ni `PUBLIC`, ni `anon`, ni `authenticated`. Aucun accès direct `INSERT/UPDATE/DELETE` au magasin canonique ; RLS et trigger d'immutabilité maintenus.
2. Le dossier est verrouillé dans la transaction par `SELECT ... FOR UPDATE`. Vérifier l'épisode, issuer/security, baseline, pointeur attendu, statut des sorties et de l'autorisation avant écriture. Aucun SELECT applicatif préalable ne remplace cette vérification.
3. Le **même commit SQL** insère un snapshot immuable et déplace `research_dossiers.current_snapshot_id`. Toute erreur, y compris un échec du déplacement du pointeur, rollback l'insertion.
4. `snapshot_id` inexistant + pointeur inchangé = tentative d'insertion admissible ; ID existant + payload, identité, contrat, version et pointeur actuel strictement identiques = retry idempotent ; toute collision divergente = refus `23505`, y compris si le pointeur est resté sur un autre snapshot.
5. CAS différent = refus `40001`, sans mutation. Aucune publication implicite, aucun reclassement d'un run ancien, aucune réécriture de snapshots historiques.
6. `contract_version` / `schema_version` pour V3.1 = **couple à figer séparément**, ici le banc d'essai emploie les sentinelles provisoires `04_SCREENER_SCHEMA_V3_1` / `3.1.0`. Leur présence dans le banc ne les déclare pas autorisées en production.
7. Un EXACT possède un nombre d'actions strictement positif, pas de bornes ni de champ inconnu ; un BOUNDED possède deux bornes strictement positives, ordonnées, `COMPLETE`, aucune classe non bornée et aucun point de remplacement ; un UNKNOWN ou un mouvement non prouvé bloque un snapshot ANALYZE final.
8. Temporalité V3.1 : `valuation_date=time_origin=economic_share_count_date=share_count_as_of_date=per_share_output_date=DATA_CUTOFF` ; `reference_price_date <= DATA_CUTOFF`; `market_cap_date=reference_price_date` pour prix numérique ; pont EV selon FCFF vs FCFE/Owner Earnings. Aucun transfert silencieux du dénominateur d'une date à l'autre.
9. Les règles `I3B_VALIDATED_SNAPSHOT_WRITER_V1.2` de propagations de plages, classification croisant les bornes, sécurité I2 et audit des sources doivent être garanties par le validateur **côté serveur et signé/attesté**, non reconstituées par une fonction SQL à partir des indicateurs textuels.

### 2.3 Séquence transactionnelle requise

`authorization exact user command → proof pack (D2–D6, freeze approved) → transaction (lock run / dossier, re-verify CAS, verify exact manifest & evidence authority) → immutable insert + canonical pointer → receipt → record_orotitan_publish_result`.

Toute étape manquante = NO GO. Les opérations de contrôle avant la transaction sont consultatives ; elles ne produisent jamais à elles seules `I2_PASS` ou `I3B_PASS`.

## 3. Banc d'essai PostgreSQL 17 — hors production

`tests/postgres/run-d7-writer-candidate-tests.sh` :
- refuse tout hôte PostgreSQL non loopback ;
- crée `orotitan_d7_<pid>` puis le détruit ;
- installe seulement dans cette base les fixtures historiques I1/I3/I3-B/V2 et les contraintes sentinelles V3.1 ;
- installe le prototype **dans le schéma privé `d7_sandbox`**, sans fonction publique V3.1 ;
- injecte via superutilisateur une attestation **`D7_TEST_ONLY_APPROVED`** pour simuler l'approbation du dossier et vérifier le comportement transactionnel ; aucun rôle API ne peut créer cette attestation ;
- démontre à la fois un flux de contrôle positif sandbox (pour vérifier que les refus ne sont pas des succès trompeurs) et des refus négatifs (autorisation absente, hash divergent, mauvais issuer/security, dates, dénominateur, CAS stale, collision d'ID, replay historique, suppression de privilèges, rollback).
- contrôle que les writers V1/V2 restent installés avec leurs signatures préexistantes.

**Ce test ne valide pas** l'analytique V3.1, l'autorisation réelle ni le schéma D6 non gelé. L'attestation de test ne doit jamais être migrée en production. Il ne constitue ni migration ni recette de mise en production.

## 4. Conditions de levée du NO GO

Avant tout merge/déploiement du futur writer : approuver formellement un schéma de provenance et ses pins, définir une autorité d'attestation persistée infalsifiable et liée aux octets, revoir la migration V3.1 et le plan de rollback, protéger les RPC et RLS, tester les courses concurrentes réelles et les replays historiques, puis obtenir une commande explicite distincte `GO PUBLISH`.

**D7 = contrat proposé + tests physiques négatifs en base jetable ; pas de writer V3.1 installé.**
