# J1 — PRÉ-JALON FUNCTIONALITY FIRST — PLAN DE CONCEPTION v0.1

**Date :** 2026-11-01. **Environnement :** OroTitan SHADOW. **Statut : DRAFT_ONLY / NO-GO BUILD**.  
**Nature :** préparation J1, pas exécution, acceptation, Research ou publication. **Auteur/relecteur indépendant :** non encore attestés. **GO BUILD J1 :** absent.

## 1. Fonctionnalité livrable, avant toute architecture

**Objectif J1 : « Run Factory P0 de bout en bout ».** Démontrer par le vrai chemin HTTP → validations → RPC persistante → relecture indépendante qu'une création de run est déterministe, sécurisée et récupérable. Le premier succès est **L'Oréal REPLAYED sans doublon**, puis un second émetteur CREATED avec le GO approprié. Aucune entrée en stage Research, aucun artefact analytique, aucun snapshot.

**État réel vérifié en lecture seule dans Supabase SHADOW :** J0 non accepté (0 décision, 1 incident C8 ouvert) ; J1 historique `EN_COURS` mais 0 preuve C1–C7, 0 acceptation ; run L'Oréal `204390a1-5cba-4b9c-9821-ca0c7df43eca` : `CREATED`, `state_version=2`, `current_stage=NULL`, 0 stage. Cette observation n'autorise aucune nouvelle écriture ni création du second émetteur.

## 2. Huit critères de sortie J1 — texte exact du registre

1. L’Oréal HTTP REPLAYED
2. Second émetteur CREATED
3. 401/403 et RLS
4. Concurrence et doublons
5. Reprise d’erreur
6. Aucun stage ou snapshot
7. Lecture Registry indépendante
8. Acceptation humaine

Chaque critère requiert une preuve horodatée et reproductible ; **C8 est une décision humaine formelle distincte**, pas une case cochée.

## 3. Recherche / réemploi vérifiés et inconnues

- **Réutilisables sur la branche** : cockpit authentifié `app/admin/research-factory/control-tower`, `run-preflight`, RPC du registre SHADOW, contrôle C8 append-only, et P0-E L'Oréal déjà enregistré.
- **À inspecter dans le code et dans un Preview autorisé avant BUILD** : route HTTP exacte de création P0-E, signature de RPC, stratégie d'idempotency key, comportements de concurrence et d'erreurs, schémas/RLS ; disponibilité d'un second émetteur canonique ; preuves de tests HTTP réels.
- **Dette connue découverte en revue C8** : route `c8-gate/accept/route.ts` vérifie encore un GO HTTP de forme `GO_ACCEPT_J0_…` ; le support de J1–J8 nécessite une correction séparée et testée **avant l'acceptation future de J1**. Ne pas contourner ce problème par un GO fictif.
- **Documentation technique officielle consultée** : [Supabase RLS](https://supabase.com/docs/guides/database/postgres/row-level-security) et [Securing the Data API](https://supabase.com/docs/guides/api/securing-your-api) : GRANT + RLS + tests négatifs, pas de privilège supplémentaire sans justification.

## 4. Comparaison et choix

| Approche | Fonctionnalité / sûreté | Coût de complexité | Décision |
| --- | --- | --- | --- |
| **A. Réemployer les routes/RPC/registre P0 existants**, les compléter uniquement là où un critère échoue | Source d'autorité unique, replays comparables, tests du chemin déjà installé | Faible si le contrat existant suffit | **Préférence**, à confirmer par inventaire HTTP/RPC |
| B. Ajouter une nouvelle couche API générique avant P0 | Possible, mais crée un second point d'orchestration et plus d'états d'erreur | Moyen | Rejet par défaut |
| C. Créer worker/queue et nouvelle machine à états J1 | Peut servir un futur volume, non nécessaire à la preuve J1 | Élevé | Rejet tant qu'aucun besoin mesuré ne l'exige |

**Critère de décision :** l'approche A gagne seulement si elle passe les 8 preuves ; sinon correction locale minimale ou retour à la conception. Pas de nouvelle table, worker, agent ou dépendance par anticipation.

## 5. Plan de livraison minimal — après autorisations ultérieures

1. **Préflight en lecture seule** : figer contrats, routes, endpoints, version du schéma, sources P0-E ; nommer les preuves manquantes et les risques. **Stop** si dérive de contrat ou mauvais environnement.
2. **Tranche verticale L'Oréal** : exécuter le replay HTTP réel contre SHADOW avec la clé initiale, vérifier même `run_id`, aucun nouveau run/stage. **Stop** si état divergent.
3. **Second émetteur** : après autorisation J1 + GO de création de run applicable, réaliser CREATED puis replay sans doublon, avec relecture exacte des UUID/versions. **Stop** si identité canonique non résolue.
4. **Fiabilisation ciblée** : 401/403, GRANT/RLS, requêtes concurrentes et idempotence, erreur transitoire/retry ; preuve que stage/snapshot/publication restent à zéro. **Stop** si un cas négatif échoue.
5. **Audit C1–C7, correction C8 côté HTTP, relecture indépendante et acceptation humaine distincte** ; aucune transition automatique vers J2.

**Rétroaction/rollback :** les tests peuvent réutiliser les runs existants en lecture/replay sans changements ; tout test nécessitant une nouvelle création sera autorisé et tracé séparément. Les données append-only ne sont pas supprimées pour faire disparaître un échec.

## 6. Stop-loss / exclusions

Arrêt immédiat si : plus d'une source de vérité pour l'identité du run, modification silencieuse des contrats FREEZE, frontière SHADOW/production incertaine, duplication, accès non autorisé, preuve non reproductible, ou nécessité de multiplier les correctifs sans pouvoir expliciter le parcours minimal. Refaire le plan plutôt qu'empiler. Exclure : Research, Deep Dive, publication, arbitrage économique, création de services spéculatifs, acceptation automatique des jalons.

## 7. Décision de pré-jalon

**DRAFT_ONLY :** J0 C8 toujours non accepté et incident ouvert ; vérification indépendante du plan et identité du reviewer absentes ; aucun `GO BUILD J1 SHADOW` vérifié.  
**Action autorisée aujourd'hui :** revue critique et préparation documentaire de ce plan. **Action interdite :** exécuter J1 ou accepter J0. Une fois le précédent jalon accepté, présenter le plan ancré à un commit/SHA-256 pour validation humaine indépendante et, seulement après, un GO BUILD spécifique.

*Ce document est un plan de construction, subordonné aux contrats OroTitan FREEZE. Il ne modifie pas les critères J1 ni les états analytiques.*
