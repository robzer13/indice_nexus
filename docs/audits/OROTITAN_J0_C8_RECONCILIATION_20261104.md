# OROTITAN — J0/C8 — CONSTAT DE RÉCONCILIATION TECHNIQUE

**Date :** 4 novembre 2026  
**Statut :** `C8_TECHNICAL_REMEDIATION_CONFIRMED — ISSUE_RESOLUTION_PENDING — J0_NOT_ACCEPTED`  
**Doctrine :** FUNCTIONALITY FIRST. Pas de nouvelle architecture.  
**Périmètre :** lecture seule ; Supabase SHADOW `awgsurdyvsyolcgpnygh`, Vercel `orotitan-research-shadow-preview`, dépôt GitHub `robzer13/indice_nexus`. Aucune écriture, migration, déverrouillage, publication ou opération sur PRODUCTION.

## 1. Incident examiné

- **Identifiant :** `issue:j0-c8-human-acceptance-transition-not-implemented-20261031`.
- **Événement d'ouverture :** J0 `event_seq=19`, `BLOCKER_OPENED`, 31 octobre 2026, 19:56:15 UTC.
- **Motif historique :** absence d'une transition persistante d'acceptation humaine autorisée et contrôlée, avec preuve d'audit et refus des acceptations automatiques.
- **État au moment du contrôle :** aucune ligne `BLOCKER_RESOLVED` pour cette référence ; 1 incident J0 ouvert ; aucun acte d'acceptation.

## 2. Constat technique indépendant, sur l'état actuellement installé

1. Les fonctions `orotitan_j0_c8_prepare_review_v1` et `orotitan_j0_c8_accept_review_v1` existent dans le schéma `public` de SHADOW. Elles sont distinctes de `orotitan_j0_c8_read_gate_v1`.
2. La préparation impose une revue persistée avec référence d'audit et SHA-256, preuve C1–C7, calcul d'empreinte des critères et watermark. Elle refuse les preuves insuffisantes et les identifiants de rejeu conflictuels.
3. La décision exige une revue antérieure valide, un approbateur déclaré distinct, une formule de consentement explicite, un `authority_ref` conforme, les verrous global et jalon activés et le respect de l'intégrité/watermark. La décision persistée et le rejeu sont contrôlés.
4. Le journal J0 est append-only (trigger `orotitan_j0_events_append_only`) ; un trigger `orotitan_j0_c8_serialize_event_insert` sérialise les insertions avec le verrou transactionnel du jalon.
5. Les sept classes de preuves C1–C7 disposent d'au moins un événement `EVIDENCE_RECORDED`. C1, C6 et C7 ont chacun au moins une preuve `VERIFIED_DIRECT_REGISTRY_READBACK` avec snapshot présent. Cela atteste la couverture documentaire, **pas le verdict métier individuel** de chaque critère.
6. Les fonctions de préparation et décision ont été testées en PostgreSQL SHADOW par le rapport technique du 31 octobre 2026, y compris un scénario positif synthétique J8 annulé par `ROLLBACK`, l'idempotence et plusieurs contrôles négatifs. Ces tests **ne constituent pas une acceptation réelle J0**.
7. La deuxième Preview Vercel `dpl_EucMmocYuv1MxWWnbxunZmVaGy2o` est `READY`, issue du CLI, sans alias Production. La lecture GET J0 authentifiée a produit `ok=true`, `scope=SHADOW_ONLY`, `activationLocked=true`, `accepted=false`, `stageStarted=false`, `published=false`. Les journaux API SHADOW constatent des requêtes C8 récentes en HTTP 200 après la correction des clés de service.
8. En lecture directe du SQL : `openBlockers=1`, `missingEvidenceIndices=[]`, `journalWatermark=19`, `eligibleForReview=false`, `acceptanceEnabled=false`, `accepted=false`. Les verrous global et J0 valent `false`, les tables de revue/décision J0 sont vides.

## 3. Conclusion d'audit circonscrite

L'énoncé technique originel « transition humaine non implémentée » **n'est plus à jour pour la capacité PostgreSQL**. Le mécanisme est désormais implémenté, et le contrôle de lecture hébergée a été confirmé avec succès. La cause historique est donc **techniquement remédiée**.

**La résolution du blocage dans le Registry n'a pas eu lieu.** Le présent constat n'est ni une approbation humaine d'acceptation, ni une attestation d'identité du réviseur/approbateur, ni une vérification positive du parcours HTTP POST de gouvernance de cette Preview (intentionnellement interdit par le mode lecture seule).

### Réserves avant toute acceptation J0

- Publier un reçu d'audit figé dans le dépôt officiel, sous `docs/audits/` à un commit immuable, avec SHA-256 des octets. Le chemin et l'empreinte sont exigés par la route HTTP de préparation C8.
- Faire valider le contenu métier des critères C1–C7, notamment les anciennes réserves C5 ; la simple présence d'une classe d'événements n'est pas une décision humaine de PASS.
- Disposer d'une revue indépendante et d'un contrôle d'identité/autorité suffisants ; les labels SQL `reviewer_label` et `approver_label` sont déclaratifs.
- Clarifier séparément l'indicateur Chrome « Dangereux » observé sur la Preview avant toute opération sensible depuis le navigateur.
- Obtenir une autorisation distincte pour l'événement append-only `BLOCKER_RESOLVED`, puis une autorisation distincte pour toute préparation/décision d'acceptation. Ne pas activer `OROTITAN_C8_ACCEPTANCE_GO_ENABLED` par défaut.

## 4. Chemin minimal de clôture, non exécuté

1. Valider/publier le reçu de réconciliation technique à une référence GitHub immuable et calculer son SHA-256. La publication GitHub n'est pas réalisée ici.
2. Après approbation explicite, enregistrer **un seul événement** `BLOCKER_RESOLVED` pour la référence historique, par le mécanisme de journal existant `orotitan_j0_record_event_v1`, en y associant le reçu d'audit ; **ne jamais UPDATE/DELETE** l'événement d'ouverture.
3. Relire indépendamment `orotitan_j0_c8_read_gate_v1('J0')` : `openBlockers` doit devenir 0, `missingEvidenceIndices` doit rester `[]`, `accepted` doit rester `false`. Contrôler aussi le nouvel événement dans le journal.
4. Soumettre ensuite le dossier J0 à un réviseur/approbateur habilité selon la gouvernance C8, avec GO d'acceptation explicite et séparé. Aucune de ces opérations n'est autorisée par le présent rapport.

## 5. Traçabilité des sources

- SQL live SHADOW, fonctions : `orotitan_j0_c8_accept_review_v1`, `orotitan_j0_c8_prepare_review_v1`, `orotitan_j0_c8_gate_check_v1`, `orotitan_j0_c8_read_gate_v1`, `orotitan_j0_record_event_v1` ; contraintes et triggers du journal.
- `OROTITAN_C8_SHADOW_REMEDIATION_REPORT_20261031.md` : SHA-256 `1177f8cbc46c2076e5994011a25f91a9e37a0221020e7b1eaee85ee4a24d1d65` (empreinte vérifiée sur l'exemplaire fourni).
- `OROTITAN_C8_SHADOW_HARDENING_V1_1.sql` : SHA-256 `aab9a84fbcf1499b2af3d0adc1826c3bc0453ab2b4096cda1d51caaa6fcbb57b` (empreinte vérifiée sur l'exemplaire fourni).
- `OROTITAN_C8_SHADOW_READONLY_REAUDIT_20261031.sql` : SHA-256 `f7f1a4e80f0d1cc49c7bf4e22f5d277d8f93ea7357bab0f61be50247e6d6e41e` (empreinte vérifiée sur l'exemplaire fourni).
- GitHub PR #438, HEAD `87eece3b359d7ae7d10da8a8b6017f3b6f3a0cb7` au contrôle, `OPEN/DRAFT` et non fusionnée.
- Source du déploiement local fourni par l'utilisateur : commit `df87ba0193ba82b724ab7b5c78694ff8c313091a`, branche `fix/orotitan-c8-preview-isolation-20261102` ; second déploiement CLI contrôlé `dpl_EucMmocYuv1MxWWnbxunZmVaGy2o`.

**Attestation de portée :** vérifications techniques en lecture seule et concordance de plusieurs sources ; pas de signature d'auditeur humain, pas de clôture persistante d'incident, pas d'acceptation J0, pas d'accès PRODUCTION.
