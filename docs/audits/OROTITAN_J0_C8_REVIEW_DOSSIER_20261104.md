# OROTITAN — J0/C8 — DOSSIER TECHNIQUE POUR REVUE HUMAINE

**Date de référence :** 4 novembre 2026  
**Statut du document :** `TECHNICAL_EVIDENCE_READY — HUMAN_REVIEW_NOT_SIGNED — J0_NOT_ACCEPTED`  
**Périmètre :** OroTitan SHADOW `awgsurdyvsyolcgpnygh` exclusivement ; aucune approbation, activation J0/J1, migration, publication Research ou intervention sur PRODUCTION.  
**Nature :** compilation de preuves techniques pour revue indépendante. Ce document n'est **ni** une signature de réviseur humain **ni** une décision d'acceptation.

## 1. Sources figées et chaîne de preuve

- Dépôt : `robzer13/indice_nexus` (public).
- Code candidat isolé, testé et audité localement : `df87ba0193ba82b724ab7b5c78694ff8c313091a` ; parent GitHub `87eece3b359d7ae7d10da8a8b6017f3b6f3a0cb7`. Le candidat n'a pas été poussé sur GitHub ; la référence au SHA local est corroborée par les vérifications de worktree et de fichiers locaux communiquées lors du déploiement, **pas** par un commit GitHub téléchargeable.
- Deuxième déploiement Vercel SHADOW : `dpl_EucMmocYuv1MxWWnbxunZmVaGy2o` ; `READY`, source CLI, alias Production absent ; URL `https://orotitan-research-shadow-preview-hm9dvpb5o-orotitan.vercel.app`.
- Constat de remédiation C8 publié : `https://github.com/robzer13/indice_nexus/blob/2fe4ffc476ce155e406321a094fc3759ac97ba61/docs/audits/OROTITAN_J0_C8_RECONCILIATION_20261104.md`, SHA-256 déclaré et vérifié : `c8ba6c944bdee358bd081a2bb815b7833eb7c5d98cd16f0564e202ae136f9b87`. Ce constat décrivait la situation **avant** la clôture de l'incident ; l'état actualisé figure ci-dessous.
- Rapport de remédiation SQL SHADOW : `OROTITAN_C8_SHADOW_REMEDIATION_REPORT_20261031.md` (fichier fourni) : scénarios PostgreSQL transactionnels dont simulation positive J8 avec `ROLLBACK`, tests négatifs, verrous désactivés après tests.
- Source d'autorité pour l'état et les preuves : tables et fonctions SQL live de `public` sur SHADOW, dont `orotitan_j0_c8_read_gate_v1('J0')` et `orotitan_j0_events_v1`.

## 2. Critères J0 — constat technique indépendant

Les intitulés ci-dessous proviennent de `public.orotitan_j0_milestones_v1.gate_criteria` pour `J0` ; les preuves identifiées proviennent des événements SHADOW. `REPORTED` signifie preuve enregistrée, **pas** validation humaine automatique.

| Critère | Libellé canonique | Événement retenu (séquence / UUID) | Constat technique | Réserve de revue |
|---|---|---|---|---|
| C1 | États relus du Registry SHADOW | #4 / `20de0d7d-87f3-4bc0-b126-92bb5f8a1733` | `VERIFIED_DIRECT_REGISTRY_READBACK`, snapshot de run `CREATED`, version 2 ; concorde avec le live | Revoir la pertinence métier du run et du périmètre |
| C2 | Registre J0–J8 persistant | #16 / `d49bb260-0684-4768-809b-0ed9da34e779` | 9 jalons SHADOW, 8 critères chacun, lecture SQL live | Événement `REPORTED`, réexaminer les tests historiques |
| C3 | Preuves et décisions immuables | #17 / `1683abb9-0d4d-47ed-bcd0-2805f9b93a5c` | RLS activée sur journal et quatre tables C8 ; trigger append-only du journal constaté | Événement `REPORTED`, vérifier le périmètre des tests négatifs |
| C4 | Exceptions traçables | #10 / `16570f12-a702-4906-a8ec-72ff8414dc6e` | Historique J0 : 3 événements `BLOCKER_OPENED`, 3 événements `BLOCKER_RESOLVED` ; 0 blocage courant | Événement `REPORTED` |
| C5 | Contrôles d’accès et négatifs | #14 / `5cec3e1c-588a-4ae6-bbae-da3c9632446c` | Refus HTTP non authentifié, redirection SSO Vercel, RLS et contrôles négatifs antérieurs documentés ; code C8 read-only inspecté | Événement `REPORTED` ; la Preview C8 interdit volontairement les POST : pas de test positif réel de la route de décision |
| C6 | Démonstration interface authentifiée | #7 / `6df500a8-fd01-479d-84c5-0a84e77b530d` | Snapshot `VERIFIED_DIRECT_REGISTRY_READBACK`, run `CREATED`, version 2 ; GET J0 authentifié sur 2e Preview `ok=true` ; HTTP 200 observés dans les journaux API | L'avertissement Chrome « Dangereux » reste non élucidé, reporté par l'opérateur |
| C7 | Relecture indépendante du reçu J0 | #18 / `130ae28f-3054-431f-9d37-f33898fb6dfd` | Snapshot `VERIFIED_DIRECT_REGISTRY_READBACK`, un reçu d'admission P0-E présent, run `CREATED` version 2 | L'événement Registry n'équivaut pas à une signature humaine indépendante de la revue d'acceptation C8 |
| C8 | Acceptation explicite distincte | Aucune décision d'acceptation | RPC de préparation et de décision installées, avec vérification d'intégrité, watermark, idempotence, verrous et reviewer/approver distincts **déclarés** | **NON EFFECTUÉ** ; aucun GO d'acceptation ni identité humaine attestée |

**Concordance du run :** ID `204390a1-5cba-4b9c-9821-ca0c7df43eca`; état live `CREATED`, `state_version=2`, `current_stage=NULL`. Les snapshots C1/C6/C7 portent également `CREATED` et version 2. Un reçu P0-E est présent.

## 3. Contrôle de sécurité et dernier état persistant

L'incident historique C8 a été résolu par une insertion append-only unique :

- Référence : `issue:j0-c8-human-acceptance-transition-not-implemented-20261031`.
- Ouverture conservée : `BLOCKER_OPENED`, séquence #19.
- Résolution persistée : `BLOCKER_RESOLVED`, séquence #37 ; événement `b67237b9-59ca-45fb-baf4-e02d590edd37`, enregistré le 4 novembre 2026 à 08:34:29 UTC ; référence au constat GitHub figé et à son SHA-256.
- Relecture SQL après insertion : `openBlockers=0`, `missingEvidenceIndices=[]`, `journalWatermark=37`, `eligibleForReview=true`, `accepted=false`, `preparedReviews=0`, `acceptanceEnabled=false`, `acceptanceAllowedNow=false`.
- Les verrous SQL global et J0 restent `enabled=false`. Aucune décision n'a été persistée, aucun stage Research lancé, aucun jalon J1 autorisé.
- L'environnement `OROTITAN_C8_READ_ONLY_ENABLED=1` rend la Preview de qualification **non inscriptible** : aucune préparation ou acceptation ne doit être tentée depuis cette URL.

## 4. Contrôles encore requis avant une décision humaine

1. **Revue métier des sept preuves.** Une personne habilitée doit examiner ces éléments, leurs références et leurs réserves, notamment les tests négatifs C5, et prononcer un avis documenté. L'éligibilité calculée par SQL n'est pas un jugement métier.
2. **Identité et indépendance.** Désigner explicitement un réviseur humain et un approbateur humain distincts. Le SQL vérifie des labels différents, mais n'authentifie pas à lui seul deux personnes physiques : l'autorité doit être vérifiée hors des simples champs déclaratifs.
3. **Audit immuable pour `prepare`.** La route HTTP de préparation requiert un document sous `docs/audits/*.md` à un SHA GitHub complet et le SHA-256 exact de ses octets ; la référence doit être lisible par son fetch HTTPS avant utilisation. Le présent dossier peut devenir cette source après publication et contrôle.
4. **Parcours d'écriture contrôlé.** La Preview C8 actuelle est volontairement **read-only** et ne peut pas effectuer le POST de préparation ni d'acceptation. Ne pas désactiver son garde-fou ni réutiliser le gate Research pour débloquer le POST. Toute exécution ultérieure devra être cadrée et autorisée distinctement, sans ouvrir les mutations Research non liées.
5. **GO d'acceptation distinct.** Avant `ACCEPTED`, exiger une revue persistée validée, consentement exact `ACCEPTER J0 SHADOW`, référence d'autorité adéquate, verrous global/J0 activés de manière contrôlée et drapeau runtime de décision explicitement autorisé. Aucune de ces étapes n'est implicitement accordée par l'éligibilité ou le présent dossier.
6. **Avertissement Chrome différé.** L'opérateur a demandé de reporter l'enquête « Dangereux ». Ne pas saisir de secrets ni effectuer d'opération administrative sensible dans cette Preview avant clarification ; utiliser un canal attesté et borné pour les futures étapes.

## 5. Avis technique à soumettre, non signé

**Recommandation technique : `READY_FOR_INDEPENDENT_HUMAN_REVIEW`**, fondée sur le registre SHADOW et les contrôles Vercel/SQL cités. **Ce n'est pas `J0_ACCEPTED`.** L'avis humain sur C1–C7, l'identité de deux acteurs distincts, les actes de préparation et d'acceptation demeurent à réaliser dans l'ordre, avec leurs GO propres.

**Réviseur humain :** NON DÉSIGNÉ.  
**Avis du réviseur :** NON RENDU.  
**Approbateur distinct :** NON DÉSIGNÉ.  
**GO ACCEPT J0 :** NON DONNÉ.  
**Decision ID :** AUCUN.  
**J1 :** NON AUTORISÉ.

Aucune donnée de production, aucun secret et aucune clé API n'est inclus dans ce dossier.
