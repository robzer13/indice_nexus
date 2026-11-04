# OROTITAN — C8 — AMENDEMENT DE GOUVERNANCE SOLO (PROPOSITION v0.1)

**Date :** 2026-11-04  
**Décision d'orientation :** option B choisie par le responsable du projet (exploitation individuelle).  
**Statut normatif :** `DRAFT — NOT FROZEN — NOT IMPLEMENTED — NOT ACTIVATED`. L'option de conception est choisie, mais le contrat C8 en vigueur n'est **pas encore amendé**.  
**Doctrine :** `FUNCTIONALITY FIRST` ; reprise du dispositif C8, sans nouvelle architecture ni deuxième autorité.  
**Périmètre :** GitHub `robzer13/indice_nexus`, Supabase SHADOW `awgsurdyvsyolcgpnygh`, Vercel Preview SHADOW. **Jamais de mutation PRODUCTION** `cugpgtzygqqlxetyeven`.

## 1. Pourquoi cet amendement est nécessaire

Le propriétaire exploite OroTitan seul. Le C8 installé exige pourtant deux labels distincts : `orotitan_j0_c8_accept_review_v1` renvoie `J0_C8_REVIEWER_APPROVER_NOT_DISTINCT` pour le même réviseur et approbateur déclarés. Le formulaire HTTP d'acceptation conserve cette logique d'approbation séparée. La pratique consistant à renseigner deux faux labels pour une seule personne est interdite.

L'objet de l'amendement est de **remplacer explicitement, pour le mode SOLO, la séparation de personnes par une séparation d'actes et des contrôles compensatoires vérifiables**. Cela réduit la garantie d'indépendance humaine : le journal doit l'indiquer honnêtement. L'audit automatisé ou par un modèle d'IA peut constituer un contrôle technique supplémentaire, **jamais une seconde signature humaine**.

Ce texte ne change aucun des huit libellés officiels du jalon J0, aucun contrat analytique FREEZE, aucune règle de démarrage de J1, ni les règles d'accès à PRODUCTION.

## 2. Décision d'architecture retenue

Un seul circuit C8, avec **deux modes explicitement tracés** :

- `STANDARD_TWO_PERSON` : dispositif actuel, par défaut, exigeant des personnes distinctes ; doit rester inchangé pour tous les enregistrements existants.
- `SOLO_OWNER` : exception documentée et volontaire, activable **pour J0 exclusivement dans la première livraison**. L'unique personne autorisée est successivement réviseur de son dossier et approbateur de la décision. Son identité déclarée doit être identique dans les deux étapes, sans prétention de revue humaine indépendante.

Pour J1–J8, étendre la règle uniquement après qualification dédiée ; **aucun élargissement automatique** de ce GO J0.

### Conditions cumulatives obligatoires pour SOLO_OWNER

1. **Amendement versionné** : texte arrêté et référencé sous commit GitHub immuable, avec empreinte SHA-256. Ne jamais faire passer une v0.1 non approuvée pour le nouveau contrat FREEZE.
2. **Dossier de preuve** : les sept preuves C1–C7 demeurent contrôlées, avec `missingEvidenceIndices=[]`, `openBlockers=0`, et rapport d'audit GitHub à commit immuable, dont les octets correspondent à l'empreinte annoncée. Le dossier doit déclarer expressément « revue par le propriétaire, sans indépendance humaine ».
3. **Identité et autorité** : la même identité de propriétaire est déclarée dans la revue et dans la décision, et son autorisation est établie par un canal nominatif vérifiable. Une session d'administration à mot de passe partagé et deux libellés arbitraires ne constituent **pas** une preuve forte d'identité. La méthode d'attestation opérateur devra être fixée puis testée avant activation.
4. **Séparation des actes** : `PREPARE_REVIEW` persiste d'abord une revue immuable, avec `governance_mode=SOLO_OWNER`, références de preuves, watermark, empreinte du rapport et empreinte de la revue ; une **seconde action explicite** décide ensuite, sur ce reçu exact.
5. **Consentement distinct** : l'acceptation solo exige une formule explicite et non équivoque, proposée : `ACCEPTER J0 SHADOW EN MODE SOLO`, ainsi qu'une référence GO **réservée au mode SOLO et à J0**. Le simple choix « option B » ne vaut **pas** ce consentement.
6. **Verrous indépendants** : conserver le garde runtime C8, le verrou SQL global, le verrou SQL spécifique J0, l'authentification applicative, le contrôle d'origine, les permissions minimales, les contrôles d'idempotence, de watermark, de preuve et d'intégrité des reçus. Par défaut, tout est désactivé (*fail-closed*).
7. **Aucune cascade** : `J0_ACCEPTED` n'autorise ni J1, ni démarrage Research, ni publication Production. Chaque action future demande un GO propre.

## 3. Changement technique minimal à préparer, non appliqué

**Source d'implémentation** : repartir du candidat C8 audité `df87ba0193ba82b724ab7b5c78694ff8c313091a` (commit local, non présumé disponible à distance), plutôt que réinstaller sur une branche GitHub plus ancienne. La PR #438 est conservée en l'état.

| Composant | Modification ciblée | Exigence de compatibilité |
|---|---|---|
| Tables `orotitan_j0_c8_reviews_v1` / `orotitan_j0_c8_decisions_v1` | Persister un `governance_mode` explicite, et une référence immuable à l'autorité propriétaire / l'amendement. Réutiliser les tables existantes : **pas de nouvelle table**. | Défaut `STANDARD_TWO_PERSON` pour les historiques ; append-only conservé. |
| RPC `orotitan_j0_c8_prepare_review_v1` | En mode SOLO seulement, inclure le mode et la preuve de gouvernance dans le calcul du reçu et de son SHA-256. Ne pas permettre de changer de mode après création. | Les anciennes revues restent vérifiables ; migration additive et testée en PostgreSQL jetable. |
| RPC `orotitan_j0_c8_accept_review_v1` | Maintenir le refus `same reviewer/approver` en mode standard. En mode SOLO : exiger la **même identité de propriétaire**, le même `governance_mode` scellé, le GO J0 SOLO, la formule de consentement dédiée et la vérification des critères/watermark/hashes/verrous. | Le mode SOLO ne doit jamais pouvoir être activé par un alias déclaratif de personne ni par une simple étiquette soumise par le client. |
| Route HTTP `/admin/research-factory/control-tower/c8-gate/accept` | Reconnaître le mode scellé par le serveur ; refuser les mélanges de mode, les GO d'un autre jalon, l'absence de session, le mauvais `Origin` et toute acceptation sans flag runtime. | Le mode `OROTITAN_C8_READ_ONLY_ENABLED=1` de la Preview actuelle reste interdit en écriture. |
| Tests et CI | Ajouter un scénario nominal SOLO en base **jetable / transaction annulée** et des tests de refus, puis une revue de diff indépendante. | **Aucune** migration ou acceptation live pendant le développement. |

Cette table décrit un **cahier des charges**, pas un patch SQL ou TypeScript déjà écrit. Le détail final de la liaison d'identité dépend de la vérification du mécanisme d'authentification disponible. Si aucune attestation suffisamment solide n'est réalisable sans disproportion, l'acceptation demeure `BLOCKED` et ce risque doit être soumis explicitement à l'opérateur.

## 4. Tests de non-régression obligatoires

- **Standard** : approbateur = réviseur → REFUS ; deux identités distinctes légitimes → ancien parcours préservé.
- **Solo sans amendement effectif / sans identité attestée / sans référence d'autorité** → REFUS.
- **Solo avec mode falsifié, étiquette inventée, revue STANDARD réutilisée ou consentement standard** → REFUS.
- **Sans trois verrous/GO requis**, avec mauvaise référence, mauvais jalon, mauvais hash, watermark périmé, incident ouvert ou preuve absente → REFUS.
- **Sans session administrateur**, mauvaise origine, autre projet Supabase, clé client publique, ou tentative de réutilisation hors SHADOW → REFUS.
- **Même clé idempotente** → `REPLAYED` avec même identifiant ; même clé avec autre contenu → REFUS.
- **Mode SOLO nominal** → reçu d'acceptation SHADOW persistant et relu **dans un test isolé**, avec `stageStarted=false`, `published=false` ; aucune mutation réelle de J0 dans cette phase.
- **Postflight SHADOW** : tant que GO ACCEPT J0 SOLO distinct n'est pas donné, 0 décision nouvelle, verrous SQL `false`, Preview actuelle inchangée, aucun lancement J1.

## 5. Séquence d'exécution et conditions d'arrêt

1. **Cadrage terminé** : cette v0.1 exprime l'option choisie ; prendre connaissance des contrats FREEZE et constater les éventuels conflits. Aucun conflit contractuel ne peut être contourné silencieusement.
2. **Amendement formel** : examiner, corriger et approuver le texte en tant qu'avenant de gouvernance ; enregistrer les preuves de cette décision séparément du GO d'acceptation J0.
3. **Implémentation locale uniquement** : changer les composants ciblés à partir du vrai commit candidat, migration additive isolée et tests PostgreSQL / application / sécurité ; arrêter au moindre échec.
4. **Audit du nouveau code** : revue indépendante du diff, des invariants et des tests ; vérifier qu'aucune protection Preview/SHADOW ou production ne régresse.
5. **Qualification SHADOW sous autorisation séparée** : migration et Preview isolées, contrôle négatif puis éventuel circuit d'écriture de préparation sous GO limité. Ni déploiement ni mutation sans nouvelle autorisation explicite.
6. **Revue persistée** : le propriétaire examine C1–C7, atteste les réserves et produit un reçu scellé. La revue n'est pas l'acceptation.
7. **GO ACCEPT J0 SOLO distinct** : après relecture du reçu exact et contrôle de l'identité, confirmation humaine explicite puis activation temporaire **bornée à J0** ; inscrire et relire le reçu `ACCEPTED`, vérifier `stageStarted=false` et `published=false`, puis refermer les verrous.

**STOP** si : état SHADOW divergent, tests manquants, preuve insuffisante, source Git locale non reproductible, identité non attestable, projet Vercel ambigu, avertissement navigateur réapparu, accès Production, ou tentative d'acceptation automatique.

## 6. État de référence à la rédaction (lecture antérieure)

- `J0` : `openBlockers=0`, `eligibleForReview=true`, `missingEvidenceIndices=[]`, `journalWatermark=37`.
- `J0` : `accepted=false`, 0 revue persistée, 0 décision, verrous global / jalon `false`.
- Dossier : `docs/audits/OROTITAN_J0_C8_REVIEW_DOSSIER_20261104.md`, commit GitHub `2021d5b2ec0f5ca48ea18a88a86e937f8d17a721`.
- Candidat applicatif audité : `df87ba0193ba82b724ab7b5c78694ff8c313091a`, non présumé publié sur GitHub.
- Preview de lecture seule : `dpl_EucMmocYuv1MxWWnbxunZmVaGy2o`. **Ne pas réutiliser comme endpoint d'acceptation.**

**Conclusion :** `SOLO_OWNER_PATH_SELECTED — AMENDMENT_DRAFT — NOT_CODED — NOT_ACCEPTED`. Cet avenant ne modifie pas les contrats supérieurs tant qu'il n'a pas été approuvé et implémenté sous contrôle.
