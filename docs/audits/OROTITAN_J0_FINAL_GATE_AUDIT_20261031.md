# OroTitan Run Factory — AUDIT FINAL DU JALON J0

**Référence :** OROTITAN_J0_FINAL_GATE_AUDIT_V1 · 31 octobre 2026  
**Environnement audité :** Supabase **SHADOW** `awgsurdyvsyolcgpnygh`; Vercel Preview SHADOW `dpl_A5txzp53Wfy3hkZ6WjSWdwGb75ps` (READY).  
**Production `cugpgtzygqqlxetyeven` :** hors périmètre. Aucun appel de migration, SQL, déploiement ou modification de production durant cet audit. Il ne s'agit pas d'un audit indépendant de l'état de la production.  
**Autorité de l'opération :** GO AUDIT FINAL J0 — vérifier les 8 critères, compléter les preuves, produire un verdict, **sans acceptation automatique**, ni J1, ni Research.

## Verdict : BLOCKED_FOR_FORMAL_ACCEPTANCE — J0 NON ACCEPTÉ

**7 critères sur 8 disposent de preuves techniques favorables (C1 à C7), avec la réserve C5 ci-dessous.** Le critère C8 **n'est pas exécutable de bout en bout** : le système refuse l'acceptation automatique (contrôle positif), mais aucun chemin autorisé ne permet encore de **persister une acceptation humaine** au terme d'un GO distinct. Le jalon **ne doit pas être déclaré `ACCEPTE`**.

- Position enregistrée J0 : `initial_status=EN_COURS`, champ historique et immuable.
- Position opérationnelle calculée dans le cockpit : **`BLOQUE`**, car un incident C8 est ouvert.
- Jalons acceptés : **0**. Événements `GATE_REVIEW_REQUESTED` : **0**.
- Aucun GO pour J1 / Research / publication n'a été donné dans cet audit.

**Blocage enregistré et relu :** `issue:j0-c8-human-acceptance-transition-not-implemented-20261031` ;
événement `445c4f47-fb26-4445-bc3d-72fa02f798eb`, `event_seq=19`, `BLOCKER_OPENED`.
Il ne sera pas clôturé par la seule production de ce rapport.

## Matrice des 8 critères, selon la nomenclature effective de SHADOW

| Critère | Définition figée | Verdict | Preuve persistée et vérification |
|---|---|---|---|
| **C1** | Etats relus du Registry SHADOW | **PASS** | `20de0d7d-87f3-4bc0-b126-92bb5f8a1733` ; Registry L’Oréal `CREATED`, v2, cutoff 2026-10-30, SHA contractuel exact, 0 stage/artefact/snapshot. |
| **C2** | Registre J0-J8 persistant | **PASS** | Reçu `d49bb260-0684-4768-809b-0ed9da34e779` (seq 16, `REPORTED`) ; SQL indépendant : 9 codes distincts, ordinals 0 à 8, 8 critères par jalon, 0 hors SHADOW, 0 accepté. |
| **C3** | Preuves et décisions immuables | **PASS sur le modèle de contrôle applicatif et DB** | Reçu `1683abb9-0d4d-47ed-bcd0-2805f9b93a5c` (seq 17, `REPORTED`) ; tables RLS, 2 triggers immuabilité actifs ; **7 refus négatifs** réussis dans `BEGIN/ROLLBACK` ; événements d'approbation de la feuille de route et de GO J0 persistés. |
| **C4** | Exceptions traçables | **PASS** | Incident `issue:j0-c4-live-20261031`, ouvert seq 8 et résolu seq 9 ; 6 négatifs historiques et nouvel audit ; preuve `16570f12-a702-4906-a8ec-72ff8414dc6e`. C5 ouvert seq 12, résolu seq 15. |
| **C5** | Contrôles d’accès et négatifs | **PASS AVEC RÉSERVE** | Test HTTP hébergé `5cec3e1c-588a-4ae6-bbae-da3c9632446c` : origine 403, type 400, champs 400, acceptation forgée 400. Reçu vérifié par SQL ; incident C5 clôturé `1fe8d1d4-fe9e-4d77-a44e-936a0b919469`. `anon`/`authenticated` sans accès SQL/RPC, `service_role` sans UPDATE/DELETE ; 401 applicatif vérifié localement mais non observé derrière le SSO Vercel. |
| **C6** | Démonstration interface authentifiée | **PASS** | Test réel par l'opérateur : `6df500a8-fd01-479d-84c5-0a84e77b530d`, `event_seq=7`, `VERIFIED_DIRECT_REGISTRY_READBACK`. L'événement a été relu indépendamment dans SHADOW. |
| **C7** | Relecture indépendante du reçu J0 | **PASS** | Preuve additionnelle `130ae28f-3054-431f-9d37-f33898fb6dfd` (seq 18), `VERIFIED_DIRECT_REGISTRY_READBACK`, avec `verified_snapshot` non nul : run, issuer, security, dossier, cutoff, deux événements, 1 reçu P0-E, 0 stage, 0 artefact ; relecture indépendante du reçu C6 également effectuée. |
| **C8** | Acceptation explicite distincte | **BLOQUÉ** | L'absence d'acceptation automatique est prouvée, notamment par les tests de refus `MILESTONE_ACCEPTED`. Mais la table des jalons bloque toute UPDATE et la RPC J0 refuse les événements d'acceptation ; aucune RPC de décision humaine n'existe dans le périmètre J0. **Aucune décision humaine d'acceptation n'est accordée**, et son enregistrement contrôlé reste à concevoir. |

### Distinction impérative des niveaux de preuve

- `VERIFIED_DIRECT_REGISTRY_READBACK` est produit par la RPC J0 seulement pour une référence `registry:run:<UUID>` ; il inclut une copie des faits lus directement au moment de l'événement.
- `REPORTED` est le statut exact des preuves SQL/HTTP/commit autres que `registry:run`. Les qualifier de preuves automatiquement certifiées serait inexact. Leur consistance a été contrôlée **hors de l'événement** pendant cet audit (SQL direct, tests, reçu et déploiement).
- Les 4 refus HTTP C5 ont été enregistrés par la Preview sous le commit `b1749b75cfa13ea75a0660401d8837f95fc1afd8`, déploiement `dpl_A5txzp53Wfy3hkZ6WjSWdwGb75ps` **READY**.
- La matrice automatisée disponible pour ce correctif a exécuté **737 tests : 737 réussis, 0 échec**, plus TypeScript et lint sans erreur (Vercel Sandbox ; les statuts GitHub externes ne sont pas assimilés à une certification CI).
- La protection SSO n'a pas été désactivée. Les tests HTTPS vérifient les 403/400 **générés par OroTitan derrière SSO** ; ils ne prouvent pas un 401 applicatif sans session dans cette même Preview, car Vercel intercepte cette situation. Un 401 local et les refus de rôles SQL complètent cette couverture, sans la rendre identique.

## État du Registry contrôlé après les nouveaux reçus

| Observation | Valeur vérifiée |
|---|---:|
| Jalons J0–J8 | **9** |
| Événements du journal J0 | **19** |
| Événements rattachés à un critère | **13** |
| Preuves `VERIFIED_DIRECT_REGISTRY_READBACK` | **4** |
| Jalons `ACCEPTE` | **0** |
| Incidents J0 ouverts | **1 : C8 uniquement** |
| Run L’Oréal `204390a1-5cba-4b9c-9821-ca0c7df43eca` | **CREATED**, `state_version=2` |
| Événements Registry de ce run | **2** |
| Reçus de création P0-E | **1** |
| Stages / artefacts / snapshots L’Oréal | **0 / 0 / 0** |
| `published_at` du run | **NULL** |

Les trois preuves complétées par le présent audit portent les numéros `event_seq=16` (C2), `17` (C3) et `18` (C7). L'événement `19` constate le blocage C8. Le journal antérieur n'a pas été modifié.

## Matrice supplémentaire de sécurité, exécutée dans SHADOW

L'audit a exécuté sept opérations interdites dans une transaction **explicitement annulée** (`BEGIN / ROLLBACK`). Résultat : **7 refus sur 7**.

1. UPDATE d'un événement J0 existant → `J0_APPEND_ONLY_HISTORY`.
2. DELETE d'un événement J0 existant → `J0_APPEND_ONLY_HISTORY`.
3. UPDATE de `initial_status` pour fabriquer `ACCEPTE` → `J0_APPEND_ONLY_HISTORY`.
4. DELETE d'un jalon → `J0_APPEND_ONLY_HISTORY`.
5. Rejeu de la même clé idempotente avec payload différent sous `service_role` → `J0_IDEMPOTENCY_CONFLICT`.
6. RPC avec type d'événement `MILESTONE_ACCEPTED` sous `service_role` → `J0_EVENT_INVALID`.
7. Résolution d'un incident inexistant sous `service_role` → `J0_ISSUE_STATE_CONFLICT`.

Contrôle des droits direct : `anon` et `authenticated` n'ont ni SELECT/INSERT/UPDATE/DELETE sur le journal ni exécution de la RPC ; `service_role` n'a pas UPDATE/DELETE sur le journal et ne peut pas mettre à jour les jalons. Les deux tables sont protégées par RLS et triggers. Il s'agit d'**immutabilité applicative et en base sous le schéma actuel**, non d'une garantie cryptographique contre un propriétaire de base pouvant changer le DDL.

## Non-conformité prioritaire : C8

**Sévérité : bloquante pour la certification opérationnelle de J0.**

Le dispositif actuel est volontairement *fail-closed* : il refuse de fabriquer une acceptation. Toutefois, le workflow ne possède pas le mécanisme opposé : **recevoir et persister une autorisation humaine distincte, vérifiée, rejouable sans doublon et auditable**. Une simple réponse « GO ACCEPT J0 » dans un chat ne suffit pas à rendre le jalon `ACCEPTE` dans le Registry actuel.

**Remédiation recommandée, à autoriser séparément :**

1. Concevoir un **registre append-only des décisions de gate** distinct des preuves et du champ `initial_status`. Éviter de modifier les événements historiques.
2. Exiger une authentification administrateur et un acte explicite lié à `J0`, à un **manifeste d'audit/version/hash**, aux **8 critères**, et à l'identité de la décision.
3. Refuser toute acceptation si un incident bloquant est ouvert, si la preuve est manquante ou obsolète, si la version auditée a changé ou si le snapshot du Registry ne correspond plus.
4. Prévoir idempotence, contrôle de concurrence, tests `anon/authenticated`, 401/403, faux GO, relecture indépendante et reçu immutable.
5. Séparer trois décisions : **accepter J0**, **autoriser le démarrage J1**, **autoriser Research**. La première ne déclenche jamais automatiquement les deux suivantes.
6. Rejouer le scénario dans SHADOW, puis demander une nouvelle autorisation humaine pour **l'acceptation réelle**, sans publication ni production.

**Réserve complémentaire de gouvernance :** la ligne J1 a été initialement semée avec `initial_status=EN_COURS` pour refléter du code préexistant, alors qu'aucune autorisation d'exécution de J1 n'a été délivrée. Cet intitulé ne doit **pas** être traité comme un GO ; le futur registre distinguera impérativement avancement technique, statut de gate et autorisation de lancement.

## Décision et frontière

- **Verdict d'audit : 7 critères techniquement satisfaits, 1 critère bloquant.**
- **Décision d'acceptation J0 : NON ACCORDÉE.**
- **J0 calculé : BLOQUÉ ; J0 historiquement enregistré : EN_COURS.**
- **J1 / J2 / Research / Deep Dive / Integration : aucun démarrage autorisé par cet audit.**
- **Production : aucune opération.**
- **Action suivante : GO dédié à la conception/qualification d'un mécanisme C8**, puis audit revalidé avant de solliciter une **approbation J0 distincte**.

Le présent rapport est un avis d'audit argumenté et traçable, **pas** un certificat d'acceptation signé.
