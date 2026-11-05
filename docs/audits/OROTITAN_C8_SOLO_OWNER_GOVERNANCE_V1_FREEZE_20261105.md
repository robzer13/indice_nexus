# OROTITAN — C8 — AVENANT DE GOUVERNANCE `SOLO_OWNER` — V1.0

**Date de décision :** 2026-11-05  
**Statut documentaire :** `RATIFIED — GOVERNANCE FREEZE V1.0`  
**Statut opérationnel :** `NOT DEPLOYED — NOT ACTIVATED — J0 NOT ACCEPTED`  
**Doctrine :** `FUNCTIONALITY FIRST`  
**Périmètre exclusif :** J0/C8 du projet OroTitan Equity Research sur Supabase **SHADOW** `awgsurdyvsyolcgpnygh` et sa Preview Vercel isolée. **PRODUCTION** `cugpgtzygqqlxetyeven` est strictement hors périmètre.

## 1. Décision de ratification et traçabilité

Le responsable du projet a explicitement choisi l'option B, puis a déclaré dans la conversation du **5 novembre 2026** : **« Ok je le ratifie »**, à la suite de la proposition de ratifier la gouvernance solo J0. Cette déclaration vaut **accord de gouvernance exprimé par l'opérateur**, et non signature électronique qualifiée ni vérification d'identité civile. Son association à un compte propriétaire authentifié devra être établie avant toute opération réelle d'acceptation.

La présente version **ratifie les clauses de gouvernance de la proposition v0.1 du 4 novembre 2026**, sans modifier rétrospectivement les octets de cette proposition historique :

- Proposition de référence : `docs/audits/OROTITAN_C8_SOLO_GOVERNANCE_AMENDMENT_PROPOSAL_V0_1_20261104.md`.
- Commit documentaire de référence : `cd7c6ff3908df90b786d08b5239c36c54a8213f2`.
- SHA-256 de la proposition de référence : `b4bcf8f31520ca5a66f05e8bd41e7232be2846d00e300e520cf2815a538e82e9`.

**Portée normative :** cet avenant constitue une exception locale explicite à la règle de séparation de personnes du circuit C8, **pour le mode SOLO_OWNER de J0 seulement**. Il ne modifie aucun contrat méthodologique ou analytique FREEZE, les critères C1–C8, le mécanisme STANDARD existant, ni les règles d'autorisation de J1–J8.

## 2. Politique de gouvernance ratifiée

### 2.1 Deux modes, sans confusion

- `STANDARD_TWO_PERSON` reste le mode **par défaut** et conserve l'exigence de deux personnes déclarées distinctes. Il doit rester compatible avec les enregistrements historiques.
- `SOLO_OWNER` est autorisé comme **exception pour J0 SHADOW exclusivement**. Une seule personne peut réaliser deux actes distincts, successifs et traçables : préparation/revue puis décision d'acceptation. Ces actes ne constituent **pas** une revue humaine indépendante.
- Aucune extension automatique à J1–J8 et aucune réutilisation du GO J0 pour un autre jalon.

### 2.2 Identité propriétaire et contrôles compensatoires

L'acceptation SOLO requiert cumulativement :

1. La session administrateur existante **et** une session Supabase Auth **SHADOW** attestée côté serveur par une vérification effective de l'utilisateur, avec e-mail confirmé, compte non anonyme et correspondance exacte avec l'UUID du propriétaire inscrit sur une liste d'autorisation serveur. Le navigateur ne peut pas désigner librement l'identité autorisée ; les labels de texte ne constituent pas une attestation.
2. Une référence GitHub immuable et une empreinte SHA-256 pour **cet avenant ratifié**, ainsi que pour le dossier d'audit de J0.
3. Une préparation de revue enregistrée de manière immuable, liée à l'identité attestée, au mode SOLO, aux sept preuves C1–C7, au SHA du dossier, aux critères et au watermark du journal. La revue doit indiquer explicitement qu'elle est **réalisée par le propriétaire, sans indépendance humaine**.
4. Une seconde action d'acceptation, distincte, visant le reçu exact de revue et nécessitant la formule **`ACCEPTER J0 SHADOW EN MODE SOLO`**, accompagnée d'une autorisation `GO_ACCEPT_J0_SOLO_*` spécifique, délivrée séparément.
5. Les protections historiques C8 : contrôle d'origine, validation stricte des entrées, fail-closed, contrôle des incidents ouverts, intégrité des preuves et empreintes, refus des revues périmées, anti-rejeu, reçu d'acceptation persistant, relecture indépendante, permissions minimales et garde des frontières SHADOW/PRODUCTION.
6. Les gardes séparées du circuit d'acceptation : autorisation du mode SOLO selon l'avenant ratifié, flag runtime d'acceptation, verrou SQL global et verrou spécifique J0. **Toutes restent désactivées par défaut**.
7. Un refus strict de tout croisement entre `STANDARD_TWO_PERSON` et une autorisation `GO_ACCEPT_J0_SOLO_*`, également au niveau SQL. Les valeurs `NULL` ne doivent pas contourner les validations d'identité ni d'empreinte.

L'e-mail confirmé atteste la maîtrise d'un compte, **pas** l'identité civile, une signature qualifiée ou une seconde revue humaine. Cette limite est explicitement acceptée **pour la gouvernance J0 SHADOW**, sous réserve des contrôles ci-dessus.

### 2.3 Absence d'effet automatique

`RATIFICATION ≠ CODE DEPLOYED ≠ SHADOW ACTIVATED ≠ REVIEW PREPARED ≠ J0 ACCEPTED ≠ J1 GO`.

La présente ratification **n'autorise pas** : installation/migration distante, déploiement Vercel, création du compte propriétaire, activation de verrous, préparation d'une revue persistée, acceptation J0, écriture sur PRODUCTION, ouverture de J1, lancement Research ou publication. Chacune de ces actions exige son autorisation opérationnelle distincte et les prérequis applicables.

## 3. Références techniques et qualification préalable

- Candidat local, branche `fix/orotitan-c8-solo-owner-20261105`, base auditée `df87ba0193ba82b724ab7b5c78694ff8c313091a` ; modifications SOLO locales non encore publiées sur la branche applicative.
- Audit indépendant CODEX 03B : deux défauts précédemment constatés (séparation de modes et nullabilité) corrigés ; une condition de non-nullité SHA et son test ont été ajoutés localement avant qualification PostgreSQL 17.
- Test local fourni par l'opérateur le **5 novembre 2026**, Docker PostgreSQL `170011` : `ALL_FIVE_PG17_MATRICES_PASS`, couvrant Registry V1, Shared Research Inbox, Recovery SHADOW, C8 STANDARD et C8 SOLO. Cette preuve est un **journal d'exécution locale**, pas une preuve de déploiement SHADOW.
- Rapport J0 antérieur : `docs/audits/OROTITAN_J0_C8_REVIEW_DOSSIER_20261104.md` au commit `2021d5b2ec0f5ca48ea18a88a86e937f8d17a721`, SHA-256 `676b447124343dbbb24aa9c62a13ff31beb82ecbe523eaa569b899ba42eaa4ce`. La revue réelle devra porter explicitement la réserve d'absence d'indépendance humaine.

Les sorties de tests et les audits locaux doivent être recontrôlés si la version effectivement déployée diverge des fichiers qualifiés ; aucun état serveur actuel n'est présumé.

## 4. Séquence opérationnelle autorisable ultérieurement

1. Versionner et figer le **candidat de code exact testé**, et confirmer la cohérence du diff et de ses empreintes, sans intégration implicite à la branche principale.
2. Sous GO distinct, installer **uniquement** la migration additive C8 SOLO sur Supabase SHADOW et configurer une Preview isolée ; confirmer que les verrous d'acceptation et les autres écritures restent fermés. Ne jamais agir sur PRODUCTION.
3. Sous GO distinct, configurer le compte propriétaire Supabase Auth SHADOW, tester son parcours nominatif et sa allowlist, puis effectuer les lectures négatives/positives requises.
4. Préparer sous autorisation la revue humaine **par le propriétaire** et lire le reçu immuable, sans accepter le jalon.
5. Après nouveau contrôle du journal, des critères et du reçu, exiger une autorisation **GO ACCEPT J0 SOLO** distincte et la confirmation exacte `ACCEPTER J0 SHADOW EN MODE SOLO` ; ne lever les verrous que de façon bornée, puis vérifier le reçu `ACCEPTED` et les refermer.
6. Vérifier `stageStarted=false`, `published=false`, absence de changement PRODUCTION et **aucun GO J1**.

En cas de divergence, absence d'identité attestée, incohérence d'empreinte, défaut de sécurité ou preuve incomplète : `BLOCKED`, sans acceptation.

## 5. Enregistrement et authenticité

Cette pièce matérialise la **déclaration de ratification** reçue dans la conversation, avec les limitations précisées ci-dessus. Le lien GitHub figé et le SHA-256 des octets de cette version servent de référence documentaire. **Ce document n'est ni une preuve d'identité nominative authentifiée ni une preuve de décision enregistrée dans Supabase.**

**État à l'issue de cette ratification :** `SOLO_OWNER_GOVERNANCE_RATIFIED — SHADOW_NOT_ACTIVATED — J0_NOT_ACCEPTED`.
