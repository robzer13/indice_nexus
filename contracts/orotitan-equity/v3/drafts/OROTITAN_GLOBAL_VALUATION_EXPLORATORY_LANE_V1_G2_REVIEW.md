# OROTITAN — Revue méthodologique globale G2 : Valuation Exploratory Preview

**Date :** 2026-11-11  
**Statut :** APPROVED_FOR_DRAFT_IMPLEMENTATION_AND_SHADOW_READ_ONLY_TESTS. **NOT FROZEN / NOT PINNED / NOT CANONICAL / NOT DEPLOYED.**  
**Périmètre :** tous émetteurs ; sources publiques uniquement ; aucune sollicitation des Relations investisseurs.  
**Objet :** revue du projet OROTITAN_VALUATION_ADMISSION_MINIMAL_PATCH_V3_2_PROPOSED, sans changer les contrats FREEZE ni les Runs existants.

## Sources et précédence

| Autorité existante | Preuve Git immuable | Exigence conservée |
|---|---|---|
| Economic Share Count Methodology V1 | blob bfd034c6a650bb2bd0db1ad3cb3d4476aa9ca66e | EXACT puis RIGOROUS_BOUND, sinon UNKNOWN |
| Deep Dive Stage V3 incorporé en V3.1 | blob 0cbf3c8ecc08b6b947152345f7a1532163d993c1 | UNKNOWN bloque tout modèle *canonique* et tout VALUATION_LOCK |
| Deep Dive Stage V3.1 | blob 2cc80f42a37930044465b6a8112b4ba1b9b4d7b2 | valuation date = cutoff ; actionnariat et EV bridge aux dates gouvernées |
| DCF Timing Authority V1 | blob 4316d8e574151882ae25f83c3f7b97f8bad48cd3 | ACT/365F, cashflow timing, FCFF/WACC et cohérence |
| Valuation Date Alignment Authority V1 | blob 14fe1dcfe413475690e945a78131753dda9df419 | publication versus date économique ; interdiction de lookahead |
| Deep Dive Stage V2 | blob ef100df9d2efbd27605e6df801b48caeadd73275 | lignée, Fundamentals Lock, Valuation Lock, Certification |

## Résultats de revue

**G2-01 / APPROUVE POUR PROTOTYPAGE UNIQUEMENT :** ajouter un module de calcul exploratoire séparé de l'admission canonique. Le statut ECONOMIC_SHARE_COUNT = UNKNOWN conserve CANONICAL_VALUATION_ADMISSION = NO.

**G2-02 / APPROUVE :** autoriser un DCF hypothétique de valeur d'entreprise avec FCFF/WACC explicites, calendrier précis et hypothèses de terminale, sans actionnariat ni dette nette.

**G2-03 / APPROUVE SOUS CONDITION :** des scénarios de capitaux propres ou de valeur par action exigent un pont hypothétique étiqueté et des scénarios d'actions positifs explicités, jamais un dénominateur canonique.

**G2-04 / REFUSE :** toute émission de VALUATION_LOCK, READY_FOR_CERTIFICATION, MOS, OVS, OQS, Investment Score, classe, publication, ordre, Stage Manifest et snapshot canoniques depuis ce module.

**G2-05 / REFUSE :** seuil de matérialité arbitraire, substitution d'un stock historique, assimilation des droits de vote suspendus à la trésorerie d'actions, utilisation d'un document publié après cutoff.

**G2-06 / LIMITE :** le contrôle des dates et des chaînes de source par l'outil ne certifie pas que le fichier est réellement public ou que les prévisions sont économiquement valables. Le chargement/attestation amont reste un contrôle séparé.

**G2-07 / CONDITIONS AVANT DEPLOIEMENT :** approbation formelle d'une nouvelle doctrine G2 et pin prospectif ; revue de code et sécurité ; protection d'accès ; isolation des secrets et du réseau SHADOW ; tests de non-mutation ; procédure de déploiement dédiée. Aucun changement implicite de V3.1.

## Implémentation contrôlée proposée

- Module pur : lib/orotitan-equity/v3/valuation-exploratory-preview.ts. Aucune dépendance aux SDK Supabase, Vercel, I2/I3B ou OMS ; aucune API, table ou side effect.
- Tests : tests/equity-v3-valuation-exploratory-preview.test.ts.
- Pas de route Next.js, pas de déploiement, pas d'activation par configuration de production.
- Les contrôles SHADOW de cette mission sont des tests **read-only** de cohérence amont et de non-mutation, pas des tests de déploiement HTTP.
- Les scénarios chiffrés de tests sont **entièrement synthétiques**, jamais une estimation de Schneider Electric.

## Verdict

**CONDITIONAL_GO_DRAFT_PR_AND_READ_ONLY_SHADOW_TESTS.**  
**NO_GO_DEPLOYMENT / NO_GO_CANONICAL_VALUATION.**

La gouvernance G2 reste à approuver explicitement avant activation. Le Run Schneider db00ce77-04eb-492b-a76b-164ecbe9ccfe conserve le cutoff 2026-10-30 et son CHECKPOINT Fundamentals V1 ; aucun Run successeur n'est créé par ce correctif.
