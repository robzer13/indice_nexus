# ASML — G001 / G006 — revue de preuves pré-admission v0.1

**Dossier :** OroTitan Equity Research / Deep Dive ASML / observation read-only
**Cutoff figé :** 2026-09-21, clôture XAMS
**Recherches réalisées :** 2026-10-30
**Classification :** RESEARCH_ONLY · NON_ADMITTED · NOT_CERTIFIED · NO_REGISTRY_WRITE
**Autorité :** aucune ; pièce de travail, soumise à Evidence Ledger et Pilotage. Ne modifie pas les contrats FREEZE.

## 0. Portée et décision

Cette revue ne constitue ni une preuve scellée, ni une admission Registry, ni la levée de `ASML-G001`/`ASML-G006`.
Les URL pointent vers des pages externes ; aucun hash de leurs octets n'est attesté par cette revue.
Distinction obligatoire : `SOURCE_DATE` versus `AS_OF_DATE` / `DATA_CUTOFF` versus `RETRIEVED_AT`.

## 1. Identité du titre

- ASML Holding N.V. — ordinaire ; ISIN **NL0010273215**.
- Venue visée : **Euronext Amsterdam (XAMS)** ; devise **EUR** ; unité EUR par action.
- Sources : [fiche officielle ASML](https://www.asml.com/en/investors/shares) ;
  [venue Euronext XAMS](https://live.euronext.com/en/product/equities/nl0010273215-xams).
- **Ne pas confondre** avec Euronext Global Equity Market (BGEM) ni NASDAQ USD.

## 2. G001 — point de départ certain et limite

ASML AGM 2026, record date **2026-03-25** :
- issued shares **388,147,674** ;
- treasury shares **3,224,029** ;
- outstanding shares **384,923,645** ;
- invariant exact : 388,147,674 − 3,224,029 = 384,923,645.
Source primaire : https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026

SEC Form 6-K déposé **2026-07-15**, EU-IFRS Statutory Interim Report for six months ended **2026-06-28**, Exhibit 99.4 : 
- number of **issued and outstanding** shares : **384.1 million** à 2026-06-28, **arrondi** à 0.1m ;
- H1 2026 treasury purchases : **1.7 million** (arrondi) ;
- H1 2026 issuance : **0.4 million** (arrondi) ;
- diluted weighted-average shares : **385.3 million** (H1 EPS), à NE PAS substituer au dénominateur instantané.
Sources officielles disponibles avant cutoff :
https://www.sec.gov/Archives/edgar/data/937966/000162828026048235/0001628280-26-048235-index.htm
https://www.sec.gov/Archives/edgar/data/937966/000162828026048235/statutoryinterimreport20.htm

Les 384.1m ne sont pas un nombre de titres exact ; le 2026-09-21 est encore NON_RECONCILED.

### 2.1. Rachats communiqués entre le 29 juin et le 18 septembre 2026

Ces annonces ont toutes été **publiées au plus tard le 21 septembre 2026**.
Les comptes sont la somme arithmétique des lignes de rachats quotidiens des communiqués hebdomadaires de l'émetteur :

| Annonce (source date) | Transactions (trade date) | Titres rachetés | Pièce primaire |
|---|---|---:|---|
| 2026-07-06 | 2026-06-29 → 2026-07-03 | 48,397 | [ASML](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-173) |
| 2026-07-13 | 2026-07-06 → 2026-07-10 | 50,763 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-174) |
| 2026-07-20 | 2026-07-13 → 2026-07-17 | 121,302 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-175) |
| 2026-07-27 | 2026-07-20 → 2026-07-24 | 249,158 | [ASML](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-176) |
| 2026-08-03 | 2026-07-27 → 2026-07-31 | 273,335 | [ASML](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-171) |
| 2026-08-10 | 2026-08-03 → 2026-08-07 | 266,460 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-178) |
| 2026-08-17 | 2026-08-10 → 2026-08-14 | 248,818 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-179) |
| 2026-08-24 | 2026-08-17 → 2026-08-21 | 252,684 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-180) |
| 2026-08-31 | 2026-08-24 → 2026-08-28 | 254,235 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-181) |
| 2026-09-07 | 2026-08-31 → 2026-09-04 | 314,178 | [ASML](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program) |
| 2026-09-14 | 2026-09-07 → 2026-09-11 | 223,178 | [ASML](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-0) |
| 2026-09-21 | 2026-09-14 → 2026-09-18 | 304,500 | [ASML](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-1) |
| **TOTAL** | **2026-06-29 → 2026-09-18** | **2,607,008** | **12 communiqués** |

**IMPORTANT :** les 2,607,008 actions représentent uniquement le flux brut des rachats déclarés sur cet intervalle.
Ce n'est PAS la diminution nette certifiée de titres économiques.
Manquent la position de départ exacte au 2026-06-28, les éventuelles distributions d'autodétention,
émissions, annulations, la réconciliation trade-date versus settlement-date,
et la couverture exhaustive jusqu'à la clôture du 2026-09-21.

La communication du **2026-09-28** rapporte **25,500 rachats le 2026-09-21**, mais fut
**publiée après cutoff**. Elle peut éclairer un run nouveau/refresh, NON le run au cutoff figé sans
autorisation expresse :
https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-2

### 2.2. Signal de conflit sur la page ASML 2025 (ne pas interpoler)

La page générale `https://www.asml.com/en/investors/shares` affiche sous
«Issued and outstanding per 31 December 2025» les lignes :
`issued 385,417,665`, `treasury 2,730,009`, `outstanding 388,147,674`.
Elles ne satisfont PAS `issued - treasury = outstanding`.
Sa note des grands actionnaires identifie cependant les **385,417,665**
comme titres hors autodétention. Le bilan doit privilégier le 6-K officiel et
les données du record date AGM 2026 en attendant la clarification de l'émetteur.
**Ne pas corriger silencieusement** ces champs dans la source : enregistrer un conflit.

## 3. G006 — prix de référence, venue et statut

Des archives de fournisseurs tiers attribuent au **2026-09-21** :
- `ASML.AS` / Amsterdam, prix de clôture affiché **EUR 1,481.40** ;
- open EUR 1,482.60 ; high EUR 1,496.60 ; low EUR 1,462.80 ; volume 450,484.
- Sources de **corroboration non autoritatives** (pas des pièces Euronext XAMS) :
  - https://finance.yahoo.com/quote/ASML.AS/history/
  - https://stockanalysis.com/quote/ams/ASML/history/
  - https://www.abcbourse.com/cotation/NL0010273215n

L'historique officiel Euronext **BGEM** publie pour le même ISIN et jour
`EUR 1,482.00`, mais **MIC/venue BGEM ≠ XAMS** : interdire cette substitution.
https://live.euronext.com/en/product/equities/nl0010273215-bgem

La fiche **officielle XAMS** confirme l'identité du marché, mais la ligne
du 2026-09-21 n'a pas été retrouvée directement dans la réponse consultable :
https://live.euronext.com/en/product/equities/nl0010273215-xams

**Conclusion : CANDIDATE_EUR_1481_40 / G006 BLOCKED**.
Obtenir la ligne d'archive/End-of-Day Euronext sur le MIC XAMS, comparer `valuation close` /
`unadjusted close`, noter corporate actions et préserver des octets vérifiables.
Le « Close » Yahoo est décrit comme *split-adjusted* ; cela ne certifie pas
le `UNADJUSTED_CLOSE` exigé par le contrat.

## 4. Gates de contrôle et prochaines pièces obligatoires

- `G001 = BLOCKED` : exact `issuedAtOpening`, `treasuryAtOpening`, un ledger de TOUS
  mouvements pertinents jusqu'au cutoff, statuts de règlement, preuve de couverture exhaustive
  et preuve d'émetteur datée <= 2026-09-21. Le 6-K contient des nombres ARRONDIS.
- `G006 = BLOCKED` : source historique **officielle Euronext XAMS**, preuve de
  l'absence d'ajustement, EUR/action, ISIN, date exacte et actions sur titre.
- `DD-PRIMARY-SOURCES` : un PDF Q2 2026 de 832,285 octets a été capturé dans
  Supabase SHADOW ; aucun document de cette note n'est admis/archivé par ce simple commit.
- `DD-ROIC-OE` : non abordé ; non substituable.
- Pas de `MARKET_CAP` certifiée tant que les deux gates ne sont pas satisfaits.
- Aucune action sur Supabase Registry, aucune publication, aucune fusion vers main.

## 5. Critères pour sortir de PRE-ADMISSION

1. Rassembler/sauvegarder les sources primaires avec capture privée
   (`retrieved_at`, `source_date`, taille, SHA-256, MIME, texte des champs).
2. Traiter le conflit de capital 2025 par pièce originale, sans correction implicite.
3. Réconcilier tous les mouvements selon le contrat financier figé ; déterminer la
   base d'enregistrement des rachats (trade/settlement) et les mouvements de droits.
4. Acquérir la pièce officielle Euronext XAMS 2026-09-21 ; distinguer clairement
   `Close`, `Valuation Close`, `Adjusted Close` et source-date.
5. Relecture indépendante puis admission via le chemin de gouvernance autorisé.
