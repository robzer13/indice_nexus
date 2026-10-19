# OroTitan — Besi : audit financier forensique FY2021–FY2025 (V0.1, non certifié)

**Date de l'audit :** 19 octobre 2026 · **Cutoff du run :** 12 octobre 2026 · **Issuer :** BE Semiconductor Industries N.V. (BESI) · **Run Research :** `48b9426b-4d0d-4858-9d87-efcd13ac415d`.

**STATUT : `DRAFT_RESEARCH_INPUTS_NOT_CERTIFIED`.** Ceci est un contrôle financier sur sources primaires/états publiés et un ensemble de formules reproductibles, **pas** un `ANALYSIS_INPUT_LOCK`, un `RESEARCH_FINAL`, un calcul approuvé d'Owner Earnings ou de ROIC, un `GO DEEP_DIVE` ou une autorisation de publication. Aucun artefact scellé ni état Supabase n'a été modifié.

## 1. Autorités et méthode de recoupement

- Communiqué issuer « FY2022 results » du 22 février 2023, états de flux annuels 2022 et comparatifs 2021 **qualifiés audités dans le communiqué** : [source FY2021–2022](https://www.globenewswire.com/fr/news-release/2023/02/22/2612890/0/en/be-semiconductor-industries-n-v-announces-q4-22-and-full-year-2022-results.html).
- Communiqué issuer « FY2023 results » du 22 février 2024, états de flux annuels 2023 et comparatifs 2022 : [source FY2023](https://www.globenewswire.com/news-release/2024/02/22/2833345/0/en/BE-Semiconductor-Industries-N-V-Announces-Q4-23-and-Full-Year-2023-Results.html).
- Rapport annuel ESEF FY2024 du 28 février 2025, **Consolidated Statement of Cash Flows** page 191 et notes IFRS : [source FY2024 / comparatifs FY2023](https://financialfilings.com/filings/be-semiconductor-industries-nv/annual-report-esef/2025/5830006/).
- Communiqué financier officiel FY2025 du 19 février 2026, états de flux 2025 et comparatifs 2024 : [source FY2025 Euronext](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/02/19/Connect_q4-2025.pdf), [transcription accessible des tableaux](https://www.nasdaq.com/press-release/be-semiconductor-industries-nv-announces-q4-25-and-full-year-2025-results-2026-02-19).
- Répertoire émetteur des [rapports annuels originaux FY2021–FY2025](https://www.besi.com/investor-relations/financial-reports-and-publications/financial-reports/).
- Baseline Research : `EVIDENCE_LEDGER v003`, `RESEARCH_SOURCE_MANIFEST v004`, `RESEARCH_GAP_REGISTER v003`, `INPUT_SUFFICIENCY_ASSESSMENT v002`, immuables, run ci-dessus, déposés dans `robzer13/real-orotitan`. Le présent calcul **ne modifie aucun** de ces artefacts.

Les tableaux de flux publiés sont en milliers d'euros ; **conversion /1 000 vers M€**. EBIT, chiffre d'affaires, revenu net, cash, dette et equity sont en **M€ arrondis au dixième** dans les highlights ; ne pas interpréter leurs décimales comme une précision auditée au millier. Les chiffres 2023–2025 ont été recoupés avec les comparatifs des exercices adjacents. **G012 demeure PARTIALLY_RESOLVED** tant que les rapports standalone audités 2021/2022, leurs états ESEF, notes et opinions d'audit n'ont pas été inspectés exhaustivement.

## 2. Résultats financiers observés (M€ sauf marges)

| Exercice | Revenus | EBIT | Marge EBIT | Net income (arrondi) | CFO audité |
|---|---:|---:|---:|---:|---:|
| FY2021 | 749,3 | 317,6 | 42,39 % | 282,4 | 277,854 |
| FY2022 | 722,9 | 294,1 | 40,68 % | 240,6 | 271,871 |
| FY2023 | 578,9 | 213,4 | 36,86 % | 177,1 | 208,572 |
| FY2024 | 607,5 | 195,6 | 32,20 % | 182,0 | 201,083 |
| FY2025 | 591,3 | 173,1 | 29,27 % | 131,6 | 178,141 |

De 2021 à 2025, chiffre d'affaires **−21,1 %**, EBIT **−45,5 %**, CFO **−35,9 %**. La compression de la marge (42,39 % → 29,27 %) montre pourquoi l'année haute FY2021 ne saurait servir d'« état normal » économique sans cycle complet ni lecture des marchés finaux.

## 3. Reconstitution des flux et résiduels (M€)

| Exercice | CFO | PPE capex décaissé | R&D développement capitalisé décaissé | **CFO − PPE − développement** | SBC passée en charges | **Sensibilité − SBC** | Marge résiduel / CA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 | 277,854 | 5,337 | 23,015 | **249,502** | 16,409 | **233,093** | 33,30 % |
| 2022 | 271,871 | 6,780 | 21,613 | **243,478** | 15,259 | **228,219** | 33,68 % |
| 2023 | 208,572 | 6,899 | 21,121 | **180,552** | 19,107 | **161,445** | 31,19 % |
| 2024 | 201,083 | 12,039 | 19,437 | **169,607** | 30,067 | **139,540** | 27,92 % |
| 2025 | 178,141 | 15,795 | 25,994 | **136,352** | 16,375 | **119,977** | 23,06 % |
| **2021–25 cumulé** | **1 137,521** | **46,850** | **111,180** | **979,491** | **97,217** | **882,274** | — |

**Formules :**

1. `cash_residual = cfo - ppe_capex - capitalized_development_cash`.
2. `sbc_sensitivity = cash_residual - sbc_expense`.
3. `cash_residual_margin = cash_residual / revenue`.
4. `CFO = pretax_income + D&A + SBC + financial_expense_adj + change_in_working_capital - cash_taxes_paid + net_interest_received_paid`.

**Contrôle indépendant : le résidu de reconstitution des CFO est égal à 0,000 M€ sur chacune des cinq années.** Le JSON joint permet la vérification automatique avec les montants originaux en M€.

**Interprétation prudente :** le premier résiduel est un **cash-flow après PPE et R&D immobilisée, non un Owner Earnings**. Le deuxième est une sensibilité de coût économique de la rémunération en actions : la charge IFRS de SBC ne correspond ni à une sortie de cash de l'exercice ni nécessairement à la dilution réelle. Il serait abusif de déclarer `owner_earnings_certified=true`. Les flux de working capital restent cycliques et n'ont pas été normalisés.

**Périmètre FY2025 :** acquisition de propriété d'investissement de **5,164 M€** présente comme ligne distincte des flux d'investissement, exclue du « PPE capex » usuel ci-dessus ; elle doit figurer dans l'analyse des décaissements totaux et du caractère récurrent/non récurrent, sans l'imputer automatiquement à la maintenance. Les mouvements des dépôts financiers ne sont pas des investissements opérationnels récurrents.

## 4. Retraitement fiscal : la distorsion FY2024

Dans le rapport annuel FY2024, Besi constate **6,528 M€** de charge fiscale sur **188,520 M€** de résultat avant impôt, soit un taux apparent de **3,46 %**. La société attribue le taux anormalement bas à environ **18,2 M€ d'avantages fiscaux nets**, et indique un taux d'environ **13,1 % hors ajustements** (contre 13,6 % l'année précédente). [Rapport annuel FY2024 ESEF](https://financialfilings.com/filings/be-semiconductor-industries-nv/annual-report-esef/2025/5830006/).

**Simple illustration reproductible, non admission :** en maintenant le résultat avant impôt FY2024 et en appliquant 13,1 %, le bénéfice après impôt serait `188,520 × (1 − 0,131) = 163,824 M€`, au lieu de 181,992 M€ publié à la précision des états. **Écart = 18,168 M€**. Cette illustration ne constitue ni un taux fiscal normatif FY2021–FY2025, ni un ajustement de cash taxes, ni une prévision.

FY2025 : **22,307 M€** de charge fiscale sur 153,948 M€ de résultat avant impôt dans le registre de preuves issu du rapport FY2025 ; le cash taxe réellement payé FY2025 est **28,252 M€** dans le tableau de flux. La différence confirme l'obligation de distinguer impôt comptable, impôt décaissé, différé et exceptionnel.

## 5. Qualité du capital investi / ROIC : ne pas certifier sur une soustraction de bilans

| Fin d'exercice | Cash & deposits | Total debt | Total equity | Net cash (cash − dette) | `equity + debt − cash` |
|---|---:|---:|---:|---:|---:|
| 2021 | 672,2 | 301,8 | 619,3 | 370,4 | 248,9 |
| 2022 | 671,7 | 325,2 | 628,5 | 346,5 | 282,0 |
| 2023 | 413,5 | 300,5 | 421,4 | 113,0 | 308,4 |
| 2024 | 672,3 | 528,5 | 501,3 | 143,8 | 357,5 |
| 2025 | 543,0 | 507,0 | 416,4 | 36,0 | 380,4 |

La dernière colonne est **un indicateur purement financier de bilan**, **pas un capital investi opérationnel approuvé**. Elle ne réconcilie pas à elle seule actifs opérationnels, goodwill et acquisitions, incorporels / R&D, passifs opérationnels non porteurs d'intérêts, locations IFRS 16, actifs excédentaires, trésorerie nécessaire vs non opérationnelle ou change. Une moyenne début/fin correcte nécessite aussi FY2020 pour le ROIC FY2021.

**Définition de travail demandant certification :** `ROIC = NOPAT normalisé / moyenne(capital investi opérationnel, début/fin)`. Doivent être tracés séparément `EBIT IFRS`, changements de comptabilisation R&D, taxes récurrentes, goodwill/operating intangibles, capital circulant, leases et actifs hors exploitation. **Ne pas calculer le ROIC à partir de la colonne indicative** : ce serait un risque élevé de fausse précision.

## 6. Statuts Research et travaux nécessaires

| Contrôle | État après cet audit | Prochaine preuve pour clôturer |
|---|---|---|
| G012, couverture comptes 2021–22 | **PARTIAL**, concordance CFO/Capex/R&D établie | Vérification exhaustive des rapports standalone audités, annexes ESEF et opinions |
| G005, capital investi / ROIC | **OPEN** | Reconstitution des bilans opérationnels / notes d'acquisitions, R&D et fiscalité 2020–25 |
| G006, owner cash | **PARTIAL** | Séparer R&D maintenance/croissance, SBC cash/dilution, normaliser taxes et BFR |
| G016, fiscalité durable | **OPEN** | Notes fiscales FY2021–25 et 2024 Swiss credit / impôts différés |
| G007 et G014, allocation du capital | **OPEN** | Bridge convertibles, actions propres, dividendes et variations nettes de dette |
| C003 / G013, clients hybrid bonding 18 vs 15 | **UNRESOLVED** | Clarification émetteur, définitions et périmètre identiques |

**Décision :** `CONTINUE_RESEARCH`, `ready_for_deep_dive=NO`. Aucun résultat ne doit être incorporé à la version scellée de Research ou à l'Integration V3.1 sans révision contrôlée, justification à la date de cutoff et tests indépendants.

## 7. Fichiers et réutilisation

- `BESI_FY2021_FY2025_FORENSIC_INPUTS_V0_1.json` : 5 exercices, valeurs financières, pointeurs vers les sources, calculs, contrôles, statuts d'admission.
- `tests/equity-besi-research-financial-audit.test.ts` : tests de concordance des CFO, résiduels, sources et absence de faux handoff.
- La PR qui contient ce dossier doit rester **DRAFT / non fusionnée** jusqu'à une revue de la couverture documentaire et de la méthodologie économique.

**Cette étape produit une base reproductible, pas une valorisation ni un nouveau snapshot.**
