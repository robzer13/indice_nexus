# OroTitan · ASML · Research pré-admission — 30 octobre 2026

**RUN_ID :** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24`  
**Issuer :** ASML Holding N.V. · ISIN NL0010273215 · XAMS · EUR  
**Run :** ANALYZE / INITIAL · Process V3.1 · Research Stage Contract V2.0  
**DATA_CUTOFF :** 2026-10-30  
**Classification :** RESEARCH_WORKING_CANDIDATES / NON_ADMITTED / NOT_SEALED / NOT_CERTIFIED  
**Authority :** **NONE**. This is not the frozen Evidence Ledger, a FINAL stage manifest, an admitted source or a valuation.

Machine-readable companion: [candidate evidence register](./ASML_2026-10-30_a1e59126_RESEARCH_PRE_ADMISSION.json).

## I. Research pre-flight (verified)

- Registry status last observed: `ACTIVE / RESEARCH / IN_PROGRESS`, run state version 3, 0 artifacts and handoff `NOT_EVALUATED`.
- Security/dossier: existing ASML canonical identity; **no published canonical snapshot**.
- The old 2026-09-19 and 2026-09-21 ASML RUNs remain separate. Their gates are **not** inherited into this RUN.
- `SOURCE_DATE <= DATA_CUTOFF` applies. **Reporting-period end 2026-09-27 ≠ publication date 2026-10-14 ≠ cutoff 2026-10-30.**
- All numbers below are source-referenced CANDIDATES, with no raw byte capture or authoritative admission.

## II. Primary-source inventory

| Key | Primary document | Publication date | Financial as-of / period | Source status |
|---|---|---|---|---|
| S01 | [SEC Form 6-K accession 0001628280-26-065813](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/0001628280-26-065813-index.htm) | 2026-10-14 | quarter ended 2026-09-27 | Filing indexed; raw bytes NOT archived |
| S02 | [SEC Exhibit 99.1 — Q3 2026 issuer release](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/pressreleasefinancialresul.htm) | 2026-10-14 | Q3 2026 | Content verified by inspection; NOT sealed |
| S03 | [SEC Exhibit 99.2 — Q3 IR presentation](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/a2026_10x14xpresentation.htm) | 2026-10-14 | Q3 2026 | Content verified by inspection; NOT sealed |
| S04 | [SEC Exhibit 99.3 — unaudited US GAAP statements](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/financial_statementsxusx.htm) | 2026-10-14 | 2026-09-27 / 9M | Content verified by inspection; NOT sealed |
| S05 | [ASML issuer quarterly release](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026) | 2026-10-14 | Q3 2026 | Same issuer facts as SEC, not independent confirmation |
| S06 | [ASML share repurchases — 26 October](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-6) | 2026-10-26 | trades 19–23 October | Exactly 343,310 shares; still no full share capital ledger |

**Acquisition warning:**  SEC Exhibit 99.3 embeds its financial slides as separate referenced images. **Do not archive only the HTML wrapper and claim the complete financial statements were sealed**: the linked slide images (or equivalent full official PDF) must be captured and integrity-checked as a consistent set. SEC-hosted exhibits and issuer pages are original *locations* but inspection of their web-rendered contents does **not** provide captured raw bytes, content SHA-256, sealed custody or official Registry admission. ASML's investor static-file PDF endpoints can return HTTP 403 from cloud environments; SEC HTML is a candidate alternative, NOT automatically certified.

## III. Evidence candidate ledger (partial) — all numbers EUR millions unless indicated

Figures from SEC Exhibit 99.3, unaudited US GAAP:

| Fact | Q3 2026 | 9M 2026 where relevant |
|---|---:|---:|
| Total net sales | **11,662.0** | 29,755.4 |
| System sales | 8,373.2 | 21,217.3 |
| Service / options (Installed Base Management) | 3,288.8 | 8,538.1 |
| Gross profit | 6,699.8 | 16,380.3 |
| Operating income | 5,172.8 | 11,786.8 |
| Net income | 4,960.4 | 10,634.7 |
| Net cash from operating activities | **11,797.6** | **11,315.0** |
| Change in assets and liabilities within operating CF | **+7,009.0** | +236.2 |
| Purchase of PP&E (cash outflow absolute) | 413.3 | 1,115.1 |
| Purchase of intangible assets (cash outflow absolute) | 33.9 | 140.4 |
| Buyback cash outflow | 4,071.0 | 6,148.4 |

Other material facts:

- Q3 2026 gross margin **57.5%**, operating margin **44.4%**; net income includes a **€509.7m gain on equity investments** (non-core classification requires later scrutiny).
- Quarter-end cash **€8,030.9m** plus short-term investments **€4,704.2m** = **€12,735.1m** on 27 September 2026, not automatically net cash.
- Basic weighted-average shares **383.1 million** in Q3 (rounded and period average), **NOT** economic shares outstanding at the valuation cutoff.
- Issuer FY2026 sales guidance **€44–45bn**; Q4 sales guidance **€14.2–15.2bn**, both management forward-looking estimates as at 14 October. No automatic probability or realization.
- As of 26 October, the issuer's latest weekly notice covers **19–23 October** with 67,063 + 68,250 + 67,400 + 69,297 + 71,300 = **343,310** repurchased shares. Other flows and the exact treasury ledger remain unverified.

## IV. Cash generation — calculation candidates, NOT Owner Earnings

Arithmetic only (values €m):

| Check | Arithmetic | Result | Authority |
|---|---|---:|---|
| Q3 CFO minus PP&E and intangible purchases | 11,797.6 − 413.3 − 33.9 | **11,350.4** | *Raw simplified cash-flow bridge, not normalized FCF/Owner Earnings* |
| 9M CFO minus PP&E and intangible purchases | 11,315.0 − 1,115.1 − 140.4 | **10,059.5** | *Raw simplified cash-flow bridge, not normalized FCF/Owner Earnings* |
| Q3 CFO excluding recorded net asset/liability changes | 11,797.6 − 7,009.0 | **4,788.6** | *Sensitivity only: not normalized/sustainable CFO* |

**Period check:** Q1 2026 CFO **−€2,185.6m**, Q2 **+€1,703.0m**, Q3 **+€11,797.6m** sum to the reported nine-month **+€11,315.0m**. Cash conversion is exceptionally sensitive to the third-quarter working-capital movement.\n\n**Red flag for methodology:** about **59.4%** of Q3 operating CF comes from reported changes in assets/liabilities. The financial statements also show very large contract-liability movements. Distinguish customer prepayments, deferred revenue and working-capital timing before any cash-conversion, ROIC, owner-earnings or DCF judgments. Q3's nonoperating gain must be isolated from operating return quality.

## V. Valuation-date alignment and market data

**The date 2026-10-30 is a calendar-day cutoff, not proof of a completed market session.**

- Secondary historical market sources [FT](https://markets.ft.markitdigital.com/data/equities/tearsheet/historical?s=ASML%3AAEX) and [Investing.com](https://www.investing.com/equities/asml-holding-historical-data) corroborate **€1,504.80** for Amsterdam close **2026-10-29**, volume **660,425**, not an independently sealed official Euronext XAMS EOD record.
- Any **2026-10-30** vendor quote obtained during market hours is `INTRADAY`, not `2026-10-30 UNADJUSTED_CLOSE`. If the final XAMS closing price is not available within the observation cutoff, use an explicitly dated 29 October close per the pinned date-alignment authority.
- **Market capitalization calculation remains BLOCKED**: exact `ECONOMIC_SHARE_COUNT` at a compatible instant is not established.

## VI. Hypothesis register (questions, not thesis)

1. **H-001** — To what extent do demand plans for 2027–2030 become confirmed orders, shipments, revenue and economic returns, subject to customer CAPEX and export licensing?
2. **H-002** — How much of Q3 cash generation reflects customer advance payments / working-capital timing as opposed to a repeatable operating cash yield?
3. **H-003** — Are the returns on High-NA EUV and capacity expansion robust to supplier constraints, cost intensity and the customers' roadmaps?

## VII. Conflict candidates (not silent reconciliation)

- **C-001 — Share count:** reported *weighted-average* basic EPS shares of 383.1m cannot be used as an exact closing economic denominator. Q3 repurchases ≈2.8m are not the whole capital-movement bridge.
- **C-002 — Quality of cash:** record Q3 CFO is materially affected by asset/liability changes; do not extrapolate or identify with Owner Earnings.
- **C-003 — Price provenance:** secondary EUR 1,504.80 close Oct 29 is not official Euronext XAMS market-close evidence for Oct 30.

## VIII. GAP Register and INPUT_SUFFICIENCY = NO

| ID | Priority | Research gap | Exit requirement |
|---|---|---|---|
| GAP-001 | **CRITICAL** | Unsealed primary bytes | Original Q3 sources privately captured with retrieval time, exact bytes and SHA-256, secure locator and read-back |
| GAP-002 | **CRITICAL** | Economic shares | Exact issued, treasury, settlements, issuances, cancellations and share plan changes through cutoff |
| GAP-003 | **CRITICAL** | Official market price | Euronext XAMS as-of price and unadjusted/adjusted convention, exact time alignment |
| GAP-004 | **CRITICAL** | Sustainable operating cash | Bridge working capital, deferred revenue, recurring cash conversion, maintenance vs growth capex, investment gains |
| GAP-005 | HIGH | ROIC denominator | Invested capital/NOPAT accounting reconciliation and nonoperating assets |
| GAP-006 | HIGH | Outside-view and risk | Independent competitors, customer equipment CAPEX, China restrictions, supplier and order-conversion checks |
| GAP-007 | HIGH | Business and governance | 2025 20-F, customer concentration, capital allocation and governance source coverage |
| GAP-008 | HIGH | Research stage deliverables | Full *authoritative* ledger lineage, classification inputs verified against frozen taxonomy, assumptions/conflicts, Research Stage Manifest and self-audit |

**Classification inputs:** Netherlands issuer/listing, ASML/XAMS identity. No frozen taxonomy code assigned and no PEA eligibility assessment.

**Output state:** `RESEARCH IN_PROGRESS`; `READY_FOR_DEEP_DIVE = NOT_EVALUATED` in Registry. This review itself cannot change that gate.

## IX. Next authorized Research actions

1. Capture Q3 source bytes from a trusted issuer/SEC origin **into private source storage**, no uploads to public GitHub; hash, timestamp and verify custody.
2. Prepare full financial evidence ledgers (normalized claims + conflicts + calculation proof + exact as-of dates) using frozen Research contract, including 2025 20-F and interim 2026.
3. Reconcile exact share denominator and XAMS quote at a coherent market price instant.
4. Produce required Research manifests only once bytes have been sealed and independently reviewed. Do not self-certify or enter Deep Dive on a human summary.

**Prohibitions:** no Registry artifact creation from this review; no stage completion; no canonical snapshot; no `GO PUBLISH`; no final investment verdict.
