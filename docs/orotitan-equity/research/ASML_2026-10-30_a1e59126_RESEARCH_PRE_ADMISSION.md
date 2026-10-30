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

**Period check:** Q1 2026 CFO **−€2,185.6m**, Q2 **+€1,703.0m**, Q3 **+€11,797.6m** sum to the reported nine-month **+€11,315.0m**. Cash conversion is exceptionally sensitive to the third-quarter working-capital movement.\n\n**Red flag for methodology:** about **59.4%** of Q3 operating CF comes from reported changes in assets/liabilities. The financial statements also show very large **noncurrent** contract-liability movements (p5 and p9). This is not a direct measure of the total cash advances received. Distinguish customer prepayments, deferred revenue and working-capital timing before any cash-conversion, ROIC, owner-earnings or DCF judgments. Q3's nonoperating gain must be isolated from operating return quality.

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

## VIII bis. Q3 acquisition transport blocker (CI observed 2026-10-30)

Automated read-only acquisition test against SEC exhibits 99.1–99.3 on GitHub Actions:
**EX-99.1 returned HTTP 403** directly from `www.sec.gov`, without redirect.
Other exhibits were not considered verified by the aborted strict probe.
ASML investor download URLs for the Q3 PDFs have also returned HTTP 403
to automated web access. Do not circumvent origin restrictions, and do not
convert a rendered webpage into invented original source bytes.

A strictly gated SHADOW Preview offers a **browser-supplied PDF** fallback:
`/admin/research-q3-sources`. It accepts only a size-bounded PDF, stores
SHA-addressed bytes and an immutable receipt in the PRIVATE SHADOW bucket,
with mandatory readback. A browser upload is not proof that bytes originated
from ASML/SEC. Its receipt explicitly records
`sourceOriginAuthenticated:false`,
`sourceDocumentAuthenticityVerified:false`, `registryAdmitted:false`.
Such a receipt cannot satisfy Research Evidence Ledger admission or unfreeze
`READY_FOR_DEEP_DIVE` without independent source authentication.
A browser-provided Q3 file was subsequently archived by the administrator; see Section VIII ter below. This research review itself has not uploaded any Q3 file.

## VIII ter. Captured Q3 browser-supplied PDF — SHADOW observed 2026-10-30

The administrator submitted a 12-page **Financial statements US GAAP PDF** through the strictly gated SHADOW upload page. A direct read-only query of Supabase Storage confirms **one PDF object** and **one receipt JSON** in the PRIVATE bucket `orotitan-source-files-v1`:

- `PDF_SIZE_BYTES = 1,369,295`, MIME `application/pdf`, created and last updated **2026-10-30T13:02:08.032176Z**.
- `SHA256_FROM_SHA_ADDRESSED_PATH = 562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9`.
- `PDF_OBJECT = research-acquisition-v1/asml/asof-2026-10-30/q3-2026/browser-supplied/562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9.pdf`.
- `RECEIPT_OBJECT = research-acquisition-v1/asml/asof-2026-10-30/q3-2026/browser-receipts/562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9.json`, 926 bytes, recorded immediately after PDF.
- The authenticated application reported **readback SHA-256 success**, but this research review has NOT independently recomputed the SHA by downloading the private bytes.

**Interpretation:** private custody / content-addressed receipt = observed. Source authenticity / independent byte identity with ASML or Euronext = NOT VERIFIED. A dated Euronext-hosted copy is available as [possible official binary reference](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/10/14/cpr01_notified_Financial%20statements%20US%20GAAP%20Q3%202026.pdf), but its bytes **were not downloaded and compared**. The PDF's text and document content have not been reviewed through an admitted forensic evidence procedure.

`GAP-001 = PARTIALLY_PROGRESSSED_NOT_CLOSED`; `READY_FOR_DEEP_DIVE = NOT_EVALUATED`; no source acceptance, stage finalization, data certification or Registry write.

## VIII quater. Independent copy SHA-256 cross-check — MATCH (2026-10-30)

**Evidence input**: user supplied a separately downloaded PDF labeled
`cpr01_notified_Financial statements US GAAP Q3 2026.pdf`, following the
[official Euronext-hosted Q3 2026 financial statements link](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/10/14/cpr01_notified_Financial%20statements%20US%20GAAP%20Q3%202026.pdf).

**Independent local-file hash computation** (not a website text snippet):

| Field | Euronext-labeled submitted PDF | SHADOW private stored capture receipt |
|---|---|---|
| SHA-256 | `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9` | `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9` |
| Byte length | **1,369,295** | **1,369,295** |
| Page count | **12** | 12 reported by the administrator, not separately measured from SHADOW bytes |
| PDF metadata | title: `Financial statements US GAAP Q3 2026`; producer: Workiva/Wdesk; creation 2026-10-13 UTC; PDF 1.4 | original SHADOW file bytes not directly downloaded by research agent |

**Binary comparison result:** `USER_SUPPLIED_EURONEXT_COPY_EQUALS_SHADOW_CAPTURE_SHA256 = YES`. The original SHADOW capture was written under a content-addressed path with mandatory
application-side readback. The new uploaded PDF was independently SHA-256 hashed
from the complete 1,369,295 local bytes, giving the **same** hash and size.

**Document content spot-check:** T3 2026 sales **EUR 11,662.0m** and net income
**EUR 4,960.4m** are printed on PDF page 3; gross margin **57.5%** on page 4;
CFO **EUR 11,797.6m** and asset/liability change **EUR +7,009.0m** on page 6.
These support the existing pre-admission facts, but do not replace the
normal Evidence Ledger and independent quantitative review.

**Limit of proof:** this comparison proves **content identity between two files**.
The second file is an administrator/user-supplied download associated with the
previously identified Euronext public URL, **not a cryptographically authenticated
server-to-server Euronext download**. PDF title, metadata, publisher branding and
the user's filename cannot alone establish chain-of-custody/origin cryptographically.
The stored SHADOW object itself was not independently downloaded for SHA calculation
during this review. The issuer's Euronext publication context remains documentary
corroboration, not proof of independent economic information.

**Decision:** `BINARY_MATCH_TO_SEPARATELY_SUPPLIED_EURONEXT_COPY = VERIFIED`;
`SHADOW_APPLICATION_READBACK = VERIFIED_BY_APPLICATION`;
`EXTERNAL_DOWNLOAD_CHAIN = NOT_CRYPTOGRAPHICALLY_ATTESTED`;
`SOURCE_ORIGIN_AUTHENTICATED = PENDING_AUTHORITY_REVIEW`;
`REGISTRY_ADMISSION = NO`.
`GAP-001 = EVIDENCE_OF_SOURCE_IDENTITY_PROGRESS / OPEN`.
`READY_FOR_DEEP_DIVE = NOT_EVALUATED`; NO Registry mutation.

**Further cash/ROIC/multiple scenarios (v0.2):** [Full working-capital and noncurrent contract-liability correction, LTM FCF sensitivity, NOPAT and rejected naive ROIC](./ASML_2026-10-30_a1e59126_OWNER_EARNINGS_ROIC_VALUATION_SENSITIVITY_V0_2.md); [machine-recomputable scenario data](./ASML_2026-10-30_a1e59126_ROIC_CASH_MULTIPLES_MODEL_V0_2.json). These are non-admitted research diagnostics, **not** Owner Earnings, ROIC, market capitalization or official close certification.

**Reconciliation companion (2026-10-30):** [PDF source qualification / issued-share vs treasury / working capital / market close / control gates](./ASML_2026-10-30_a1e59126_SOURCE_CAPITAL_CASH_PRICE_RECONCILIATION_V0_1.md).

**G001/G006 authority conflict review (v0.3):** [dated issuer share ledger, four mutually inconsistent vendor share-counts, employee treasury delivery mechanics and exact issuer-authority request (NOT SENT)](./ASML_2026-10-30_a1e59126_G001_G006_AUTHORITY_CONFLICT_V0_3.md). **Conclusion:** exact 2026-10-30 economic shares UNKNOWN; 2026-10-29 XAMS EUR 1,504.80 vendor close is corroborated, but original official unadjusted EOD custody remains missing. Do not average vendors or calculate certified market cap.

**G001/G006 materiality resolution proposal (v0.4):** [two-track solution, 0.532% vendor-count sensitivity, 2026-10-05 Euronext capital-increase notice and bounded Pilotage adjudication](./ASML_2026-10-30_a1e59126_G001_G006_MATERIALITY_RESOLUTION_PROPOSAL_V0_4.md). It is a **PROPOSAL only**: analytical RESEARCH working scenarios may continue, but it neither waives a frozen source requirement nor permits stage completion, official market capitalization, ROIC/Owner Earnings certification or publication. The **2026-10-08 Euronext corporate action** requires original notice verification.

**Working ASML business / runway / outside-view Research (v0.1, 2026-10-30):** [source-scoped research note](./ASML_2026-10-30_a1e59126_BUSINESS_MOAT_RUNWAY_OUTSIDE_VIEW_RESEARCH_WORKING_V0_1.md) and [machine-readable evidence candidates](./ASML_2026-10-30_a1e59126_BUSINESS_OUTSIDE_VIEW_EVIDENCE_CANDIDATES_V0_1.json). Evidence covers 2025 annual report, Q3 2026 issuer statements, customer TSMC and Samsung 2026 reports, core lithography economics, technical adoption, service mix, supply-chain concentration and risk hypotheses. These are **working-source candidates and hypothesis tests, not a final moat/runway/quality verdict**. They proceed independently of incomplete `G001`, without overriding its certification gate.

**Order → revenue → installed base → cash research (v0.2, 2026-10-30):** [report](./ASML_2026-10-30_a1e59126_ORDER_TO_REVENUE_IBM_CASH_SENSITIVITY_V0_2.md) and [machine-reproducible scenario inputs](./ASML_2026-10-30_a1e59126_ORDER_IBM_CASH_WORKING_MODEL_V0_2.json). **2026 Q1–Q3 numeric bookings and 2026 Q3 order backlog are not publicly quantified in the reviewed issuer results; 2025 year-end backlog = EUR 38.797bn historical only.** Separate down-payment and liability lag from original PO, outstanding backlog and recognized sales. Conditional Q4 2026 IBM ~EUR 3.7bn leads to FY2026 IBM ~EUR 12.238bn, explicitly *management forecast derived*. LTM simple cash-flow price/yield **stress scenarios** vary with retention of the net assets/liabilities CFO line; neither Owner Earnings nor approved valuation, price target or DCF. Does not require G001 exact shares to continue Research work; G001/G006 official certification still open.

**IBM service-vs-upgrade durability and multi-year US GAAP FCF forensics (v0.3):** [source-based analysis](./ASML_2026-10-30_a1e59126_IBM_REVENUE_DURABILITY_AND_CASH_QUALITY_RESEARCH_V0_3.md) and [reproducible source/metric model](./ASML_2026-10-30_a1e59126_IBM_DURABILITY_WORKING_MODEL_V0_3.json). 2023/24/25 IBM revenue **EUR 5.6199 / 6.4942 / 8.1930bn** and segment **gross profit EUR 2.3485 / 3.1302 / 4.1677bn** (US GAAP SEC 20-F XBRL R3); margins **41.79 / 48.20 / 50.87%**, before unallocated operating costs and capex. SEC ASC 606 Note 2 establishes separate contract-period service, usage maintenance, billable spares and episodic field upgrades; **no disclosed euro breakdown or contractual recurring mix**. LTM actual IBM to Q3 2026 **EUR 10.6722bn**, but historical all-IBM sales stress anchors **must not be described as current contractual floors**. Total contract liability 2025 **EUR 19.3726bn**, recognized revenue from prior obligation **EUR 11.5166bn** is **NOT exclusively IBM**. 2023/24/25 group US GAAP FCF **EUR 3.2472 / 9.0831 / 11.0273bn**; 2026 LTM FCF **EUR 20.9994bn** remains working-capital timing sensitive. **Exact durable IBM revenue share and IBM cash profitability cannot be calculated**; proceed to quantitative stress testing without inferring guarantees, final verdicts, or advancing Stage Gate.

**2027–2029 sales, segment gross-margin and operating-cash stress scenarios (Research v0.4):** [scenario methodology and nine-yearcase report](./ASML_2026-10-30_a1e59126_GROWTH_MARGIN_CASH_CONVERSION_RESEARCH_REPORT_V0_4.md), [reproducible JSON](./ASML_2026-10-30_a1e59126_GROWTH_MARGIN_CASH_CONVERSION_SCENARIOS_V0_4.json). Anchored to ASML's October 2026 FY sales/gross margin **guidance**, 2026 Q3 US GAAP actual, and 2025 SEC US GAAP IBM service margin **50.87%**. Carrying 2025 IBM margin into FY2026 is **only a modeling assumption**; system margin 57.946% is the algebraic residual, not a reported figure. 2027–2029 downside/central/upside assume separate systems/IBM growth, product GM, R&D/SG&A, tax, capex, D&A and operating cash timing. By 2029 group revenue is EUR **40.774 / 56.112 / 64.582bn**; EBIT margin **33.676 / 40.975 / 44.171%**; operational *cash stress proxy* EUR **9.965 / 17.226 / 19.788bn**. **PROXIES ONLY**, not CFO, normalized FCF, FCFF, Owner Earnings, official forecasts, terminal verdicts or DCF; assumptions have no assigned probabilities. Exact IBM contracted recurring fraction still **UNKNOWN**, and G001/G006 do not stop analytic continuation but formal Research completion stays gated.\n\n**IBM margins and consolidated sustaining-CapEx evidence (v0.5, cutoff 2026-10-30):** [primary US GAAP margin/PPE calibration and no-double-count analysis](./ASML_2026-10-30_a1e59126_IBM_MAINTENANCE_CAPEX_MARGIN_CALIBRATION_REPORT_V0_5.md), [structured calculation/sensitivity model](./ASML_2026-10-30_a1e59126_IBM_MAINTENANCE_CAPEX_MARGIN_CALIBRATION_MODEL_V0_5.json). Official 2025 group PPE depreciation **EUR 916.0m** and group cash PPE + intangible CapEx **EUR 1,631.2m**; accounting PPE additions **EUR 1,596.2m**, mostly described as capacity and R&D growth support but not a disclosed maintenance-vs-growth split. **IBM upkeep CapEx is not separately disclosed**; direct customer service parts/labor already enter IBM COGS and corporate sustaining technology R&D already reduces US GAAP earnings. FY2025 service GM **50.869%** (not IBM contract-only margin). Hypothetical 2029 central scenario stress of IBM GM ±5 percentage points, group capex ±1% sales and WC cash timing ±2% sales drives a group cash **STRESS PROXY** from **EUR 14.893–19.559bn**, **not an Owner Earnings bound**; the prior working v0.4 case stays unchanged. Do not mislabel annual company depreciation as IBM maintenance or an exact replacement cash budget. No gate change or official Registry artifact.\n\n## IX. Next authorized Research actions

1. Review and approve the now-confirmed SHA match to the separately uploaded Euronext-labeled PDF; establish admissible source provenance, file custody, issuer/filing context and authority requirements independently. Only then consider Registry admission (not automatic).
2. Prepare full financial evidence ledgers (normalized claims + conflicts + calculation proof + exact as-of dates) using frozen Research contract, including 2025 20-F and interim 2026.
3. Reconcile exact share denominator and XAMS quote at a coherent market price instant.
4. Produce required Research manifests only once bytes have been sealed and independently reviewed. Do not self-certify or enter Deep Dive on a human summary.

**Prohibitions:** no Registry artifact creation from this review; no stage completion; no canonical snapshot; no `GO PUBLISH`; no final investment verdict.
