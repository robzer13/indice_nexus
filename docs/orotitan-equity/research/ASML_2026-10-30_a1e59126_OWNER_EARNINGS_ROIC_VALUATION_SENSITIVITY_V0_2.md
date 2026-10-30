# ASML — 2026-10-30: Owner Earnings, ROIC and equity valuation sensitivity v0.2

**RUN** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` · `ACTIVE/RESEARCH`  
**Instrument** ASML / NL0010273215 / **Euronext Amsterdam XAMS** / EUR  
**Economic as-of** last published Q3 statements 2026-09-27; full-session price **2026-10-29**; run cutoff **2026-10-30**  
**Status** `PRE_ADMISSION / ILLUSTRATIVE / NOT_REGISTRY_EVIDENCE / NOT_OWNER_EARNINGS_CERTIFIED / NOT_ROIC_CERTIFIED / NO_PUBLISH`.  
**Period policy:** Do not use 2026-10-30 closing prices before the XAMS session closes. Source publication dates must be <= cutoff.  
**PDF custody:** user-provided Euronext-labeled ASML financial statements US GAAP Q3 2026; 12 pages; 1,369,295 bytes; SHA-256 `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9`; SHA equal to SHADOW private receipt. External server-to-server source byte origin not cryptographically attested.  
**Inputs:** [issuer Q3 official statement](https://www.investor.asml.com/quarterly-results), [SEC 6-K filing dated 2026-10-14](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/0001628280-26-065813-index.htm), [issuer exact shares AGM 2026](https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026), [Euronext XAMS intraday transactions of last completed Oct 29 session](https://live.euronext.com/en/popout-page/getIntradayPrice/NL0010273215-XAMS).

## 1. Directly grounded GAAP figures (EUR millions)

Source: the 12-page independently SHA-matched official-looking document, physical PDF pages **3, 5, 6, 7, 9, 10**. Audited status: **UNAUDITED** (PDF p11).

| Indicator | 2025 Q4 | 2026 Q1 | 2026 Q2 | 2026 Q3 | 2026 9M | Source page |
|---|---:|---:|---:|---:|---:|---|
| Operating income (EBIT) | 3,431.1 | 3,157.8 | 3,456.1 | 5,172.8 | 11,786.8 | 3, 7 |
| Net income | 2,839.6 | 2,756.7 | 2,917.6 | 4,960.4 | 10,634.7 | 3, 7 |
| CFO from operations | 11,410.3 | -2,185.6 | 1,703.0 | 11,797.6 | 11,315.0 | 6, 10 |
| Change in operating assets/liabilities in CFO | 7,640.7 | -5,158.1 | -1,614.7 | 7,009.0 | 236.2 | 6, 10 |
| Cash spent on PP&E | 447.9 | 402.4 | 299.4 | 413.3 | 1,115.1 | 6, 10 |
| Cash spent on intangible assets | 22.5 | 19.9 | 86.6 | 33.9 | 140.4 | 6, 10 |

**Rounding:** 2026 quarterly operating income sums to 11,786.7m but the separately presented nine-month figure is 11,786.8m. Preserve a **0.1m displayed rounding residual**. For LTM use 2025 Q4 plus the 2026 9M reported total, not a silently overwritten quarterly sum.

### 1.1 Noncurrent contract liability — classification correction

In the **US GAAP** consolidated balance sheet on physical PDF p5, `Contract liabilities` of **3,366.3m as at Dec 31 2025** and **9,300.6m as at Sep 27 2026** appear in **NONCURRENT liabilities**. The Q2 prior quarter p9 gives **3,570.3m** in the same **noncurrent** presentation. Therefore **+5,730.3m Q2→Q3 is a NONCURRENT balance-sheet movement, not the change in *total* customer prepayments or the CFO change from contract liabilities.** Current liabilities are aggregated in this summary GAAP table; their contract subcomponents are not disclosed here. Do not add 5,730.3 to the CFO's 7,009.0 or call it `customer_advances_received`. More granular regulatory financial statement notes or company confirmation required.

An **EU-IFRS statutory interim report** published July 2026 presents current and noncurrent contract liability lines separately. But EU-IFRS liability amounts and capital are **not** interchangeable with US GAAP financials for invested capital / ROIC. Share *numbers*, which represent legal shares, are distinct from accounting-basis amounts.

## 2. Cash quality / LTM arithmetic — reproducible, not normalized

All amounts in EUR millions, using the disclosed GAAP p6/p10 values:

| Bridge | Expression | Result | Authority |
|---|---|---:|---|
| 2026 Q3 simple FCF | CFO 11,797.6 − PP&E 413.3 − intangibles 33.9 | **11,350.4** | arithmetic only |
| 2026 9M simple FCF | CFO 11,315.0 − PP&E 1,115.1 − intangibles 140.4 | **10,059.5** | arithmetic only |
| 2025 Q4 simple FCF | CFO 11,410.3 − PP&E 447.9 − intangibles 22.5 | **10,939.9** | arithmetic only |
| LTM CFO | Q4 2025 11,410.3 + 9M 2026 11,315.0 | **22,725.3** | arithmetic only |
| LTM capitalized PPE and intangibles cash outflows | 470.4 + 1,255.5 | **1,725.9** | arithmetic only |
| **LTM simple FCF** | 22,725.3 − 1,725.9 | **20,999.4** | NOT Owner Earnings |
| LTM net change in operating assets/liabilities in CFO | Q4 2025 +7,640.7 + 2026 9M +236.2 | **+7,876.9** | reported net flow bridge |
| LTM CFO **excluding that single reported line** | 22,725.3 − 7,876.9 | **14,848.4** | heuristic, not sustainable CFO |
| LTM simple FCF **excluding that single line** | 20,999.4 − 7,876.9 | **13,122.5** | heuristic sensitivity, not normalized Owner Earnings |

These show **EUR 7.8769bn or 37.5%** of LTM *simple FCF* is due to the net positive `Changes in assets and liabilities` item. **Do not claim all of it is non-recurring**: customer advances and cycle timing may be recurring elements of the business model, while another period may reverse them. Neutralization is a sensitivity bound, not an approved adjustment.

Further bridges required: equipment shipment cash conversion vs customer deposits/prepayments, contract liability *current and noncurrent* changes, accounts receivable/unbilled assets, inventory reserves, tax, FX/noncash movements, depreciation vs **maintenance** capex, investments and share-based compensation, debt service, working capital seasonality, backlog/order cancellations and technology cycle.

The 2026 Q3 net income includes a **EUR 509.7m non-operating gain on equity investments** (p3) and the indirect CFO adjustment **−EUR 508.8m** (p6). **Do not deduct it twice** from cash flow; reconcile its cash/noncash treatment in separate schedules.

## 3. Preliminary LTM operating return / ROIC diagnostics

Inputs (GAAP; EUR millions):
- 2025 Q4 EBIT **3,431.1** + 2026 9M EBIT **11,786.8** = **LTM EBIT 15,217.9**.
- 2025 Q4 reported pretax earnings **3,442.3** + 2026 9M **12,435.5** = **15,877.8**; taxes 618.7 + 2,081.2 = **2,699.9**, **blended group effective tax rate ≈ 17.0042%**.
- Mechanical LTM **NOPAT proxy** = 15,217.9 × (1 − 17.0042%) = **EUR 12,630.2m**. The group tax rate is **not necessarily a valid tax rate for normalized operating profit**; equity investment gains, tax credits, deferred taxes and discrete items must be reconciled.
- NOPAT sensitivity at assumed 15% / 20% / 25% operating taxes = **12,935.2 / 12,174.3 / 11,413.4m**. These are scenario parameters, not estimates of the actual statutory rate.

**Denominator diagnostic (deliberately NOT a certified ROIC)**. The most naive formula,
`equity + reported long-term debt − cash − short-term investments`, yields:

| GAAP balance-sheet date | Equity | Long-term debt | Cash | Short-term investments | Crude invested-capital *proxy* |
|---|---:|---:|---:|---:|---:|
| 2025-12-31 | 19,612.2 | 2,709.0 | 12,916.0 | 405.9 | **8,999.3** |
| 2026-09-27 | 22,063.9 | 1,963.0 | 8,030.9 | 4,704.2 | **11,291.8** |

It gives proxy average **10,145.55m** and an implausibly high **124.5%** if divided into the mechanical NOPAT. **This is explicitly a method FAIL, not "ASML ROIC = 124.5%"**. It mixes an incomplete debt measure (current maturities, finance leases), potentially excess vs operational cash, major customer-financed negative working capital, non-operating investments and two different seasonal balance-sheet dates. It also mixes a 12-month numerator and a Dec/Sep 9-month denominator window. A defensible ROIC requires average *quarterly or monthly* invested capital with consistent GAAP classification, NOPAT operating tax bridge, current interest debt + leases, customer advance treatment, capitalized development if appropriate, non-operating assets and a reliable opening period.

**ROIC status: BLOCKED / not estimated**; the diagnostics demonstrate where a simplistic computed ratio becomes misleading.

## 4. Equity valuation sensitivity — vendor scenario only

**Price**: EUR 1,504.80 on **2026-10-29**, 660,425 shares traded, corroborated at [FT](https://markets.ft.markitdigital.com/data/equities/tearsheet/historical?s=ASML%3AAEX), [StockAnalysis](https://stockanalysis.com/quote/ams/ASML/history/) and [Investing.com](https://www.investing.com/equities/asml-holding-historical-data). Separately, Euronext XAMS's [official intraday transaction page for the October 29 session](https://live.euronext.com/en/popout-page/getIntradayPrice/NL0010273215-XAMS) lists transactions at **17:39:46 / 1,504.80 EUR / Trading at Last**. The page supports the **official venue print** but not, by itself, a sealed official unadjusted EOD file.

**Cutoff discipline:** the research was conducted 2026-10-30 **14:32 CET** (before the end of the Amsterdam session). The 30 Oct intraday quote must **not** be substituted for a completed Oct 30 close. A major intraday move may make an Oct 29 valuation significantly different from a later close.

The exact **issued less treasury at compatible as-of instant** remains **unavailable** from published issuer disclosures. Official [AGM 2026 exact shares at 2026-03-25](https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026):
`388,147,674 issued − 3,224,029 treasury = 384,923,645 economic shares` (historic exact, NOT October). The official [statutory interim at 2026-06-28](https://www.investor.asml.com/quarterly-results) discloses **384.1m** rounded shares, about **1.7m H1 repurchases** and **0.4m share issuances** in millions rounded. Subsequently the issuer disclosed **1,299,377 gross repurchases** between Sep21–Oct23, as itemized in the preceding research note. No full share issuance / transfer / cancellation ledger or Oct26–30 post-cutoff-disclosed trades available at this instant.

Vendor scenarios — **NOT approved economic-share quantities**:

| Hypothetical shares (millions) | Equity value at EUR 1,504.80 (EUR bn) |
|---:|---:|
| 380.00 | 571.824 |
| 381.00 | 573.329 |
| 382.91 | **576.203** |
| 384.05 | 577.918 |
| 385.00 | 579.348 |

Using `382.91m` as a **purely illustrative vendor share-count placeholder** and LTM 2025 Q4 net income 2,839.6 + 2026 9M net income 10,634.7 = **EUR 13.4743bn**:

| Multiples on **illustrative EUR 576.203bn market cap** | Multiple | Main deficiency |
|---|---:|---|
| Price / LTM **reported GAAP net earnings** | **42.76×** | Non-operating €509.7m Q3 investment gain, cyclical quality |
| Price / LTM **simple FCF** | **27.44×** | Net CFO working-capital timing inflates cash conversion |
| Price / LTM **FCF excluding reported changes in operating assets/liabilities** | **43.91×** | Omitting all net WC changes is itself not a valid normalization |

At this price, **1 million shares of share-count error changes estimated equity value by EUR 1.5048bn**; every 1% price variation changes the value by ≈ EUR 5.76bn at the 382.91m placeholder. These ratios demonstrate high sensitivity to cash-normalization choices, **not an investable valuation or an approved forecast**.

**Enterprise value is not certified**. US GAAP p5 reports 2026-09-27 cash **8,030.9m**, short-term investments **4,704.2m**, and long-term debt **1,963.0m**, but these are not a full net-debt reconciliation; current borrowings and leases, non-operating investments and timing must be reconciled.

## 5. G001/G006 conclusion and Research action sequence

- `G001_BLOCKED_EXACT_2026_10_30_ECONOMIC_SHARES`: official AGM 2026 exact shares and official H1 2026 rounded share bridge, Q3 average-weighted share count and public weekly share repurchases still insufficient. Never substitute 383.1m (Q3 EPS weighted average), 384.1m (H1 rounded issued/outstanding), 385.42m (vendor stale 2025 value) or 382.91m (vendor alternative).
- `G006_OFFICIAL_VENUE_TRADE_SUPPORTED_EOD_UNSEALED`: venue-specific Euronext official late-trade line **1,504.80** on 29 Oct supports vendor closing-price corroboration; still require Euronext historical EOD non-adjusted extraction and custody. Dividend ex-date XAMS 26 October 2026 EUR 1.88 must not be silently adjusted.
- `OWNER_EARNINGS_NOT_CERTIFIED`: use the **20,999.4m vs 13,122.5m** LTM cash conversion sensitivity only; pursue customer advances' economic function and recurring working-capital cycles.
- `ROIC_NOT_CERTIFIED`: current two-point simple denominator is explicitly rejected; reconstruct GAAP-only invested capital quarterly and operating taxation, note data missing.
- `SOURCE_PDF`: byte match to second supplied Euronext-labeled file confirmed; full source origin and immutable admission review still distinct.
- `RESEARCH_STAGE`: `IN_PROGRESS` and `READY_FOR_DEEP_DIVE=NOT_EVALUATED`; no registry write or publication.

**Next actual primary artifacts:** official shareholder/corporate action ledger at cutoff; official Euronext XAMS unadjusted **2026-10-29 EOD record**; issuer current/non-current customer-deposit note with cash settlements; 2025 US GAAP annual cash flow and 2026 interim US GAAP notes (separate from EU-IFRS); operating invested-capital policy and monthly working-capital profile. Do not quietly remove blockers to advance stage.
