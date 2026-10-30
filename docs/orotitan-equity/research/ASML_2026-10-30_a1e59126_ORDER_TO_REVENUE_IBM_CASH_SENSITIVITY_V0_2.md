# ASML — Order-to-revenue evidence, Installed Base Management and cash-quality sensitivity
## Research working paper v0.2 — run a1e59126 — cutoff 2026-10-30

**Authority:** `NON_AUTHORITATIVE_RESEARCH_WORKING_ONLY`; **NO** finalized moat, runway, business-quality verdict, DCF, fair value, target, expected return or investment recommendation. No frozen-methodology exception, no Registry admission, no stage advance, no publication.

**Issuer:** ASML Holding N.V. / NL0010273215 / XAMS in EUR.  
**Financial as-of:** Q3/9M 2026 ended **2026-09-27**; FY2025 ended **2025-12-31**.  
**Information as-of:** issuer Q3 release **2026-10-14**, prior issuer releases **2026-01-28**, **04-15**, **07-15**, and 2026-10-14 investor call. All sources <= cutoff.  
**Price comparison:** last fully completed session at research time **2026-10-29**; **EUR 1,504.80** independently corroborated by XAMS Trading-at-Last prints + historical vendors. Original official EOD file not sealed. Ordinary-share denominator **382.91 million vendor placeholder**; not certified G001 count.

## A. Evidence map, with authority and non-equivalence

| ID | Evidence available by cutoff | Source / release date | What it does NOT show |
|---|---|---|---|
| O01 | FY2025 **net bookings EUR 28.035bn**, 2025 Q4 **EUR 13.158bn** (EUV EUR 7.4bn), FY2025 ending **system backlog EUR 38.797bn** | [ASML issuer FY2025 results, 2026-01-28](https://investor.asml.com/news-releases/news-release-details/q4-2025-financial-results) | No Q3 2026 bookings or backlog snapshot; not FY2026 revenue |
| O02 | T1 2026 issuer release gives sales, units, profitability and guide; **not quarterly bookings/backlog** | [ASML Q1, 2026-04-15](https://www.investor.asml.com/news-releases/news-release-details/q1-2026-financial-results) | No quantitative T1 orders |
| O03 | T2 2026 release: stronger demand and commitments, planned 2027 capacity expansion; **not quarterly bookings/backlog** | [ASML Q2, 2026-07-15](https://investor.asml.com/news-releases/news-release-details/q2-2026-financial-results) | No quantitative T2 orders |
| O04 | T3 2026 release: order momentum increasing, 2027/2028 customer demand appears above planned supply; customers **requesting** POs for 2029/2030; **no numeric quarterly bookings/backlog** | [ASML Q3, 2026-10-14](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026) | Requests to submit orders ≠ purchase orders accepted; supply ≠ recognized revenue |
| O05 | CFO explains accepted POs generally involve meaningful down payments and costly cancellation; noncurrent liability movements include down payments and **lag signed PO** | [ASML Q3 investor call, 2026-10-14, third-party transcript](https://stockanalysis.com/quote/ams/ASML/transcripts/651248-q3-2026-investor-call/) | Noncurrent balance growth ≠ precise bookings, deposits cash received or order backlog. Transcript not privately sealed as original source |
| O06 | 12-page **US GAAP Q3** PDF, SHA-256 `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9`, matches independently user-provided Euronext-labeled copy and SHADOW receipt | [Euronext dissemination Q3 2026 financial statements](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/10/14/cpr01_notified_Financial%20statements%20US%20GAAP%20Q3%202026.pdf) | Unauthenticated server download chain and no Registry source admission |
| O07 | Detailed Q3 product capacity and IBM outlook: 2026 low-NA EUV output approximately 65; plans +30% 2027 in relevant lines; Q4 IBM sales approximately EUR 3.7bn | [Q3 ASML IR presentation, SEC filing exhibit](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/a2026_10x14xpresentation.htm) | Capacity planning is not accepted orders, nor certainty of shipment/revenue |

### Interpretation of disclosed order metrics

In the last **quantified** issuer results statement (FY2025), net bookings include written-authorized **system sales orders and inflation-related adjustments**; order backlog includes accumulated system sales orders with written authorization not yet recognized as system sales. The figures **do not include every future service sale**. FY2025 system revenue **EUR 24.474bn** = FY2025 total sales EUR 32.667bn − IBM EUR 8.193bn. The **38.797 / 24.474 = 1.585×** historical system-revenue comparison is a dated ratio, **NOT** a two-year backlog coverage forecast, booking conversion ratio or proof 2027 sales.

**IMPORTANT DISCONTINUITY:** Q1, Q2, Q3 2026 quarterly earnings releases omit an *actual* numerical net-bookings / system-backlog amount. Treat `Q3_2026_NET_BOOKINGS = UNKNOWN` and `Q3_2026_BACKLOG = UNKNOWN`. In particular, do not derive a pseudo-booking amount from increased noncurrent contract liabilities or from capex at TSMC/Samsung. Treat issuer management's order momentum as **QUALITATIVE_ISSUER_CLAIM**.

### Order-to-cash recognition chain

`customer capex plan → equipment discussions → requests to place POs → accepted written PO with down-payment / contractual cancellation consequences → backlogged undelivered systems → construction / shipment → customer acceptance and US GAAP revenue-recognition conditions → invoicing / release of liabilities / final settlement → recurring or timing-sensitive CFO`.

This is a **logical disclosure framework**, not an assertion that every contract has exactly the same schedule, down-payment or acceptance trigger. Avoid counting advances and sales as two independent economic value streams.

As of Q3 2026:
- Request to place PO for 2029/2030: **not yet accepted**, per management's investor call; do not include in backlog.
- Accepted new POs and exact backlog as of 2026-09-27: **UNKNOWN quantitative**; 2025 historical only.
- Noncurrent *contract liabilities* (US GAAP PDF pp5,9) **EUR 3,570.3m on 2026-06-28 → EUR 9,300.6m on 2026-09-27**, delta **+5,730.3m**. Management explicitly says the noncurrent liability category includes down payments and that payment comes after the PO. This is a *lagging indicator*, **not bookings, not all current plus noncurrent customer advances, not a period cash receipt or automatic CFO delta**.
- Quarterly cash-flow line **"Changes in assets and liabilities" +EUR 7,009.0m** for Q3 (PDF p6) is a separate, multi-component cash reconciliation; **never add it to the noncurrent liability balance change**.

## B. IBM / service bridge and projected FY2026 (separate actuals from guidance)

### B1. Reported revenue, direct evidence (EUR millions)

| Period | Total net sales | Installed Base Management (IBM) | Systems = Total − IBM | IBM mix | Source |
|---|---:|---:|---:|---:|---|
| FY2024 | 28,263 | 6,494 | 21,769 | 22.98% | 2025 issuer annual result |
| FY2025 | 32,667 | 8,193 | 24,474 | 25.08% | 2025 issuer annual result |
| Q1 2026 | 8,766.9 | 2,487.5 | 6,279.4 | 28.37% | Q3 GAAP PDF p7 |
| Q2 2026 | 9,326.5 | 2,761.7 | 6,564.8 | 29.61% | Q3 GAAP PDF p7 |
| Q3 2026 | 11,662.0 | 3,288.8 | 8,373.2 | 28.20% | Q3 GAAP PDF pp3,7 |
| 9M 2026 | 29,755.4 | 8,538.1 | 21,217.3 | 28.69% | Q3 GAAP PDF p3 |

**IBM label = "net service and field option sales" in issuer reporting. It can include discretionary productivity upgrade projects; it is NOT synonymous with an annuity-like contractual subscription.** Consequently, growing mix `22.98→25.08→28.69%` cannot be called stable recurring annual cash flow without isolating maintenance, high-ticket upgrades, usage and installed fleet.

### B2. Mechanical FY2026 reconstruction — management forecast, NOT actual sales

Q3 2026 management presentation guides FY2026 **EUR 44–45bn**, Q4 revenue **EUR 14.2–15.2bn**, and Q4 IBM sales **around EUR 3.7bn**. Q4 IBM is a **rounded approximation**.

| Measure | Method | Conditional FY2026 |
|---|---|---:|
| IBM FY2026 | Reported 9M IBM 8,538.1 + **guided ~3,700** | **~12,238.1m** |
| IBM growth vs FY2025 | 12,238.1 / 8,193 − 1 | **~+49.37%** |
| FY2026 systems revenue | 44–45bn FY revenue − ~12.2381bn IBM | **~31.7619–32.7619bn** |
| System-revenue growth vs FY2025 | 31.7619–32.7619 / 24.474 − 1 | **~+29.78% to +33.86%** |
| IBM mix of FY2026 | ~12.2381 / 44–45 | **~27.20%–27.81%** |
| Implied Q4 total from FY guide − reported 9M | 44–45bn − 29.7554bn | **~14.2446–15.2446bn** |
| Implied Q4 systems from FY-guide bridge | Q4 implied total − ~3.7bn IBM | **~10.5446–11.5446bn** |

Q4 guidance itself is **EUR 14.2–15.2bn**; the 0.0446bn difference in implied FY bridge is a **normal result of rounded annual/quarterly guide endpoints**, not a discrepancy to fix by changing historical actuals. **Do not backfill IBM FY2026 as realized**.

### B3. Management Q3 call cross-check — revenue product mix and Q4 guide

[ASML's October 14 prepared remarks, reproduced in a secondary transcript](https://stockanalysis.com/quote/ams/ASML/transcripts/651248-q3-2026-investor-call/) (primary *speaker* CFO Roger Dassen, secondary *text source*) provide independent **narrative reconciliation** of the public financial PDF but **not an independent issuer or a sealed official transcript**:

- Q3 net systems **~EUR 8.4bn**, including **~EUR 4.1bn EUV** (one High-NA system) and **~EUR 4.3bn non-EUV**, with systems sold by end-use **54% logic / 46% memory**; rounded sums must not replace PDF's **EUR 8,373.2m** exact system sales.
- **Q3 IBM was ~EUR 0.4bn above management guidance**, primarily due to demand for **EUV/DUV productivity upgrades**; CFO cites high-margin service components as a key gross margin driver. Thus IBM's **EUR 3.289bn** is neither uniformly maintenance service nor proof of recurring contract cash economics.
- CFO explicitly forecast **~50% IBM year-on-year growth FY2026**, **EUV above 45%**, non-EUV systems **~20%**, memory system sales **~75%** and logic system sales **~10%**. Our **~49.37%** IBM arithmetic from Q4 management guidance is therefore directionally consistent; all these forecasts remain **not realized**.
- CFO expects FY2026 China at **~20% of ASML revenue**; do not mislabel this as **realized** 2026 geographic exposure or extrapolate last year's FY2025 China **29.1%** as Q3 2026. For 2027 and beyond a normalized exposure mix remains unverified.
- During Q&A management confirmed 2029–2030 **requested orders are not accepted yet**, because ASML is planning fair allocation among existing customers. This makes the separation between a request and a contracted backlog **particularly important**.

## C. Quality of cash generation — separate 3 measures

**Input (Q4 2025 + 2026 9M US GAAP, physical financial statement pp6,10; EUR millions):**
- LTM CFO **22,725.3**.
- LTM cash PP&E purchases **1,563.0** and cash intangible purchases **162.9**.
- LTM simple FCF = **20,999.4**.
- Separately disclosed LTM **Changes in assets and liabilities** = **+7,876.9** (Q4 2025 +7,640.7 and 2026 9M +236.2).
- LTM FCF *excluding that single net cash-flow line* = **13,122.5**. This is a **stress lens only**; subtracting ALL changes in assets and liabilities is not an economic-normalized Owner Earnings method.

**Why current CFO might mislead:** signed orders can yield large down payments before high-priced tools are built and delivered; cash receipts do not imply the economic obligation to fulfill the sale disappeared. Likewise, a depressed CFO period can reflect refunds, deposits and working capital timing. The company may profit structurally from customer financing, so **do not presume all +7.877bn is nonrecurring**, either.

### C1. Retention-of-observed-net-WC-line sensitivity (purely mechanical)

At an **illustrative** equity value **EUR 576.203bn**, using `2026-10-29 Euronext Amsterdam EUR 1,504.80` × **382.91m vendor-estimated shares** (neither exact share count nor official EOD primary bytes certified), define
`CANDIDATE_CASH_FLOW = 13,122.5m + r × 7,876.9m`
with **r = 0%, 25%, 50%, 75%, 100% of the observed positive net-assets/liabilities cash-flow line**.

| r (hypothetical fraction retained) | Candidate LTM cash flow, EUR bn | Per vendor share, EUR | Price / candidate cash | Candidate yield |
|---:|---:|---:|---:|---:|
| 0% | 13.1225 | 34.27 | 43.91× | 2.277% |
| 25% | 15.0917 | 39.41 | 38.18× | 2.619% |
| 50% | 17.0610 | 44.56 | 33.77× | 2.961% |
| 75% | 19.0302 | 49.70 | 30.28× | 3.303% |
| 100% | 20.9994 | 54.84 | 27.44× | 3.644% |

**These are NOT valuation price targets, DCF, probabilities, 2027 cash forecasts, ROIC or certified Owner Earnings.** 0% and 100% are not empirically verified bounds; none of the intermediate percentages has been fitted or justified as expected economics. The objective is to show how much the **reported working-capital net movement** matters, relative to smaller denominator uncertainty G001.

### C2. Forensic questions

1. Obtain 2025 US GAAP audited current/noncurrent liability and cash-flow notes and full 2026 interim current/noncurrent customer-contract note; reconcile down-payment balance movement vs actual customer deposits and cash conversion.
2. Separate customer-funded advance cash flow from durable product EBIT margin and from irrevocable cash earnings; reconcile timing of recognition/acceptance and service field upgrades.
3. Include cash tax, maintenance-vs-growth capex, share-based compensation dilution and nonrecurring gain on equity investments (Q3 gain **EUR 509.7m**, indirect cash-flow adjustment **EUR -508.8m**; **no double subtraction**).
4. Distinguish commercial services renewals, consumables, usage-based support, equipment field upgrades. Without service subcategory cash margin and customer utilization, **do not declare IBM recession-proof**.
5. Distinguish order acceptances from 'request to place orders' into 2029/2030; verify export license and customer CAPEX cancellation triggers.

## D. Model-ready scenario organization, without final investment verdict

| Working case | Commercial driver | Stress to implement | Invalid conclusion to avoid |
|---|---|---|---|
| Capacity-constrained demand | 2027–28 planned capacity below expressed client demand | Verify supplier optics, new fabs and ASP/volume independent of AI narrative | Revenue guaranteed from unaccepted 2029/30 PO requests |
| Installed-base monetization | Q4 IBM ~3.7bn management guide | Distinguish upgrade waves from annual maintenance and recurring service mix | Assume annualized T3 IBM margin and 100% renewals |
| Down-payment reversal | High 2025Q4 and 2026Q3 cash inflows | Cash-to-revenue lag, contract liabilities, working capital reversals | Treat EUR 20.999bn as perpetual owner earnings |
| Cyclical capex downturn | Customer 2027 CAPEX revised downward or demand pull-forward | System ASP/mix, lead times and customer utilization | Equate last 2025 backlog with 2027 secure earnings |

### E. Gate/lineage

`G001=OPEN`, `G006=OPEN`, `Q3_2026_NET_BOOKINGS=NOT_DISCLOSED`, `Q3_2026_BACKLOG=NOT_DISCLOSED`, `IBM_FY2026=MANAGEMENT_GUIDANCE_DERIVATIVE`, `OWNER_EARNINGS=NOT_CERTIFIED`, `READY_FOR_DEEP_DIVE=NOT_EVALUATED`. Research may independently advance other blocks. No conversion of these working calculations to an officially admitted artifact, no terminal business-quality judgment.

**Next high-value evidence:** isolate IBM service/upgrade revenue split, 2025 annual and 2026 interim liabilities granular notes, order deposits and booking policy as management describes it, EUV equipment capacity physical constraint, and independent customers' order timing. Do not reopen the precise G001 share count as prerequisite to this line of investigation.
