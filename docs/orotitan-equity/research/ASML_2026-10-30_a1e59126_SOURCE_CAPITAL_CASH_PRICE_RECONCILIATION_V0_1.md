# ASML / RUN a1e59126 — Source qualification, G001 capital, cash quality and valuation inputs
## Cutoff 2026-10-30 · PRE-ADMISSION RESEARCH ONLY · v0.1

**Issuer/security:** ASML Holding N.V. / NL0010273215 / ASML / Euronext Amsterdam XAMS / EUR.  
**RUN_ID:** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24`.  
**Snapshot:** ANALYZE / INITIAL / data_cutoff 2026-10-30 / Process V3.1 / frozen Research V2.0.  
**Authority level:** `RESEARCH_CANDIDATE`; **NOT an authoritative Evidence Ledger, not a Registry artifact, not a finalized Stage Manifest, no stage completion, no publication.**  
**Evidence boundary:** the financial report is the 2026-09-27 *accounting as-of* period, public 2026-10-14; the valuation cutoff is 2026-10-30. No post-cutoff sources.

### A. PDF qualification: S-2026Q3-US-GAAP — BINARY MATCH; admission pending

1. The [ASML official Q3 result downloads page](https://www.investor.asml.com/quarterly-results) specifies **Financial statements US GAAP PDF** in the Q3 2026 results package dated **2026-10-14**; issuer [press release](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026) independently contextualizes the publication.
2. The administrator uploaded a 12-page PDF into **private Supabase SHADOW storage** (bucket `orotitan-source-files-v1`) on **2026-10-30T13:02:08.032176Z**, source path `research-acquisition-v1/asml/asof-2026-10-30/q3-2026/browser-supplied/562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9.pdf`, with an immutable JSON receipt; application reported create-only readback SHA-256 success.
3. Separately provided, [Euronext-distributed PDF](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/10/14/cpr01_notified_Financial%20statements%20US%20GAAP%20Q3%202026.pdf) **user-supplied file** `cpr01_notified_Financial statements US GAAP Q3 2026.pdf` was SHA-256 hashed locally and inspected. **Exactly 1,369,295 bytes**; SHA-256 `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9`, matching the private SHADOW receipt. PDF 1.4, 12 pages, title "Financial statements US GAAP Q3 2026", creator Workiva, producer Wdesk Fidelity Content Translations, creation 2026-10-13T11:07:44Z, unencrypted.
4. PDF data checks by page: p3 operations; p4 ratios; p5 consolidated balance sheet; p6 cash flow; pp7–10 historical quarterly comparisons; p11 unaudited US GAAP basis; p12 forward-looking statements.
5. **Reconciliation checks:** net system sales 8,373.2 + service 3,288.8 = 11,662.0 (EURm); 11,662.0 − cost of sales 4,962.2 = gross profit 6,699.8; 6,699.8 − R&D 1,174.4 − SG&A 352.6 = operating profit 5,172.8. All differences **0.0** within displayed precision.

**Decision:** `CONTENT_IDENTITY_TO_EURONEXT_LABELED_SECOND_COPY = MATCH`;
`SOURCE_PUBLICATION_CONTEXT = DOCUMENTED`;
`SHADOW_CAPTURE_READBACK = APPLICATION_REPORTED_VERIFIED`;
`CRYPTOGRAPHIC_EURONEXT_ORIGIN_ATTESTATION = NOT_OBTAINED`;
`RAW_PRODUCTION_SOURCE_ADMISSION = NOT_DONE`.
The two user-provided/administrator-submitted copies have an identical SHA; neither creates a cryptographic attestation of the external download chain. This does not prove that the full Registry source-admission contract is met. No fake source hash or provenance promotion.

### B. G001 — economic shares (issued minus treasury): INCOMPLETE

| Date | Observable quantity | Shares | Source / treatment |
|---|---|---:|---|
| 2025-12-31 | Outside treasury (per footnote) | 385,417,665 | [ASML share information](https://www.asml.com/en/investors/shares); apparent **reversal of labels** in main table, preserve conflict. |
| 2025-12-31 | Treasury (per footnote) | 2,730,009 | Same issuer website; arithmetic `385,417,665+2,730,009=388,147,674`. |
| 2026-03-25 | Total issued | **388,147,674** | [2026 AGM record date](https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026), primary exact. |
| 2026-03-25 | Treasury | **3,224,029** | AGM primary exact. |
| 2026-03-25 | Economic / outstanding | **384,923,645** | AGM primary exact; `388,147,674−3,224,029`. **NOT** 2026-10-30 count. |
| 2026-09-27 Q3 | Basic weighted-average shares for EPS | **383.1m** | Uploaded ASML US GAAP p3 and p7; rounded **period average, NOT spot share count**. |
| 2026-10-29 | Shares-outstanding vendor estimate | **382.91m** | [StockAnalysis Amsterdam statistics](https://stockanalysis.com/quote/ams/ASML/statistics/); vendor methodology/as-of unclear; reference only, **NOT** admitted. |

#### Official 2026 buyback ledger available no later than Oct 30

All lines below are issuer disclosures **already public** at cutoff. They represent **gross purchases**, not the net change in economic shares. Units: ordinary shares.

| Announcement date | Trade-date window | Gross repurchases | Primary issuer URL |
|---|---|---:|---|
| 2026-09-28 | Sep 21–25 | 165,200 | [Issuer notice](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-2) |
| 2026-10-05 | Sep 28–Oct 02 | 277,000 | [Issuer notice](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-3) |
| 2026-10-12 | Oct 05–09 | 280,000 | [Issuer notice](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-4) |
| 2026-10-19 | Oct 12–16 | 233,867 | [Issuer notice](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-5) |
| 2026-10-26 | Oct 19–23 | 343,310 | [Issuer notice](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-6) |
| **TOTAL** | **2026-09-21 to 2026-10-23** | **1,299,377** | Verified sum of 5 weekly primary disclosures |

Prior research documented **2,607,008** additional gross shares bought **2026-06-29 to 2026-09-18**, yielding combined **3,906,385** *gross* purchases 2026-06-29–2026-10-23, conditional on the prior 12-notice ledger and without proving all capital movements. Note this is not a legitimate outstanding-share figure. ASML's [2026–2028 program](https://investor.asml.com/investor-resources/share-buyback) says up to 2 million repurchased shares may serve employee share plans; the remainder are intended for cancellation. A purchase can reduce outstanding shares by increasing treasury but a **cancellation removes issued and treasury in parallel**; employee share deliveries/issuances may reverse the effect.

**Blocking items:** exact economic shares at 2026-09-27 (2026 Q3 shows only rounded weighted average); complete 2026 share issue/cancellation and employee-delivery register; unsettled trades and settlement-date convention; no 2026-10-26–30 buyback notice publicly dated by the cutoff (weekly reporting lag). Vendor counts `382.91m`, `383.10m` (EPS weighted avg), and `385.42m` (some outstanding-share aggregators) are **not interchangeable**. `G001=BLOCKED_EXACT_COUNT`.

**Methodology amendment (2026-10-30):** the initial narrative did not distinguish the US GAAP balance sheet's noncurrent contract liabilities from total contract liabilities. This is now corrected; a numerical +5,730.3m noncurrent movement is not all customer cash collections. The [extended quality/ROIC sensitivity v0.2](./ASML_2026-10-30_a1e59126_OWNER_EARNINGS_ROIC_VALUATION_SENSITIVITY_V0_2.md) documents the correction and reproducible alternatives. 

### C. Cash-flow reconciliation (unaudited US GAAP)

All amounts **EUR millions**, directly from the PDF, p6 (Q3 / nine months) and p10 (quarterly bridge). Positive capex shown here as absolute cash outflows.

| Measure | Q1 2026 | Q2 2026 | Q3 2026 | 9M 2026 |
|---|---:|---:|---:|---:|
| Operating CFO | **−2,185.6** | **+1,703.0** | **+11,797.6** | **+11,315.0** |
| PP&E purchased | 402.4 | 299.4 | 413.3 | 1,115.1 |
| Intangibles purchased | 19.9 | 86.6 | 33.9 | 140.4 |
| Simplified CFO − PP&E − intangibles | **−2,607.9** | **+1,317.0** | **+11,350.4** | **+10,059.5** |

Arithmetical invariants: `−2185.6+1703.0+11797.6=11315.0`, and `−2607.9+1317.0+11350.4=10059.5`.

Working capital details, Q3 2026:
- `Changes in assets and liabilities = +7,009.0`, or **59.41%** of Q3 CFO 11,797.6. This aggregate movement is NOT all customer prepayments.
- **NONCURRENT** contract liabilities rise **from 3,570.3 at 2026-06-28 to 9,300.6 at 2026-09-27**, change **+5,730.3** (US GAAP balance sheet pp5/9). These figures are **not total current-plus-noncurrent customer liabilities, not actual deposits received, and not the CFO contract-liability bridge**. The summary's current liabilities are aggregated. Interpret only as this NONCURRENT balance line.
- Accounts receivable rise from **7,252.6 to 9,252.8**, change **+2,000.2**; inventories rise from **11,739.5 to 11,907.3**, change **+167.8**. These movements generally offset cash from advances and require cash/noncash and foreign-exchange adjustments.
- Q3 includes **509.7** gain on equity investments in income and **−508.8** related adjustment in the indirect cash flow; these are not a recurring lithography operating result.
- Share-based compensation in Q3 **73.2** and in 9M **162.8** is a noncash cash-flow add-back; potential equity dilution must be treated in return quality.

**Supplementary LTM arithmetic only:** 2025 Q4 CFO 11,410.3 + 2026 9M CFO 11,315.0 = **22,725.3**. With 2025 Q4 PP&E 447.9, intangibles 22.5 and 2026 9M cash capex 1,255.5, LTM **simplified FCF = 20,999.4**. Note 2025 Q4 also had `+7,640.7` movements in operating assets/liabilities: a pronounced cash timing pattern, not proof of sustainable Owner Earnings.

**Classification:** these are **REPORTED_CFO** and **ARITHMETIC_SIMPLIFIED_FCF**, not Owner Earnings and not normalized free cash flow. Normalize customer advances/contracts, tax/working capital, maintenance vs growth capex, nonoperating investments, SBC and working capital changes before valuation.

### D. G006 — valuation price / time discipline

- **Instrument/venue:** ASML, ISIN NL0010273215, **Euronext Amsterdam XAMS**, EUR/share. Do not use BGEM, US ADR USD or secondary Frankfurt quotations.
- **Most recent fully completed XAMS trading day when reviewed during the Friday 2026-10-30 session:** 2026-10-29.
- Three secondary vendor datasets independently list **unadjusted/displayed close 2026-10-29: EUR 1,504.80**, traded **660,425 shares**:
  [FT Markets ASML:AEX](https://markets.ft.markitdigital.com/data/equities/tearsheet/historical?s=ASML%3AAEX),
  [StockAnalysis Amsterdam daily](https://stockanalysis.com/quote/ams/ASML/history/),
  [Investing.com Amsterdam daily](https://www.investing.com/equities/asml-holding-historical-data).
- **2026-10-30** vendor prices are **intraday**, vary during the session and must not be labeled the 2026-10-30 closing price.
- The issuer announced an interim dividend **EUR 1.88** with **Euronext ex-date 2026-10-26** ([ASML dividend schedule](https://investor.asml.com/investor-resources/capital-return-and-financing)); do not silently replace unadjusted with dividend-adjusted close.
- [Euronext XAMS quote identity](https://live.euronext.com/en/product/equities/nl0010273215-xams) is available but no independently acquired **official XAMS EOD record for 2026-10-29** was sealed. Vendor match does not substitute for primary exchange-close authentication.

`G006 = VALID_VENDOR_CANDIDATE_AS_OF_2026-10-29 / OFFICIAL_XAMS_EOD_NOT_ADMITTED`. A value for 2026-10-30 EOD would be *future information* during the observed session. **Never** multiply an unverified spot share count by a vendor close to produce *certified* market cap.

### E. Illustrative price/share sensitivity — NOT a certified valuation

With hypothetical **382.91m** vendor shares × **EUR 1,504.80** vendor close on 2026-10-29, market capitalization **approximately EUR 576.20bn**, matching the supplier's own estimate. This is a *vendor scenario*, NOT an admitted market cap. A 1.0m share count error moves equity value by **EUR 1.5048bn** at the stated price.

For EV, **cash + short-term investments were 8,030.9 + 4,704.2 = 12,735.1** at 2026-09-27 (PDF p5); **long-term debt was 1,963.0** (not total debt). No certified enterprise value without reconciling interest-bearing current debt, leases, nonoperating investments, minority interest, price timing and exact equity denominator.

For business-model/earnings context: 2026 9M net sales **29,755.4** and management's 2026 FY sales guidance **EUR 44–45bn** announced October 14, as forward-looking estimate, not a fact about realized FY sales.

### F. Control gates and decision register

| Gate | Status | Last missing admissible proof |
|---|---|---|
| Source file identity | `SHA_MATCH_CONFIRMED` | Formal source-origin authentication/custody and ledger admission review; PDF is unaudited |
| Source financial calculation checks | `INTERNAL_ARITHMETIC_PASS` | Frozen Evidence/Calculation Ledger trace and independent approval |
| G001 exact economic shares | `BLOCKED` | Date-specific exact issuer share ledger with treasury/cancellations/employee deliveries/settlement |
| G006 official share price | `BLOCKED` | Time-compatible official Euronext **XAMS** EOD (or separately defined admissible timestamp) |
| Owner Earnings | `BLOCKED` | Deferred revenue/capital needs, maintenance capex, cash normalization and audit |
| ROIC | `BLOCKED` | NOPAT / invested-capital and nonoperating asset reconciliations |
| Research completion | `BLOCKED` | Full required artifacts, gap/conflict/assumption ledgers, custody and stage manifest/self-audit |

**New official venue print found:** [Euronext XAMS intraday archive for ASML, 2026-10-29](https://live.euronext.com/en/popout-page/getIntradayPrice/NL0010273215-XAMS) displays Trading-at-Last prints at EUR 1,504.80 at 17:39:46. This corroborates the vendor figure at the exact venue, **without certifying the official non-adjusted EOD price record**. See v0.2 for full trace.\n\n**Disposition:** Continue `ACTIVE / RESEARCH`; `READY_FOR_DEEP_DIVE = NOT_EVALUATED` until full conditions. No registry change, no artefact admission, no GO PUBLISH, no changes to frozen process/methodology or source-dating.
