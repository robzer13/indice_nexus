# ASML G001 / G006 — source-authority conflict, cutoff 2026-10-30 (v0.3)

**RUN** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` | Process V3.1 | Research V2.0
**Scope** XAMS ASML ordinary NL0010273215, EUR; financial-data cutoff 2026-10-30.
**Authority** RESEARCH_ONLY / NOT_ADMITTED / NO_REGISTRY_WRITE / NO_PUBLISH.
**Source discipline** Underlying timestamp / publication date / access date are distinct. Only sources already public at cutoff may support this run.
**Valuation snapshot convention** Last fully completed local XAMS session during 2026-10-30 research is **2026-10-29**, not an intraday quote labeled `30 October close`.

## 1. G001: exact issued and treasury share ledger — primary evidence and formula

Issuer 2026 AGM [official record date 2026-03-25](https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026):
- 388,147,674 total issued ordinary shares (exact).
- 3,224,029 treasury shares (exact).
- **384,923,645 shares issued and outstanding outside treasury (exact as of March 25 only)**.

Issuer [EU-IFRS statutory interim report of July 15, 2026 (period ended June 28)](https://www.sec.gov/Archives/edgar/data/937966/000162828026048235/statutoryinterimreport20.htm):
- 2026-06-28 issued-and-outstanding line **384.1 million** (rounded to 0.1m).
- H1 2026 purchases of treasury shares about **1.7m** and issuance/release about **0.4m** (both rounded).
- This **does not supply** an exact `issuedMinusTreasury` share balance at June 28. Do not conflate the EU-IFRS accounting balance sheet with US GAAP invested capital.

Issuer [2025 20-F, Employee Share Purchase Plan disclosure](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R31.htm):
- Employees purchase shares through monthly payroll deductions.
- **Shares for employee purchase plans are issued from treasury on a quarterly basis** at a share price set after quarterly filings.
- Thus a treasury drawdown for employee participation **increases economic outstanding shares**, even if issued legal share capital stays unchanged.
- ASML's [2026–2028 buyback policy](https://investor.asml.com/investor-resources/share-buyback) explicitly reserves up to approximately 2.0 million repurchased shares for employee plans and intends to cancel the rest; plans are **intent**, not a measured employee-transfer count.

Gross buybacks already published in 17 official announcements on or before the cutoff:
- 2026-06-29–09-18: **2,607,008** shares (12 notices, separately referenced in [prior review](./ASML_2026-10-30_a1e59126_SOURCE_CAPITAL_CASH_PRICE_RECONCILIATION_V0_1.md)).
- 2026-09-21–10-23: **1,299,377** shares (5 notices, primary issuer [September 28](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-2), [Oct 5](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-3), [Oct 12](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-4), [Oct 19](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-5), [Oct 26](https://investor.asml.com/news-releases/news-release-details/asml-reports-transactions-under-its-current-share-buyback-program-6)).
- **Gross documented June 29–Oct 23 = 3,906,385** shares; this is an input to a bridge, **not the net loss of economic shares since June**.
- Transactions of October 26–30 **cannot be inferred** from ASML's next weekly notice if that notice is not yet public by cutoff.

The correct share bridge, at an exact common trade/settlement convention:
```
Issued_T = Issued_0 + NewLegalIssuedShares - CanceledIssuedShares
Treasury_T = Treasury_0 + SettledBuybacks - EmployeeTreasuryDeliveries
                    - OtherTreasuryDisposals - CanceledTreasuryShares
EconomicShares_T = Issued_T - Treasury_T
```
A cancellation of shares already in treasury **reduces Issued and Treasury by the same number**, changing economic shares by ZERO at that point; the initial buyback had reduced economic outstanding when ownership passed to the issuer.

**Still absent:** exact June baseline issued and treasury counts, 2026-06-29–10-30 transfers/employee settlements/other disposals, new legal issues and cancelled shares, trade/settlement timing, exact Oct 26–30 executed purchases if settlement occurred by cutoff. Exact G001 as-of cutoff remains **UNKNOWN**.

## 2. Market vendors disagree — a live Conflict Ledger entry

| Quantity | Value | Origin | Admissibility |
|---|---:|---|---|
| Exact economic shares on 2026-03-25 | **384,923,645** | ASML AGM official | Exact, **wrong date** |
| H1 issued and outstanding on 2026-06-28 | **384.1m** | ASML statutory report | Rounded, **wrong date** |
| T3 basic EPS denominator | **383.1m** | ASML Q3 2026 GAAP PDF, p3 | Period weighted average, **wrong measure** |
| Oct vendor share count A | **382.91m** | [StockAnalysis Amsterdam](https://stockanalysis.com/quote/ams/ASML/statistics/) | Vendor estimate, method/time unclear |
| Oct vendor share count B | **382,462,183** | [StatMuse ASML NASDAQ market-cap history](https://www.statmuse.com/money/ask/asml-market-cap-in-2026) | Vendor daily historical proxy (ADR linked), method unclear |
| Oct vendor share count C | **384,500,000** | [CompaniesMarketCap](https://companiesmarketcap.com/asml/shares-outstanding/) | Vendor estimate, possible stale/rounding |
| Sep 2026 vendor quarterly count | **385.42m** | [YCharts](https://ycharts.com/companies/ASML/shares_outstanding) | **Stale-looking** constant since 2025-12-31; contradictory to AGM |
| Unchanged legal issued share count assumption | **388,147,674** | ASML Dec 2025 and March 2026 | No Oct confirmation; **not an admissible Oct assumption** |

**Conflict decision:** No averaging, majority vote, arbitrary preference, or picking 383.1m because it appears in the Q3 PDF. Treat each quantity by **measurement type** first (stock vs flow, dated exact vs average, shares issued vs excluding treasury) and then by its source authority. `G001_EXACT_2026_10_30=BLOCKED`.

## 3. G006: official venue proof vs original EOD file

- Venue **Euronext Amsterdam XAMS**; ordinary ISIN NL0010273215; share currency EUR.
- Public [Euronext official XAMS recent-trades page](https://live.euronext.com/en/popout-page/getIntradayPrice/NL0010273215-XAMS) supports historical late `Trading at Last` executions **2026-10-29 17:39:46 at EUR 1,504.80** in the earlier captured research record.
- Independent vendor [Financial Times](https://markets.ft.markitdigital.com/data/equities/tearsheet/historical?s=ASML%3AAEX), [Investing](https://www.investing.com/equities/asml-holding-historical-data), and [Bourse Direct XAMS](https://www.boursedirect.fr/fr/marche/euronext-amsterdam/asml-holding-n-v-NL0010273215-ASML-EUR-XAMS/graphiques) agree on **2026-10-29 EUR 1,504.80 / 660,425 shares**.
- **Official unadjusted 2026-10-29 End-of-Day venue price as a dated immutable Euronext archive NOT separately acquired/sealed**, despite an official late-trade corroboration. Trading at Last != uniquely identified official EOD field.
- ASML [dividend schedule](https://investor.asml.com/investor-resources/capital-return-and-financing): Euronext dividend ex-date **2026-10-26** at EUR 1.88. Keep raw vs adjusted close distinct.
- Public 2026-10-30 live/intraday quotes have substantial downward movement but are **NOT** Oct 30 final close. They cannot be used as a closing price in a 14:30 CET snapshot.

**Decision:** `G006_VENDOR_CLOSE_CONFIRMED`, `XAMS_VENUE_LATE_TRADE_SUPPORTED`, `OFFICIAL_NONADJUSTED_EOD_PRIMARY_BYTES_MISSING`; **BLOCKED for certified final valuation**.

## 4. Quantify vendor share disagreement without assigning false precision

At *illustrative* XAMS 2026-10-29 displayed close EUR 1,504.80:
- Using StatMuse **382,462,183** economic shares => **EUR 575.53bn** hypothetical capitalization.
- Using StockAnalysis **382,910,000** => **EUR 576.20bn** hypothetical capitalization.
- Using CompaniesMarketCap **384,500,000** => **EUR 578.60bn** hypothetical capitalization.
- These capitalizations are only vendor-count comparison diagnostics. The delta between the lowest and highest among these three sources is about **EUR 3.07bn**. No market cap certified by this memo.
- One million economic shares is **EUR 1.5048bn** equity valuation uncertainty at this price.

## 5. Short, specific primary-authority request — NOT SENT

**Destination class:** ASML Investor Relations / Corporate Secretariat, using ASML's official contact channels; no email was sent by OroTitan.

> For an as-of 2026-10-30 valuation, please provide or direct us to a publicly distributable dated table with the **exact (non-rounded)** number of legally issued ordinary shares, treasury shares, and ordinary shares outstanding excluding treasury as at the close of 2026-10-29 (and, if available, 2026-10-30). Please include reconciled settled share buybacks, transfers from treasury to employee share plans, other treasury disposals, new issued shares, legal cancellations and the applicable trade-versus-settlement cutoff. We seek the ordinary XAMS ISIN NL0010273215, not the EPS weighted-average denominator. Please cite the official disclosure and source publication date.

**Minimum alternative**: an authoritative issuer or regulator exact-dated share/capital ledger with a signed or hash-preserved original document and explicit temporal cutoff. If the earliest new disclosure is post-2026-10-30, it **cannot** automatically be admitted to this frozen-cutoff run; use only under an authorized methodology for subsequent evidence or in a new dated run.

## 6. Controls / next decision

1. Prioritize issuer's **exact-dated issued/treasury/outstanding** table; otherwise G001 stays BLOCKED. **Do not** move share-count proxy into a publishable market-cap calculation.
2. For G006, purchase/retrieve an **Euronext XAMS unadjusted EOD export** (29 Oct), preserving bytes, field labels, timezone, source date and SHA-256; no fabricated Euronext quote certification.
3. Continue cash and invested-capital source reconciliation in [ROIC/FCF sensitivity v0.2](./ASML_2026-10-30_a1e59126_OWNER_EARNINGS_ROIC_VALUATION_SENSITIVITY_V0_2.md), with distinct economic normalization vs accounting figures.
4. No new production Registry artifacts, no stage finalization, no silent revision of historical RUNs or publication.
