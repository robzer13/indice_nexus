# OroTitan — ASML G001/G006 — Materiality-based resolution PROPOSAL v0.4

**RUN:** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` — `ACTIVE/RESEARCH`  
**DATA_CUTOFF:** 2026-10-30 (Friday, XAMS session unfinished when this review began)  
**Status:** `PROPOSED_PILOTAGE_RESOLUTION`, NOT accepted methodology, NOT a waiver of frozen contracts, NOT authorized Registry artifact/stage/GO PUBLISH.  
**Frozen contract read:** `contracts/orotitan-equity/v2/OROTITAN_RESEARCH_STAGE_CONTRACT_V2_FREEZE_V2.0.md`, §5 and §9. Research requires *evidence adequacy and no material blocking gap*; it does NOT state that every share count must be exactly determined to the last share. Completion still requires all normal artifacts, self-audit, source bytes, Registry registration, and `READY_FOR_DEEP_DIVE=YES`. **Do not silently reinterpret the deployed G001 gate**.

## The failure mode

Prior investigations repeatedly search for an exact 2026-10-30 issued-minus-treasury share number although the company releases exact official numbers only for earlier dates. Its H1 2026 share counts are rounded and disclosed repurchases are gross flows, not a complete legal/economic share ledger. Repeating the search consumes resources without reliably closing the information gap.

Separate two questions:
1. **Can we perform a useful sensitivity and provisional investment analysis now?** Yes, in **RESEARCH_WORKING_ONLY**, with explicit scenarios, documented as-of dates and authority labels.
2. **Can we assert an officially CERTIFIED October 30 economic share count, market cap, P/E or ROIC in the frozen Registry?** No, absent adequate original sources and Pilotage/contract authority.

## 1. Official evidence and newly discovered corporate action

- [ASML official AGM](https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026): **388,147,674 issued − 3,224,029 treasury = 384,923,645 outside treasury**, exact, **2026-03-25 only**.
- [AFM Netherlands historical capital notification, dated 2025-09-24](https://www.afm.nl/nl-nl/sector/registers/meldingenregisters/geplaatst-kapitaal/details?id=180975) records **388,147,674 legally issued**, but the "last updated" date on the web page is metadata, not a **2026-10-30 capital snapshot**. AFM issued capital/votes and issuer-held treasury are different measures.
- **NEW DISCOVERY:** [Euronext XAMS notices for ASML](https://live.euronext.com/en/ajax/getNoticePublicData/NL0010273215-XAMS?alias=1&pageNum=1&pageSize=50) lists **notice AMS_20261005_00873_EUR**, issued **2026-10-05**, effective **2026-10-08**, type `CE – Shares – Capital increase`. Exact quantity and legal/settlement impact **NOT YET OBTAINED**. This event invalidates a simplistic assumption that total issued shares remained constant since March. Retrieve the original notice PDF / quantity, reconcile with AFM, then treasury stock.
- Issuer's 2025 20-F documents employee stock plans drawing shares **from treasury**. These transfers raise shares outside treasury even when legal issued share count is unchanged.
- Under the already documented issuer weekly notices, gross repurchases 2026-06-29–2026-10-23 = **3,906,385**; no exhaustive Oct 26–30 notice public yet. Gross repurchases alone cannot yield exact economic shares.

## 2. Deterministic materiality triage (scenario, not confidence interval)

**Instrument / price:** ASML ordinary `NL0010273215`, venue **XAMS**, EUR **1,504.80**, fully completed **2026-10-29** session, from [FT Amsterdam EOD](https://markets.ft.markitdigital.com/data/equities/tearsheet/historical?s=ASML%3AAEX) plus [official Euronext XAMS Trading at Last executions](https://live.euronext.com/en/popout-page/getIntradayPrice/NL0010273215-XAMS). This is **not** a sealed Euronext unadjusted EOD byte record and **not** the Oct 30 close.

| Non-authoritative hypothetical economic share count | Indicative equity cap, EUR bn | Price / LTM simple FCF of EUR 20.9994bn | Price / LTM FCF *excluding one net WC line* of EUR 13.1225bn |
|---:|---:|---:|---:|
| 382 462 183 | 575.529 | 27.41× | 43.86× |
| 382 910 000 | 576.203 | 27.44× | 43.91× |
| 384 500 000 | 578.596 | 27.55× | 44.09× |

- Spread of these vendor scenarios **0.532% of the 382.91m reference share count**, or **EUR 3.067bn** theoretical market cap. **These are vendor points, NOT a guaranteed true-share bound**.
- The *same fixed share assumption* changes the simple FCF valuation multiple from **27.44×** to **43.91×** solely by choosing an unnormalized BFR adjustment. This cash-flow normalization is a far larger decision uncertainty than the inter-vendor share count disagreement.
- For robust underwriting, also stress the share count **±2%** around the central vendor proxy and observe whether the final ranking / MOS / investment decision changes. This stress is *hypothetical*, not an empirically proven interval.
- **Materiality is a two-level question**: (a) does the uncertainty change the per-share valuation or investment action by more than an approved materiality threshold? (b) can we prove the actual uncertainty lies within that range? For (b), not yet; hence no certified source count.
- **Do not substitute EPS weighted-average (383.1m) for spot shares**, assume AFM vote totals exclude treasury, or call `382,910,000` an official figure.

## 3. Concrete solution: two-track workflow (no frozen-contract bypass)

### TRACK A — Analytical continuation now; does NOT set `READY_FOR_DEEP_DIVE`

Build all Research analyses that can be completed with evidenced fundamentals: moat, runway, customer CAPEX, EUV/DUV orders, cash reconciliation, risk, management, 2025 US GAAP annual, and sector outside view. For financial sensitivity **only**, label:
- `SHARE_COUNT_INPUT_STATUS = PROVISIONAL_VENDOR_SCENARIO`
- `PRICE_INPUT_STATUS = 2026-10-29_XAMS_VENDOR_CLOSE_CROSS_CHECKED`
- `MARKET_CAP_STATUS = ILLUSTRATIVE_NOT_CERTIFIED`
- `VALUE_PER_SHARE_STATUS = SENSITIVITY_ONLY`
- `NO_PUBLICATION`
- `G001 = UNRESOLVED_EXACT_COUNT`

Do not call this a completed Fundamentals or Deep Dive stage, nor put it into a canonical product investment verdict. Conduct non-dependent research within the authorized Research stage, or in non-authoritative working papers.

### TRACK B — Independent authoritative resolution, bounded and testable

1. **New priority #1:** retrieve original **Euronext `AMS_20261005_00873_EUR` notice (2026-10-05)** and exact affected issued share count / effective legal date. Preserve original bytes, publication date, SHA-256 and impacted registered share quantity.
2. Retrieve latest pre-cutoff **AFM issued capital notifications by the *obligation/filing date***, not a website's latest-update timestamp; distinguish legal capital from voting rights and treasury.
3. Seek issuer exact treasury ledger, share cancellation/ESPP transfers, and settlement convention through [ASML Corporate Secretariat](https://www.asml.com/en/company/governance/shareholders/shareholder-meetings/agm-2026) (official address `corporate.secretariat@asml.com`). A request alone is not evidence. Any issuer response published **after** 2026-10-30 cannot silently become a prior-cutoff source.
4. Obtain an official Euronext XAMS **2026-10-29 unadjusted EOD** record, with source bytes/date/field, not BGEM or NASDAQ.
5. Submit a **G001/G006 MATERIALITY RESOLUTION** to Pilotage: allow normal Research source collection despite incomplete exact market cap, with `INSUFFICIENT` for the affected valuation input until source adequacy is demonstrated. If Pilotage determines the missing source is a **material** blocking gap under frozen contract, follow `PAUSED`/resolution protocol; do not auto-pass the stage. If Pilotage decides it is *not material*, record why with impact test and reviewer approval. No unilateral change to frozen V2 contracts or production code.

## 4. Time-box and stopping rule

The **next** G001 effort targets **one specific source**, the Euronext capital-increase notice, not another broad internet search. After obtaining the notice or recording proven inaccessibility, adjudicate the residual exact-share gap at Pilotage rather than repeating vendor searches. Value with scenario labels while the authority route proceeds; avoid unlimited exploratory loops.

**Exit result of this document:** `MATERIALITY_TRIAGE_PROPOSED`, `SOURCE_PRIORITY_CHANGED_TO_OCT_05_CAPITAL_INCREASE_NOTICE`, `G001_CERTIFICATION=BLOCKED`, `G006_CERTIFICATION=BLOCKED`, `RESEARCH_COMPLETION=NOT_AUTHORIZED`, `REGISTRY_WRITES=ZERO`.
