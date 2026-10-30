# OroTitan / ASML — Business model, competitive advantages, runway and outside-view research

**RUN**: `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` · `ANALYZE/INITIAL` · `ACTIVE/RESEARCH` · Process V3.1 · Research Stage Contract V2.0.  
**DATA_CUTOFF**: 2026-10-30; do not use facts published afterwards.  
**Classification**: `RESEARCH_WORKING_ONLY / NON_ADMITTED / NON_FINAL / NOT_INVESTMENT_THESIS`.  
**No Research-to-Deep Dive gate waiver**, no Registry admission, no canonical result, no ROIC/OE certificate, no publication. No frozen methodology modified.  
**Scope**: evidence normalization and working hypotheses only. The frozen Research contract prohibits *final* moat/runway/business quality verdicts, DCF, investment score, price target, investment thesis and terminal judgment at this stage.

## 1. Source map (origin vs independent corroboration)

| ID | SOURCE_DATE | As-of / coverage | Authority and origin | Verified key purpose / limitations |
|---|---|---|---|---|
| ASML-AR25 | 2026-02 (2025 report, available before cutoff) | FY2025 | [ASML 2025 annual report as filed on SEC](https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm) | Company-origin primary annual report. Business, lithography, supply-chain and client concentration; source web display inspected, original full filing bytes not separately sealed to this RUN |
| ASML-Q3-IR | 2026-10-14 | Q3 2026 and FY2026 guidance | [Issuer Q3 release](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026) | Company-origin primary contemporaneous disclosures, management outlook, **not** independently verified market demand |
| ASML-Q3-SEC99.2 | 2026-10-14 | Q3 product roadmap, expectations | [Issuer investor presentation filed SEC](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/a2026_10x14xpresentation.htm) | Issuer-authored same information family as Q3 IR, not independent corroboration |
| ASML-Q3-GAAP | 2026-10-14 | Q3 ended 2026-09-27 | [Issuer Q3 2026 financial statements PDF distributed by Euronext](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/10/14/cpr01_notified_Financial%20statements%20US%20GAAP%20Q3%202026.pdf) | 12 pages, SHA `562ba8423b4cdb74f48c0d033f9c1571c314f525f252cffce245ad532b03e2a9` matches the separately supplied Euronext-labeled copy and private SHADOW receipt. Source origin and Ledger admission still not certified |
| TSMC-Q3-26 | 2026-10-15 | Q3 2026 | [TSMC Q3 issuer earnings](https://pr.tsmc.com/english/news-archives?field_category_target_id=1&field_issue_date%5Bmax%5D=2026-12-31&field_issue_date%5Bmin%5D=2026-01-01&queryYear=2026&title=) and [Q3 Management Report](https://investor.tsmc.com/english/quarterly-results/2026/q3) | Distinct reporting company, independent as a customer-side market demand indicator; not evidence of specific ASML orders or chipmaker equipment split |
| TSMC-CALL | 2026-10-15 | 2026 capital budget | [TSMC earnings call transcript, third-party reproduction](https://www.fool.com/earnings/call-transcripts/2026/10/15/tsmc-tsm-q3-2026-earnings-call-transcript/) | Customer management reiterated **2026 CapEx $60–64bn**; transcript not privately archived and still forecast |
| SAMSUNG-Q3-26 | 2026-10-29 | Q3 and FY2026 capex outlook | [Samsung official newsroom](https://news.samsung.com/global/samsung-electronics-announces-third-quarter-2026-results) | Distinct customer-side issuer primary; DS CapEx expected FY2026 **KRW 76.8tn** (of group total KRW 84.4tn). Does not state ASML-specific orders |
| ASML-TSMC-HIGHNA | 2026-09-08 | High-NA roadmap | [ASML/TSMC official joint release](https://www.investor.asml.com/news-releases/news-release-details/asml-and-tsmc-announce-initiative-pioneer-industry-transition-large-format-photomasks-high-na-euv) | Intent to use High-NA for TSMC advanced nodes from 2030; 12-inch mask pilot targeted 2031 / advanced node production 2033, forecasts not realized facts |
| ASML-INTEL-HIGHNA | 2026-09-08 | High-NA use | [Intel/ASML joint release](https://www.investor.asml.com/news-releases/news-release-details/intel-foundry-and-asml-collaborate-accelerate-industry-readiness-high-na-euv) | More than one million wafers processed during combined test/R&D/select production activities; **not** one million full-volume production wafers |
| ASML-SAMSUNG-HIGHNA | 2026-09-08 | Future DRAM adoption | [Samsung/ASML official announcement on Samsung site](https://semiconductor.samsung.com/news-events/news/samsung-electronics-and-asml-expand-strategic-collaboration-for-next-generation-semiconductor-manufacturing/) | Planned High-NA adoption for future DRAM; not a firm equipment delivery schedule |

**Temporal:** no 2026-10-30 official close in these sources. Preserve 2026-10-29 price source independently from the 2026-09-27 accounting figures. No admission of post-cutoff facts.

## 2. Business-model architecture — descriptive, not a final business-quality verdict

**Economic job to be done:** lithography tools pattern semiconductor wafers. Advanced-node logic and memory manufacturing require multiple lithography exposures and overlay precision to maintain yield. Customer economics depend on resolution, system throughput, uptime, upgradeability, process integration and total cost per wafer, not merely scanner selling price.

**Distinct revenue mechanisms:**
1. **EUV systems:** advanced leading-edge logic/memory layers. ASML reports being the **only manufacturer of EUV lithography systems** in its 2025 report. High optical physics, EUV source, precision mechanics, software/process control, ecosystem compatibility.
2. **DUV systems:** more pervasive across mature and advanced-node noncritical layers; less unique than EUV and technologically substitutable in some applications. Competes on productivity, yield, uptime and cost of ownership; do **not** extrapolate EUV monopoly to all DUV.
3. **Installed Base Management (IBM):** hardware servicing, field options, productivity upgrades for already installed scanners. Supports recurring-like cash generation **but is NOT proven contractually recurring or immune to utilization/capex cycles**.
4. **Metrology, inspection, computational lithography and process-control software:** complement scanners to improve yield and process window; separate component economics and recurring rates are not fully disclosed here.
5. **High-NA EUV and integration/packaging:** product extensions with significant engineering risk, development expense, partner dependencies and uncertain adoption velocity.

**Installed base persistence:** ASML claims approximately **95% of systems sold in the last 30 years remain in use** (2025 annual report). This supports service opportunity and longevity, but is a management-derived operational metric, not an audited retention cohort or contractual renewal rate.

### 2.1 Quantified source bridge

| Key metric | FY2024 | FY2025 | 2026 Q3 | 2026 9M / guidance | Caveat |
|---|---:|---:|---:|---:|---|
| Total ASML revenue EUR bn | 28.263 | **32.667** | **11.662** | **29.7554** 9M; **44–45 FY guide** | 2026 FY guide is **forecast**, not realized |
| Installed Base Management EUR bn | 6.494 | **8.193** | **3.2888** | **8.5381** 9M | Service/field options, not audited recurring subscription |
| IBM % of sales | **23.0%** | **25.1%** | **28.2%** | **28.7%** 9M | Computed from separately reported source totals |
| Gross margin | [not required here] | **52.8%** | **57.5%** | **56% FY guide** | Guidance subject to realization |
| 2025 EUV system sales | — | **11.6bn** | — | — | ASML January 2026 2025 results: 48 systems recognized |
| 2025 DUV system sales | — | **12.0bn** | — | — | 279 systems recognized (2025) |

Calculated, **FY2026 guided revenue growth versus FY2025 is 34.7% to 37.8%**, conditional on the October 14 management forecast; no final FY2026 growth may be asserted. The FY2026 guide 44–45bn overlaps/ exceeds the **low end** of the old 2024 Investor Day 2030 range (44–60bn); therefore the old 2030 range is **not an updated terminal-year forecast**. Management says new longer-term views are planned for **June 10, 2027**; do not interpolate a new CAGR or DCF from an obsolete 2030 range.

## 3. Competitive-advantage evidence questions — NO FINAL MOAT VERDICT

| Mechanism hypothesis | Evidence supporting viability | Counter-thesis / required falsification | Current assessment |
|---|---|---|---|
| Technical barriers to EUV entry | ASML is sole EUV manufacturer; system spans source, optics, stages, metrology, control; R&D **EUR 4.7bn in 2025** | Alternative patterning, export-induced domestic innovation, changes in lithography intensity, High-NA non-adoption; DUV competitive dynamics differ | `HYPOTHESIS_SUPPORTED_NOT_SCORED` |
| Switching/integration cost | Tool process recipes, fab qualification, overlay/yield tuning, field upgrade paths and legacy installed base | Customer multisourcing on DUV, capex cycle, lower utilization, upgrades can defer new system sales | `TO_BE_TESTED` |
| Ecosystem/learning curve | Zeiss optics, long supplier/co-development network, source + software + fab process optimization | **Single-source ZEISS SMT** optics can limit production; joint dependence increases disruption severity | `DUAL_EDGED` |
| Economic returns | High gross margin and operating margin Q3, plus >3.2bn service sales | Quarterly cycle, working capital distortion, current vs noncurrent advances, maintenance-vs-growth capital, cash normalization | `ROIC_OE_UNCERTIFIED` |

**Concentration and geopolitical risk:** ASML 2025 annual report discloses that China customers represented **29.1% of FY2025 revenue**, vs **36.1% FY2024**. Do not transpose either share to Q3 2026 without new source. Export restrictions may affect shipment mix, installed-base support and China demand. Customer concentration in relatively few chipmakers adds correlated capex risk.

**Capacity friction:** 2025 annual report identifies **Carl Zeiss SMT as the sole optics supplier** under an exclusive arrangement; a prolonged interruption could materially impair ASML's ability to conduct business. A plan to raise capacity by 30% is not itself proof that supplier throughput and skilled cleanroom assembly will scale simultaneously.

## 4. Demand runway and customer outside view — measurements vs forecasts

**ASML management (Oct 14, 2026)**: Q3 revenue 11.662bn, and management said customer equipment demand forecasts have increased; some customers express interest in orders for **2029 and 2030**. Management expects current planned output for **2027–28** to undersupply customers and is examining capacity expansion beyond the previously indicated increments. The 2026 FY revenue **44–45bn** and Q4 **14.2–15.2bn** are explicit issuer guidance, not backlog conversions.

**Independent customer-side indicators**:
- **TSMC** (15 Oct): Q3 2026 revenue **US$46.72bn**, **+41.2% y/y** in USD; 2nm, 3nm and 5nm represented 8%, 33%, 28% of wafer revenue respectively; advanced <=7nm **79%**. TSMC management reiterated **2026 CapEx US$60–64bn**. These support the existence of customer advanced-node activity; **do not** attribute all CapEx to lithography/ASML. TSMC 2030 High-NA intention is a roadmap, not a binding order.
- **Samsung** (29 Oct): 2026 Q3 group CapEx **KRW 29tn**, Device Solutions **KRW 27.2tn**; planned FY2026 group **KRW 84.4tn** and DS **KRW 76.8tn**. Advanced DRAM/HBM and foundry 2nm are directly relevant process areas, but this is **not proof** of ASML shipment timing. Samsung's High-NA roadmap remains planned.
- **Intel Foundry** (8 Sep joint statement): over one million wafers processed using High-NA systems across various testing/early manufacturing and selected volume layers; this is an **adoption signal with important scope restrictions**, not a uniform full-scale adoption number.

**Evidence independence classification:** ASML release and Q3 presentation = **same issuer family**. TSMC and Samsung financial disclosures = **separate primary issuers and independent of ASML as reporters**, but as equipment customers they are commercially related and their investments are not ASML orders. These count as **outside-view demand corroboration only**, not direct evidence of ASML bookings.

**Open hypothesis register**:
- `H-B1`: Advanced-node complexity increases the lithography value per wafer faster/slower than overall fab CapEx.
- `H-B2`: Services/upgrades make cash generation less cyclical; measure IBM margin and attach rate over prior recession years, including non-leading-edge China installed base.
- `H-B3`: Current customers' 2027/28 capex roadmaps convert into **binding orders, system deliveries, customer acceptance and GAAP revenue** without material pushouts.
- `H-B4`: High-NA improves customer cost per good wafer relative to multipatterning; test capital cost, cycle times, yield and source/optics reliability, rather than relying on projected productivity.
- `H-B5`: Capacity growth (+30% or more target) can be executed across Zeiss optics, advanced components, ASML cleanrooms, qualified staff and logistics without quality or working-capital deterioration.

### 4.1 Independent equipment-industry check (SEMI) and changing 2026 guidance

**Independent industry body — SEMI**, 2026-09-03: [Worldwide equipment billings for 2026 Q2](https://www.semi.org/en/semi-press-release/global-semiconductor-equipment-billings-increased-23-percent-year-over-year-in-q2-2026-semi-reports) increased **23% y/y** to **US$40.53bn**, up **11% quarter on quarter**. This is an equipment-wide industry observation, not solely ASML, not exclusively wafer lithography, and is a distinct data origin from TSMC/Samsung/ASML.

**SEMI 2026-07-14 equipment market projection**: [industry OEM equipment forecast](https://www.semi.org/en/semi-press-release/global-semiconductor-equipment-sales-forecast-to-reach-a-record-229-billion-dollars-in-2028-semi-reports) anticipates **US$165.9bn in total global equipment OEM sales for 2026 (+23.2% y/y)**, with **wafer fab equipment US$143.9bn (+23.1% y/y)**. This is a forecast based on SEMI models, **not yet realized equipment sales**, and its categories do not equal ASML's addressable EUV/DUV revenue.

**Issuer 2026 management forecast revisions:** [Jan 28, 2026 Q4 2025 release](https://www.asml.com/en/news/press-releases/2026/q4-2025-financial-results) estimated 2026 ASML net sales **EUR 34–39bn** at 51–53% gross margin; [July 15 Q2 2026 results](https://investor.asml.com/news-releases/news-release-details/q2-2026-financial-results) revised this to **EUR 43–45bn** and 54–56%; [Oct 14 Q3 results](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026) updated to **EUR 44–45bn** at around 56%. This is a large **positive guidance revision**, not three realized outcomes or a certified 2027–2030 earnings trajectory.

**Counter-thesis check:** strong upstream industry billings and producer capex can reflect a cyclical peak as well as a sustained higher equipment spending base. Require historical comparisons, backlog **quality** and end-demand economics before translating such signals into permanent ASML fair value.

## 5. Research risk/forensic matrix (working priorities)

| Risk | Known evidence | Leading indicator / test | Financial consequence |
|---|---|---|---|
| Customer concentration and semicap cycle | Limited number of equipment-buying chipmakers (2025 report) | TSMC/Samsung/Intel fab CAPEX, cancellations/pushouts, equipment lead times | Order conversion and gross margin |
| China/export controls | China 29.1% of 2025 sales and export-rule uncertainty | Country demand and license changes, support restriction, substitution | Mix shifts, DUV sales and service risk |
| Critical optics bottleneck | Zeiss SMT sole optical source | Supplier capacity updates, uptime, delivery lead times | Missed capacity expansion/shipments |
| High-NA economics | TSMC/Samsung roadmap; Intel select use | Verified cost per wafer, productivity, customer adoption scope | Return on R&D / adoption timing |
| Working-capital and customer advances | Q3 CFO EUR 11.798bn includes **+7.009bn** assets/liabilities movement; `noncurrent` contract liabilities Q3 EUR 9.301bn (not total customer advances) | Detailed current+noncurrent contract-liability notes, Q4 reversal, billing and deposits | Large variability of cash multiple/OE |
| Investments and nonoperating items | Q3 reported net income includes EUR 509.7m equity-investment gain | Nonoperating P&L and gains vs normal EBIT | Earnings quality / unadjusted P/E |
| Share count and valuation timing | G001 unresolved; 2026-10-29 close EUR 1504.8 vendor/late-trade corroborated, unsealed EOD | Official issuer issued/treasury ledger, XAMS EOD source | **Certification blocked, exploratory scenarios permitted** |

## 6. Research working-case / trigger structure (NOT investment-thesis verdict)

No point estimates of moat value, return-on-capital advantage, price target, expected return or investment verdict are authorized from this dossier.

| Working scenario | Business-side trigger | Main numerical assumptions to test later | What falsifies it |
|---|---|---|---|
| High demand persists | Customer capacity commitments, stable sourcing/High-NA adoption | EUV/DUV throughput, service intensity, yield, cash conversion | Supplier delays, export restrictions, cancelled capacity |
| Capacity demand moderates | AI/advanced-node spend slows or deferred fabs | Lower system sales, service support and mix, high fixed costs | Sustained firm booking-to-revenue conversion |
| Technology adoption shifts | High-NA slower adoption / better DUV process economics | Product mix, upgrades vs new systems, R&D economics | Cost/wafer proves High-NA value and deployment accelerates |

**No probabilities or return forecasts** until source-backed numerical scenarios and formal later-stage authority.

## 7. Unfinished Research requirements and non-G001 priority order

1. Add **2025 official US GAAP consolidated cash flow and accounting policy notes** (2025 annual + 2026 interim) and reconcile sustainable Owner Earnings and normalized taxes/invested capital. Distinguish time-concentrated advances from durable FCF, and independent annual auditor opinions from interim unaudited reports.
2. Reconcile ASML **orders, backlog and capacity** with cited definitions; **Q3 2026 net bookings numerical amount not independently verified in this dossier; DO NOT INVENT IT**. Verify quarterly source and separate annual booked backlog from non-binding future inquiries.
3. Build ASML **2024–2026 customer-geography mix** and export-rule chronology; 29.1% China is FY2025 only.
4. Test customer equipment CAPEX **to actual ASML addressable spend** (lithography equipment fraction, timing and node mix) rather than equating foundry capex to ASML sales.
5. Extend **outside view** beyond publicly disclosed customer plans to semiconductor capex cycle independent sources (equipment industry data, foundry mix, AI-cycle reversal cases).
6. Keep the exact shares and official EOD source **certification track B** separate. Continue analysis in Research working space; **no Research COMPLETE or Deep Dive bootstrap** until all *material* evidence adequacy and formal Pilotage gates met.

**Final state**: `BUSINESS_MODEL_SOURCE_COVERED_WORKING`, `EUV_TECHNOLOGY_EVIDENCE_STRONG_BUT_NO_FINAL_MOAT_VERDICT`, `DEMAND_OUTSIDE_VIEW_PARTIALLY_COVERED`, `RISK_REGISTER_EXPANDED`, `ROIC/OE_NOT_CERTIFIED`, `G001/G006_NOT_CERTIFIED`, `STAGE_IN_PROGRESS`.
