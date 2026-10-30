# ASML — Growth, gross-margin and cash-conversion stress scenarios (Research v0.4)

**RUN_ID:** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` · **DATA_CUTOFF:** 2026-10-30 · **Stage:** `ACTIVE/RESEARCH`, frozen Research V2.0.  
**Evidence period:** US GAAP through Q3 ended 2026-09-27; 2026 full-year figures **not yet realized**.  
**Status:** `NON_ADMITTED / HYPOTHETICAL_FINANCIAL_STRESS / NO_DCF_OR_FAIR_VALUE / NO_FINAL_MOAT_OR_RUNWAY_VERDICT / NO_STAGE_ADVANCE`.  
**Companion:** [reproducible input and formula model](./ASML_2026-10-30_a1e59126_GROWTH_MARGIN_CASH_CONVERSION_SCENARIOS_V0_4.json).

## 1. Objective / finding

Convert separate company disclosures for (i) systems, (ii) IBM service **plus options/upgrades**, (iii) historical margins and operating costs, and (iv) working-capital-dependent cash into internally consistent **2027–2029 operating scenarios**. This is a **research stress exercise**, neither an issuer forecast nor an expected earnings path. **Do not assign probabilities or an investment decision based on these numbers.**

The key insight: at a given pace of sales growth, the cash result is highly dependent on **mix, operating investment and changes in operating cash timing**; therefore one must neither treat all IBM revenue as contractually recurring nor annualize the extraordinary cash collections from recent reporting periods.

## 2. Issuer-sourced baseline (distinguish actual versus guidance)

| Data | Value | Evidence status |
|---|---:|---|
| **FY2025 IBM** | **EUR 8,193.0m** revenue / **EUR 4,025.3m** cost / **50.87%** gross margin | Reported FY2025 US GAAP, [SEC XBRL Operations R3](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R3.htm) |
| **2026 first 9 months IBM** | **EUR 8,538.1m** | Reported in [ASML Q3 2026 US GAAP statements](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/10/14/cpr01_notified_Financial%20statements%20US%20GAAP%20Q3%202026.pdf), p3; Euronext-labeled user copy SHA-matches SHADOW receipt |
| **2026 9M R&D and SG&A** | **EUR 3,635.9m** / **EUR 957.6m** | Reported Q3 2026 GAAP, p3 |
| **2026 Q4 guide (rounded)** | IBM ~EUR 3.7bn; R&D ~EUR 1.2bn; SG&A ~EUR 0.4bn | [ASML Q3 issuer statement](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026), forward-looking |
| **FY2026 guide** | Revenue EUR 44–45bn, group gross margin **around 56%** | Issuer management guidance, not FY actual |

**Chosen modeling anchor, not historical FY2026 earnings:** group revenue **EUR 44.5bn**, IBM **EUR 12.2381bn** (actual nine months + approx Q4 forecast), systems **EUR 32.2619bn**. FY2026 IBM gross margin is **not separately published**. We provisionally *carry forward FY2025's actual IBM gross margin of 50.87% solely to calibrate* the 2026 forecast midpoint. The system gross margin **algebraically implied** by the 56% guided aggregate becomes **57.946%**. **This figure is NOT an observed system margin.** FY2026 R&D/SG&A approximated as 9M actual plus Q4 budget imply EBIT around **EUR 18.7265bn**, not actual FY EBIT.

**IBM attribution:** contract-term maintenance and warranties; usage and spare parts; episodic productivity upgrades and options. ASML did not report their exact monetary shares. The model explicitly carries `IBM_CONTRACTUAL_RECURRING_SHARE = UNKNOWN` for **every** scenario and never values IBM as an automatic annuity.

## 3. Scenario engine and explicit economic assumptions

**Formula chain:**

```
SYSTEMS_t  = SYSTEMS_(t-1) × (1+scenario_systems_growth_t)
IBM_t      = IBM_(t-1) × (1+scenario_mixed_IBM_growth_t)
REVENUE_t  = SYSTEMS_t + IBM_t
GROSS_PROFIT_t = SYSTEMS_t × assumed_SYSTEMS_GM_t + IBM_t × assumed_IBM_GM_t
EBIT_t     = GROSS_PROFIT_t - assumed_RD_t - assumed_SGA_t
NOPAT_PROXY_t = EBIT_t × (1-hypothetical_operating_tax_rate)
CASH_STRESS_PROXY_t = NOPAT_PROXY_t + hypothetical_DA_t
                      - cash_PPE_and_intangible_capex_t + WC_cash_timing_stress_t
```

This metric is **not audited CFO, certified FCFF, normalized FCF or Owner Earnings**. The final cash line is a scenario assumption applied as a percentage of revenue; it is **NOT a measured forecast of customer advances**. SBC is already a cost in US GAAP EBIT and is not added back; future dilution, current debt, leases, deferred tax and one-off items remain outside this proxy.

### Driver assumptions (all estimates, none issuer guidance)

| Scenario | Systems revenue growth 2027/28/29 | Whole-IBM revenue growth 2027/28/29 | Systems GM 2027/28/29 | Mixed IBM GM 2027/28/29 | Cash timing stress as % of total sales 2027/28/29 |
|---|---|---|---|---|---|
| DOWNSIDE | -15/2/4% | -10/2/4% | 51.5/52.5/53.5 | 44/46/48 | -6/-2/0 |
| CENTRAL | 8/8/7% | 10/9/8% | 57.5/57.5/57 | 51/51.5/51.5 | 0/0/0 |
| UPSIDE | 16/14/10% | 15/13/11% | 59.5/60/59.5 | 54/55/54 | 2/1/-2 |

Additional explicit modeling parameters:
- **Downside:** R&D grows +7/+7/+6%, SG&A +5/+5/+5%; operating tax 20%; D&A 2.5% of sales; cash PPE/intangible spend 6/5.5/5% of sales.
- **Central:** R&D +10/+10/+9%, SG&A +9/+9/+8%; operating tax 18%; D&A 2.6% of sales; capex 5.5% of sales annually.
- **Upside:** R&D +14/+14/+12%, SG&A +12/+12/+11%; operating tax 17.5%; D&A 2.7% of sales; capex 7/7/6.5% of sales.
- R&D/SG&A cost projections compound annually from the **2026 guidance-derived anchor**. They are never back-solved from target EBIT multiples.
- **No scenario percentage is a source-observed breakdown** of maintenance/upgrades, a probability, or a certified revenue floor.
- The **2026 Q3 backlog/order amount is not numeric in the issuer's releases**. No booked order is inferred from contract liability changes.
- Supply bottlenecks (ZEISS optics), customer capex volatility, China export controls and High-NA adoption are scenario **risk mechanisms**, not proven 2027 quantitative input multipliers.

## 4. Outputs — EUR billions for amounts; margins %

| Case | Year | Systems | IBM (mixed) | **Total sales** | Group GM | EBIT margin | **Operating cash stress proxy** |
|---|---:|---:|---:|---:|---:|---:|---:|
| DOWNSIDE | 2027 | 27,42 | 11,01 | **38,44** | 49,4% | 32,2% | **6,24** |
| DOWNSIDE | 2028 | 27,97 | 11,23 | **39,21** | 50,6% | 32,7% | **8,30** |
| DOWNSIDE | 2029 | 29,09 | 11,68 | **40,77** | 51,9% | 33,7% | **9,97** |
| CENTRAL | 2027 | 34,84 | 13,46 | **48,30** | 55,7% | 41,6% | **15,08** |
| CENTRAL | 2028 | 37,63 | 14,67 | **52,30** | 55,8% | 41,5% | **16,30** |
| CENTRAL | 2029 | 40,26 | 15,85 | **56,11** | 55,4% | 41,0% | **17,23** |
| UPSIDE | 2027 | 37,42 | 14,07 | **51,50** | 58,0% | 44,3% | **17,65** |
| UPSIDE | 2028 | 42,66 | 15,90 | **58,57** | 58,6% | 45,0% | **19,81** |
| UPSIDE | 2029 | 46,93 | 17,65 | **64,58** | 58,0% | 44,2% | **19,79** |

Interpretation: the **downside** stresses a systems-order and episodic-upgrade reset, not the disappearance of every maintenance contract; **central** assumes modest service/upgrade growth but no extra positive cash timing; **upside** includes stronger system deliveries/IBM upgrade mix but higher capital expenditures and **reversal of positive cash timing in 2029**. Consequently the upside cash proxy does not mechanically follow revenue.

**Non-equivalence to realized 2026 FCF:** group LTM CFO less PPE/intangible cash capex through Q3 2026 was **EUR 20.9994bn**, with **+EUR 7.8769bn** in the reported cash-flow `Changes in assets and liabilities` line. A heuristic excluding that one line yields **EUR 13.1225bn**. Neither is normalized Owner Earnings; neither is an issuer-assessed 2027 FCF baseline. The 2027 modeled cash proxies use a distinct EBIT-to-cash bridge, **not** an annualized 2026 windfall.

## 5. Materiality and falsification tests for the next Research cycle

| Test | Evidence needed to reduce uncertainty | Research decision impacted |
|---|---|---|
| IBM maintenance vs upgrade financial mix | Primary issuer revenue + gross margin broken down by maintenance/contract duration, usage and upgrade project | Whether IBM GM >50% is repeatable; whether double-digit IBM growth persists |
| Orders converted to accounting revenue | Dated accepted POs, deferred revenue release, shipped/accepted system mix and cancellation rights; do not invent quarterly 2026 net bookings | Systems volume/growth scenarios and cash timing |
| Deferred revenue versus cash deposits | Complete **current + noncurrent** US GAAP liabilities and *cash* movements with contract assets | Working-capital cash stress calibration |
| Maintenance capex | Asset and tooling replacement, operating capacity upkeep vs growth, R&D capitalization policy, depreciation | Net cash residual after sustaining the franchise |
| Margin under high fixed cost | Historical revenue mix with R&D and SG&A 2023–2026, separate system vs IBM gross profit latest period | Downside operating leverage |
| Customer independent demand | TSMC/Samsung/Intel capex commitments versus genuine lithography equipment capacity, export licenses, ZEISS optical capacity | Scenario probability calibration if later authorized |
| Audit/provenance | Original issuer disclosures sealed with hashes and linked to frozen Ledger, stage manifest and gap adjudication | Formal evidence authority; G001/G006 continue separately |

**Falsification triggers:** IBM GM declines toward or below FY2023 service GM 41.79%; upgrade orders slip, customer utilization materially weakens, ZEISS production ramp lags, high-NA deployment delays, receivable/contract assets rise materially versus cash receipts, tax/maintenance capex exceeds assumptions. None is assigned a probability at this stage.

## 6. Governance disposition

`RESEARCH_WORKING_SCENARIOS_BUILT = YES` · `IBM_CONTRACTUAL_SHARE = UNDISCLOSED` · `NORMALIZED_OWNERS_EARNINGS = NOT_CERTIFIED` · `FINAL_MOAT/RUNWAY = PROHIBITED_IN_RESEARCH` · `G001/G006 = OPEN_AS_SEPARATE_CUSTODY_TRACK` · `READY_FOR_DEEP_DIVE = NOT_EVALUATED` · `PRODUCTION_REGISTRY_MUTATION = NO`.

Research continuation does **not** require solving exact 2026-10-30 shares in order to test operating economics; any Stage completion requires the frozen Research V2 contract's own evidence sufficiency, source custody, Registry reconciliation and Pilotage decision. No investment rating, DCF, price target or public recommendation issued.
