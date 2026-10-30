# ASML — Semiconductor downturn replay, bookings lag and cash resilience

**Research** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` · **Frozen cutoff** 2026-10-30 · **US GAAP for ASML historical financials**.  
**Classification** `RESEARCH_WORKING_NON_ADMITTED`. No final issuer forecast, ROIC/Owner Earnings certification, DCF, valuation, investment thesis, contract override, Registry admission or Stage transition.  
**Machine-reproducible data and assumptions:** [ASML 2022–2025 historical / 2027–2029 stress v0.7](./ASML_2026-10-30_a1e59126_CYCLE_DOWNTURN_HISTORICAL_REPLAY_MODEL_V0_7.json).  
**Baseline working model:** [prior v0.4 central system/IBM/cash cash-flow model](./ASML_2026-10-30_a1e59126_GROWTH_MARGIN_CASH_CONVERSION_SCENARIOS_V0_4.json). Prior v0.4 is not altered by this overlay.

## 1. Hypothesis under test: orders lead sales, cash can diverge from both

The proposition is **NOT** "ASML revenue must follow semiconductor equipment spending year-on-year." System order acceptance, gross backlog, customer lead times, system shipments/acceptance, upgrades, deposit collections and R&D/capacity commitments form separate time series. ASML no longer disclosed numeric quarterly bookings or backlog in the 2026 Q1–Q3 investor release set; no Q3 2026 booking quantity is reconstructed.

**Historical issuer-origin measurements**:

| Fiscal year | System net bookings EUR bn | System sales EUR bn | Mixed IBM EUR bn | Bookings growth y/y | Systems sales growth y/y | IBM growth y/y |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 30.674 | 15.430 | 5.743 | — | — | — |
| 2023 | 20.040 | 21.939 | 5.620 | -34.67% | 42.18% | -2.14% |
| 2024 | 18.899 | 21.769 | 6.494 | -5.69% | -0.77% | 15.56% |
| 2025 | 28.035 | 24.474 | 8.193 | 48.34% | 12.43% | 26.16% |

Primary evidence:
- [ASML FY2022 release (Jan 2023)](https://www.asml.com/en/news/press-releases/2023/q4-2022-financial-results): FY2022 gross rounded sales **EUR 21.173bn**, IBM **EUR 5.743bn**, bookings **EUR 30.674bn**; system sales **~EUR 15.430bn** by subtraction of rounded values.
- [ASML FY2023 release (Jan 2024)](https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu27.6-billion-net-sales-and-eu7.8-billion-net-income-2023): FY2023 bookings **EUR 20.040bn**, sales **EUR 27.559bn**, IBM **EUR 5.620bn**.
- [ASML 2024 US GAAP 20-F](https://www.sec.gov/Archives/edgar/data/937966/000093796625000009/asml-20241231.htm) and [FY2024 release](https://investor.asml.com/news-releases/news-release-details/q4-2024-financial-results): FY2023/24 exact systems revenue **EUR 21.9386/21.7687bn**, mixed IBM **EUR 5.6199/6.4942bn**, and 2024 bookings **EUR 18.899bn**.
- [ASML FY2025 US GAAP 20-F](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/asml-20251231.htm) and [FY2025 results (Jan 2026)](https://investor.asml.com/news-releases/news-release-details/q4-2025-financial-results): net bookings **EUR 28.035bn**, system sales **EUR 24.4743bn**, IBM **EUR 8.193bn**. Published by the cutoff.

**Observed (not forecast) disconnection:** 2023 net bookings were **−34.67%** versus 2022 while system revenue **rose approximately 42.2%**, then FY2024 system revenue changed by only **−0.77%**. This is compatible with a backlog/timing buffer, **not proof of a stable one-year lag coefficient** or immunity from cancellations. The same period shows **2023 IBM −2.15% vs 2022** and **2024 IBM +15.56%**; it is repeat-demand potential mixed with upgrade/utilization cyclicality, not a guaranteed annuity.

**Observed cash/cycle disconnect:** ASML 2023 **EBIT EUR 9.0423bn**, simple FCF **EUR 3.2472bn**; 2024 **EBIT EUR 9.0226bn**, simple FCF **EUR 9.0831bn**. EBIT was nearly unchanged (**−0.22%**) while FCF rose approximately **+179.72%**. The simple free cash flow is operating CFO less cash purchases of PPE and intangibles; it is **not normalized Owner Earnings**. Evidence: SEC FY2024 20-F and annual cash-flow tables. Historical cash flow cannot be predicted from sales growth alone.

## 2. Outside-view comparison: customer and industry cycles differ by region

Independent disclosure available at the cutoff:
- [SEMI report on FY2023 equipment billings (Apr 2024)](https://www.semi.org/en/news-media-press-releases/semi-press-releases/global-semiconductor-equipment-billings-slip-to-%24106.3-billion-in-2023-semi-reports) reported **worldwide −1.3%**, **Taiwan −27%**, **South Korea −7%** and **China +29%**. Global near-flat equipment spending masked sharply contrasting customer region trends. These are geographically distributed **equipment market billings**, not an ASML booking ledger.
- [TSMC FY2023 annual report](https://investor.tsmc.com/static/annualReports/2023/english/index.html) reported **2023 revenue −4.5% in TWD** and operating margin **42.6% vs 49.5% FY2022**. These are TSMC's *sales and earnings*, not its ASML equipment orders.
- [Samsung Jan 2024 official FY2023 capital expenditure filing](https://www.samsung.com/global/ir/reports-disclosures/public-disclosure-view.84117/) reported **KRW 48.4 trillion** of DS investment in 2023 even as semiconductor memory inventory/demand corrected. Investments can continue during cyclical weak demand because of node migration, fab infrastructure and R&D.
- **Do not translate chipmaker aggregate CapEx into euros of ASML system orders**, or assume each customer responds identically.

The history favors a **two-level cycle model**: commercial shock **at order acceptance** and later **realized system/productivity-upgrade cash impact**, with neither lag nor conversion fixed from the disclosed series.

## 3. Stress construction — clearly hypothetical, calibrated to NO historical coefficient

Take the existing v0.4 **2027–2029 CENTRAL case** as a *working scenario*, not an ASML management forecast. The replay **imagines** a new commercial orders shock in 2027, symbolically comparable to the observed 2023 booking correction, **without assuming the 2027 booking amount**.

- The stress engine holds the **2027 original central case unchanged** to demonstrate a possible order-book buffer. This is a **model design choice**, not evidence that actual FY2027 revenue would be fully insulated.
- **2028**: separately haircut original central systems and mixed IBM sales, reduce their gross margins and impose additional operating cash-timing outflow.
- **2029**: partial sales/margin recovery, with remaining operating cash timing drag.
- R&D and SG&A follow the **original central nominal expense path** instead of decreasing proportionally with sales (cost stickiness); likewise PP&E/intangible *cash* CapEx and D&A are held **at central nominal values**, not as percentages of reduced sales. This is a conservative one-way investment-commitment stress, **not proof that ASML could not cut planned CapEx**.
- Hypothetical operating tax remains 18%; SBC stays expensed in EBIT, no incremental cash addback. No effect on exact shares or XAMS valuation.
- **System sales haircuts are NOT computed by multiplying the observed historical −34.67% orders decline by a purported historical conversion coefficient.** Historical order shock is a *regime illustration*, while all 2028/29 haircuts, margin penalties and cash timing effects are **judgmental hypothesis inputs**.

### 3.1 Explicit shocks compared with the original central trajectory

| Case | System sales haircut 2028 / 2029 | Mixed IBM haircut 2028 / 2029 | Systems GM penalty pp | IBM GM penalty pp | Additional WC cash timing 2028 / 2029 (% of shocked revenue) |
|---|---|---|---|---|---|
| Limited | −5% / −3% | −2% / 0% | −1 / −0.5 | −2 / −1 | −1% / −0.5% |
| Historical-style (NOT fitted) | −12% / −8% | −5% / −2% | −2 / −1.5 | −3 / −2 | −4% / −2% |
| Severe | −25% / −15% | −10% / −6% | −4 / −3 | −6 / −4 | −8% / −4% |

The term **historical-style** identifies which macro pattern motivates stress consideration; it is **not a backtested reproduction** of an empirical 2023-to-2024 shock elasticity. The downturn numbers intentionally exceed some actual 2024 system revenue movements to examine tail exposure.

## 4. Recomputed year-by-year model — all cash numbers WORKING PROXIES

Formula (EUR million):
```
STRESSED_SYSTEMS_t = V04_CENTRAL_SYSTEMS_t × (1 − systems_haircut)
STRESSED_IBM_t = V04_CENTRAL_IBM_t × (1 − mixed_IBM_haircut)
STRESSED_GROSS_PROFIT_t =
  STRESSED_SYSTEMS_t × (V04_SYS_GM_t − systems_GM_penalty)
 +STRESSED_IBM_t × (V04_IBM_GM_t − IBM_GM_penalty)
EBIT_t = STRESSED_GROSS_PROFIT_t − V04_RD_EXPENSE_t − V04_SGA_EXPENSE_t
CASH_STRESS_PROXY_t = EBIT_t × (1 − 18% hypothetical operating tax)
  + V04_DA_EUR_t − V04_PPE_AND_INTANGIBLE_CASH_CAPEX_EUR_t
  + STRESSED_REVENUE_t × WC_timing_stress_percent
```
These are **not official CFO, normalized FCF, Owner Earnings or an actual customer-deposit forecast**; source documents do not supply the missing customer-credit and deferred revenue changes. Maintaining central nominal CapEx while lowering revenues magnifies cash stress; it is not an estimate of incremental permanent maintenance CapEx.

| Scenario | Year | Stressed sales EUR bn | EBIT margin | **Cash stress proxy EUR bn** | Cash versus original central |
|---|---:|---:|---:|---:|---:|
| LIMITED | 2027 | 48.305 | 41.6% | **15.082** | 0.0% |
| LIMITED | 2028 | 50.129 | 39.6% | **14.260** | -12.5% |
| LIMITED | 2029 | 54.904 | 40.0% | **16.097** | -6.6% |
| HISTORICAL_STYLE | 2027 | 48.305 | 41.6% | **15.082** | 0.0% |
| HISTORICAL_STYLE | 2028 | 47.054 | 37.6% | **11.095** | -31.9% |
| HISTORICAL_STYLE | 2029 | 52.574 | 38.3% | **13.825** | -19.7% |
| SEVERE | 2027 | 48.305 | 41.6% | **15.082** | 0.0% |
| SEVERE | 2028 | 41.429 | 32.9% | **6.357** | -61.0% |
| SEVERE | 2029 | 49.121 | 35.5% | **10.706** | -37.9% |

**Central comparator v0.4:** 2028 revenue **EUR 52.304bn** and cash proxy **EUR 16.302bn**; 2029 revenue **EUR 56.112bn** and proxy **EUR 17.226bn**. Under the 2028 **historical-style** model, revenue falls only from **EUR 52.304bn to EUR 47.054bn (approximately −10.0%)**, while the cash proxy drops from **EUR 16.302bn to EUR 11.095bn (−31.94%)**. Under **severe**, the 2028 sales/revenue stress is **EUR 41.429bn**, cash proxy **EUR 6.357bn (−61.01% versus central)**. These compare **hypothetical cases**, not an ASML management guidance range or statistical probability.

The **IBM stream is not fully protected**: the stress affects optional productivity upgrades, wear-based source maintenance, spares, fleet utilization and service gross margin. The model does not assume that all IBM vanishes, or that a fixed percentage of IBM is contracted maintenance — it is **undisclosed**.

## 5. Leading-indicator checklist (source requests, not current confirmed triggers)

1. **Bookings and customer agreements:** dated accepted POs, deposits/penalties, cancellations and deferrals; numeric Q1–Q3 2026 bookings not disclosed in issuer news releases. Avoid deriving bookings from contract liability balances.
2. **Customer demand:** actual/forecast TSMC/Samsung/Intel lithography-specific capacity and deployment, **not** their gross CapEx totals.
3. **IBM resilience:** maintenance contracts vs parts/OnPulse use vs productivity upgrades, service gross margin trend, service installed base by EUV/DUV/High-NA.
4. **Cash conversion:** collect actual movement in current plus noncurrent customer contract liabilities, deferred revenue, working capital and system acceptance settlement; no reconstruction from noncurrent-only quarterly line.
5. **Supply commitments:** critical Carl Zeiss SMT optics capacity and ASML production milestones, customer acceptance, export-control exposure.
6. **Operating capital/ROIC:** five-quarter gross operating assets are documented separately; net invested capital and true normalized operating tax rate remain uncertified.

**Suggested prospective stress controls** (proposals, not in production): rerun the replay if future source-confirmed annual bookings contract substantially while system revenue still grows, when IBM GM moves materially below historic 2025 50.87%, or when CFO diverges sharply from profit and customer liability balances reverse. These are prompts for **future research monitoring**, not confirmation that any condition is now met.

## 6. Governance / actionable conclusion

**The previous Research v0.4 scenario is not invalidated by the 2022–2024 episode**; it is *insufficient alone* to establish earnings durability. The historical evidence confirms a clear **orders → recognized sales → service mix → cash timing** sequence with uneven effects.

`HISTORICAL_CYCLE_SOURCE_VERIFIED = YES`; `BOOKINGS_2023_DROP_AND_REVENUE_LAG_OBSERVED = YES`; `CASH_RESILIENCE_VS_EBIT_DIVERGENCE_OBSERVED = YES`; `TSMC_SEMI_OUTSIDE_VIEW = COVERED`; `2027_29_SHOCK_PASS_THROUGH_ESTIMATED = NO`; `CASH_2028_STRESS_BOUNDS = HYPOTHETICAL_NOT_STATISTICAL`; `OWNER_EARNINGS_AND_ROIC_CERTIFIED = NO`; `RESEARCH_STAGE_COMPLETION=NOT_AUTHORIZED`.

**Next distinct Research action:** collect same-basis US GAAP historical changes in accounts receivable, inventories, deferred / contract liabilities, and cash conversion during **2022–2024**, then compare with current Q3 2026 cash spike to calibrate plausible *BFR stress* from observable cash statement history rather than indefinitely varying arbitrary WC percentage inputs. Do not reopen exact G001 shares as a prerequisite.
