# ASML — IBM gross-margin durability and corporate sustaining CapEx: calibration and stress (Research v0.5)

**RUN** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` · **Cutoff** `2026-10-30` · **Stage** `ACTIVE / RESEARCH` · **GAAP** US GAAP only.  
**Classification:** `WORKING_RESEARCH_NON_ADMITTED`; not a final financial statement, maintenance CapEx audit, Owner Earnings, final moat verdict, DCF, fair value or investment recommendation. No change to frozen Research V2.0 contracts, Pilotage, Registry or stage status.  
**Reproducibility:** [machine-readable inputs and calculated sensitivity table](./ASML_2026-10-30_a1e59126_IBM_MAINTENANCE_CAPEX_MARGIN_CALIBRATION_MODEL_V0_5.json), linked to the [2027–2029 prior operating scenario v0.4](./ASML_2026-10-30_a1e59126_GROWTH_MARGIN_CASH_CONVERSION_SCENARIOS_V0_4.json).

## 1. New primary-source evidence

The primary [2025 US GAAP statement of operations SEC XBRL R3](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R3.htm) discloses total IBM (**service AND field-option**) revenue/cost but **not** contractual service profit alone. This corroborates the already verified annual service gross margins **41.79% (2023) → 48.20% (2024) → 50.87% (2025)**. These are **historical mixed-IBM margins**, not "maintenance-only" profitability. 2025 US GAAP R&D expense **EUR 4,698.8m** is included in consolidated EBIT; direct parts and labor used to service equipment are normally included in service cost of sales. **Do not subtract those direct service expenses a second time as cash maintenance CapEx.**

The original [2025 US GAAP PPE movement schedule R93](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R93.htm), [depreciation split R95](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R95.htm), and [PPE footnote R24](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R24.htm) give:
- FY2025 **EUR 1,596.2m** accounting additions in PPE; of this **EUR 1,029.4m buildings/land**, **EUR 474.2m machinery and equipment**, **EUR 43.3m leasehold improvements**, **EUR 49.3m furniture/other**. Building additions are **64.49%** of 2025 PPE *additions*, not 64.49% of cash CapEx or "growth CapEx".
- PPE additions are described mainly as **factory and R&D expansions, capacity tooling upgrades/expansion and prototypes**; category-level monetary amounts labeled **maintenance** versus **growth** are **not** given.
- FY2025 PPE depreciation **EUR 916.0m**, split by expense function: **cost of sales EUR 532.1m**, **R&D EUR 353.2m**, **SG&A EUR 30.7m**. The EUR 532.1m cost-of-sales component spans **systems plus services**, so it **cannot** serve as an IBM depreciation allocation.
- Consolidated FY2025 **cash** purchases of PPE **EUR 1,573.6m**, cash purchases of intangibles **EUR 57.6m**, total **EUR 1,631.2m** under [annual ASML 2025 US GAAP report](https://www.sec.gov/Archives/edgar/data/937966/000162828026011377/asml-2025xannualxreportx.htm). This total is **not** equal to PPE accounting additions, and it is **not** the maintenance CapEx disclosure.
- Economic upkeep also includes **ongoing product and software R&D**, which is an operating expense under this US GAAP income statement. R&D essential for sustaining competitive capability is **not a free cash add-back** even if future product roadmaps may carry growth benefits. **Do not import the EU-IFRS development-cost capitalization policy** or mix its EBIT with US GAAP.

## 2. Three-year reported calibration — amounts EUR million

| Year | Actual IBM sales | Service and option cost of sales | IBM gross profit | IBM gross margin | **Group** PPE depreciation | **Group** PPE+intangible cash CapEx |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 5619.9 | 3271.4 | 2348.5 | 41.79% | 605.6 | 2196.2 |
| 2024 | 6494.2 | 3364.0 | 3130.2 | 48.20% | 787.3 | 2083.1 |
| 2025 | 8193.0 | 4025.3 | 4167.7 | 50.87% | 916.0 | 1631.2 |

**Interpretation:** rising IBM gross margin is observed but not proven persistent. It may reflect upgrade mix, utilization, cost efficiency and service scale, and ASML did not quantify individual contributions. The 2023 41.79% figure is a credible **observed adverse profitability reference**, not a guaranteed future floor or a forecast of all maintenance contracts.

**Historical consolidated CapEx/depreciation ratios are not maintenance ratios:** FY2023 total cash CapEx / PPE depreciation about **3.63×**; FY2024 **2.65×**; FY2025 **1.78×**. It is not justified to conclude all expenditures above PPE depreciation are growth or that cash upkeep equals PPE depreciation.

## 3. What can and cannot be classified as sustaining expenditures

| Category | Financial location | What is defensible | What is not disclosed |
|---|---|---|---|
| Spare parts, service engineers, repairs, operation of installed tools | IBM cost of sales / support expenses | Already reduce IBM gross profit | Actual maintenance *contract revenue* share and allocated cost per product |
| Essential EUV/DUV/software engineering / obsolescence prevention | US GAAP R&D operating expense | Already reduces EBIT; continuing expense may be economically essential | Company-wide maintenance R&D vs expansion R&D split |
| ASML production tooling and facilities replacement | Corporate cash PPE purchases | Real consolidated cash CapEx; some capacity required for ongoing operations | Precise **upkeep vs growth** split; IBM vs systems asset allocations |
| Capacity additions, cleanroom construction, next-gen prototypes | PPE accounting additions/cash CapEx | Expansion projects explicitly described by ASML | Exact cash amount uniquely attributable to new growth |
| Customer advance funding / contract liabilities | Group balance sheet and CFO changes | Cash timing may temporarily amplify reported FCF | Exact booked IBM cash flow vs new system cash advance bridge |

**Most important accounting control:** annual IBM **gross profit is not a segment operating profit**. R&D, sales/admin expenses, cash taxes, allocated working capital, sustaining PP&E assets and field upgrades' cost have not been mapped to independent IBM owner cash earnings. Treat `IBM_DIRECT_DURABLE_FCF = UNKNOWN`.

## 4. Corporate upkeep-capex diagnostic — only hypothetical allocations

For FY2025 corporate cash CapEx **EUR 1,631.2m** and PPE depreciation **EUR 916.0m**, impose purely *hypothetical* ratios of sustaining CapEx to physical depreciation. This is an **allocation sensitivity**, not even an estimated band.

| Hypothetical corporate sustaining capex / FY2025 PPE depreciation | *As-if* corporate upkeep, EURm | Unassigned portion of FY2025 **total** cash CapEx, EURm | Authority |
|---:|---:|---:|---|
| 0.75× | 687.0 | 944.2 | Unverified / corporate only |
| 1.00× | 916.0 | 715.2 | Unverified / corporate only |
| 1.25× | 1145.0 | 486.2 | Unverified / corporate only |

None of these three numbers is an observed maintenance budget, a statistically defensible lower/upper bound or the **IBM** maintenance investment. The residual is **unassigned**, not definitively "growth CapEx"; depreciation measures expense allocation, not replacement cash cost, and real replacement costs may be above/below historical depreciation, while cash purchases differ from accounting additions.

**Important for existing v0.4:** its cash CapEx ratios of **5–7% of group forecast sales** are group-wide forward stress assumptions that may include both expansion and sustaining expenditures. They must **not be reduced** to the depreciation-ratio proxy when computing `CASH_STRESS_PROXY`, as that would turn a corporate cash obligation into unjustified distributable cash.

## 5. IBM gross-margin normalization overlay on existing 2029 central scenario

Keep unchanged v0.4 scenario values: 2029 IBM (maintenance **plus** options/upgrades) **EUR 15,847.36m**, group total sales **EUR 56,111.76m**, **IBM gross margin assumed 51.50%**, after-tax proxy **18%**, corporate cash stress proxy **EUR 17,226.22m**. Change **only** IBM gross margin while holding systems sales/margins, operating expenses, group CapEx and operating working-capital cash timing fixed.

| IBM hypothetical gross margin in 2029 | IBM gross profit, EURbn | Change to **group EBIT**, EURbn | Recomputed **group operating cash stress proxy**, EURbn |
|---:|---:|---:|---:|
| 41.79% | 6.622 | -1.539 | 15.964 |
| 48.20% | 7.638 | -0.523 | 16.797 |
| 50.87% | 8.061 | -0.100 | 17.144 |
| 51.50% | 8.161 | 0.000 | 17.226 |
| 54.00% | 8.558 | 0.396 | 17.551 |
| 56.50% | 8.954 | 0.792 | 17.876 |

Formula: `delta_EBIT = IBM_2029_revenue × (candidate_IBM_GM − 51.50%)`;
`delta_CASH_PROXY = delta_EBIT × (1 − hypothetical_2029_tax_rate_18%)`.
This is not an observed IBM operating cash profit or a cash-flow forecast. The historical margins used as stress inputs (**41.789%, 48.200%, 50.869%**) were observed for different product mix/years and are **not** a maintenance/upgrade decomposition.

**Sensitivity units, 2029 central working case:**
- **±5 percentage points** of IBM gross margin → group cash stress proxy **±EUR 649.742m** after the hypothetical 18% operating tax.
- **±1 percentage point** of group sales for corporate cash CapEx → **∓EUR 561.118m** of cash proxy.
- **±2 percentage points** of group revenue from net working-capital *cash timing* → **±EUR 1,122.235m** cash proxy.
- If all three adverse movements occur together, the illustrative group operating cash stress proxy decreases from **EUR 17.226bn to EUR 14.893bn**. Opposite combined stress raises it to **EUR 19.559bn**. Neither endpoint is a statistical confidence interval, a company forecast, Owner Earnings nor approved distributable cash; cash tax may not track the illustrative 18% and capital structure/dilution remain excluded.

This demonstrates that **working-capital timing and gross-margin durability can dominate second-order precision in the equity share denominator**, without claiming they dominate every possible macro or investment risk.

## 6. Research next-source priorities and controls

**Highest value:** original ASML service maintenance contract/pricing/renewal data separating coverage contracts, usage/pulse programs, billable spares and field upgrade projects, ideally with margins and fleet by technology; proprietary data are likely absent from public financials, so a lack of disclosure is an **explicit model limitation**, not a reason to repeat internet searches indefinitely.

**Corporate cash:** reconcile the *actual* cash PPE spend, asset-category PPE additions and physical depreciation; add quarterly useful-life/asset-age mapping and 2026 US GAAP information where available. Any claimed single number for "maintenance CapEx" needs company disclosure, a defensible bottoms-up asset replacement model or a later formally approved range with uncertainty bounds—not arbitrary depreciation equality.

**Assumption governance:** follow frozen Research V2.0 Evidence/Calculation/Material Assumption lineage. New source URLs have **not been independently byte-sealed into the private source registry**; link/citation evidence and arithmetic exist as working references only. If authorized, file admission may be requested separately. Continue other analytical work without reopening G001 as a prerequisite; do not complete Research Stage or publish a price target based on this paper.

**Status:** `IBM_2023_25_GROSS_MARGINS_VERIFIED`, `FY2025_GROUP_PPE_DEP_AND_PURCHASES_RECONCILED`, `MAINTENANCE_CAPEX_AMOUNT_UNDISCLOSED`, `IBM_OWNER_EARNINGS_UNIDENTIFIABLE`, `2029_CASH_SENSITIVITY_CALCULATED`, `RESEARCH_STAGE_REMAINS_IN_PROGRESS`, `REGISTRY_ADMISSION=NO`.
