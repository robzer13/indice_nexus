# ASML — Return quality, financing supplied by customers and invested-capital perimeter

**RUN:** `a1e59126-1ca9-4e55-8764-54a1e3d4dc24` · `ACTIVE / RESEARCH` · **Cutoff:** 2026-10-30 · **Accounting:** US GAAP only  
**Classification:** `PRE_ADMISSION_RESEARCH_ONLY`, **not an issuer-certified ROIC, no owner-earnings estimate, no fair value, no final moat verdict, no published investment judgment**.  
**Model:** [five-quarter reproducible asset and liability reconciliation v0.6](./ASML_2026-10-30_a1e59126_ROIC_OPERATING_ASSET_CAPITAL_RECONCILIATION_MODEL_V0_6.json).  
**Methodology:** frozen Research Stage Contract V2.0; calculate / qualify / register conflicts, **no Stage completion without ledger and admission**.

## 1. What the primary filings actually make observable

[Issuer SEC 6-K Q3 2026 EX-99.3, pages 3–10](https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/financial_statementsxusx.htm) contains the five consecutive end-of-quarter US GAAP balance sheets (2025-09-28, 2025-12-31, 2026-03-29, 2026-06-28, 2026-09-27) and 2025 Q4–2026 Q3 operating income. The 2026 interim statements are **unaudited**; original SEC HTML bytes have not yet been hash-sealed into Registry.

[Issuer FY2025 20-F XBRL Consolidated Balance Sheet R6](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R6.htm) contains a **fully broken-out** 2025 current liabilities table, which the abbreviated Q3 2026 summary **does NOT publish**. [20-F lease note R25](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R25.htm) contains current and noncurrent lease obligations.

### 1.1 Identified gross operating assets — a deliberately limited perimeter

For each quarter, calculate only precisely identified operating asset categories:

```
KNOWN_CORE_ASSETS = Accounts receivable
                  + Contract assets
                  + Inventories
                  + Property, plant & equipment
                  + Goodwill + Other intangible assets
                  + Right-of-use assets
```

**Exclusions requiring later review:** finance receivables from system/customer finance arrangements, tax assets, other current/noncurrent mixed assets, loans to Carl Zeiss SMT and other entities, equity-method investments and equity investments. The Zeiss loan and equity-method interests are **strategically relevant** and not necessarily idle cash; a compatible ROIC analysis must decide whether to include their capital **and** their corresponding income in the numerator. Exclusion here is a **perimeter diagnostic**, not proof these assets are nonproductive.

| Quarter end | AR + contract assets + inventory (EUR bn) | Identified gross operating assets (EUR bn) | Without goodwill (EUR bn) | **Old invalid** equity+LT debt−all cash proxy (EUR bn) |
|---|---:|---:|---:|---:|
| 2025-09-28 | 17.494 | 30.665 | 26.077 | 16.570 |
| 2025-12-31 | 14.893 | 28.256 | 23.668 | 8.999 |
| 2026-03-29 | 16.658 | 29.869 | 25.281 | 15.159 |
| 2026-06-28 | 19.484 | 32.797 | 28.209 | 16.228 |
| 2026-09-27 | 21.613 | 35.182 | 30.593 | 11.292 |

2025-09-28 to 2026-09-27 **four-quarter time approximation** uses five snapshots, with trapezoidal endpoint weights `[1,2,2,2,1]/8` (not a count-5 simple arithmetic mean). Results:
- **Identified gross operating assets EUR 30.962bn** average.
- **EUR 26.373bn** excluding goodwill, another *definition sensitivity*, not an economic correction.
- **EUR 17.647bn** average identified trade receivables, contract assets and inventory.

The operational core asset base grows **EUR 28.256bn at 2025-12-31 → EUR 35.182bn at 2026-09-27**; in the same period customer receivables grow **EUR 3.023bn → 9.253bn**. Gross assets do not capture the financing provided by customers and suppliers. The old debt/cash proxy fluctuates sharply with financial liquidity and timing and was properly rejected by the prior working report; **do not publish it as ROIC**.

## 2. NOPAT and gross-asset productivity, NOT net ROIC

Reported US GAAP LTM EBIT to 2026 Q3 = **2025 Q4 EUR 3,431.1m + 2026 first nine months EUR 11,786.8m = EUR 15,217.9m**. The reported group tax rate on pretax profit was approx 17.004%, but **this is not necessarily the normalized operating tax rate** (equity-method and non-operating gains, temporary differences, geographical mix). Use hypothetical tax assumptions as explicit sensitivity.

| Hypothetical operating tax | Calculated EBIT × (1−tax), EUR bn | Return on identified **GROSS** operating assets | Return excluding goodwill |
|---|---:|---:|---:|
| 15% | 12.935 | 41.78% | 49.05% |
| 17% | 12.631 | 40.80% | 47.89% |
| 20% | 12.174 | 39.32% | 46.16% |
| 25% | 11.413 | 36.86% | 43.28% |

**Reading:** at 17% hypothetical operating tax, **EUR 12.631bn** NOPAT **proxy** / **EUR 30.962bn** identified gross operating asset base ≈ **40.8%**. This is **not ASML ROIC**. Net invested capital should deduct appropriate *operating* liabilities and separate operating cash, debt leases and strategic investments consistently. The alternative ex-goodwill return is not a "tangible ROIC" because the asset perimeter remains incomplete.

Matched financial-period EBIT numerator and five snapshot gross asset denominator improve the **time alignment** relative to prior two-point comparisons, but quarter-end proxy average is still not a perfect daily asset average.

## 3. Direct confirmation: the customer-financing channel is material, but carries future obligations

The 2025 full [SEC 20-F balance R6](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R6.htm) reports **current contract liabilities EUR 16,006.3m**, **noncurrent contract liabilities EUR 3,366.3m**, combined **EUR 19,372.6m**. Current contract liabilities are **66.0%** of total current liabilities **EUR 24,263.9m**.

**2025 current liabilities exact reconciliation:**

```
Accounts payable                3,521.8
Accrued and other current       2,567.3
Current tax liabilities           486.6
Short-term/current debt         1,681.9
Current contract liabilities   16,006.3
                              ---------
Total current liabilities      24,263.9   EUR millions
```

A *simplified working-capital perimeter diagnostic* (EUR millions) uses:
```
AR 3,023.0 + contract assets 440.6 + inventory 11,429.3
  − accounts payable 3,521.8 − accrued/other current 2,567.3
  = +8,803.8 before contract liabilities

  − current contract liabilities 16,006.3
  = −7,202.5 after current contract liabilities

  − noncurrent contract liabilities 3,366.3
  = −10,568.8 after ALL contract liabilities
```

This is a **perimeter sensitivity**, not ASML's official working capital or invested capital: accrued liabilities may contain non-operating items and lease obligations; finance receivables, taxes, customer contract assets and other assets need correct matching. The **noncurrent contract liability** is not current-period CFO and the 2025 complete liability balance cannot be substituted into the Q3 2026 24,904.3m aggregate current liabilities. 2026 Q3 only explicitly shows **9,300.6m noncurrent contract liabilities**; **the current part is not separately disclosed in the GAAP summary**.

Per [2025 SEC ASC606 note R13](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R13.htm), these customer balances reflect **future delivery/installation/service obligations** and down payments, not a perpetual gift of interest-free capital. Customers may receive repayment in defined circumstances. The benefit to funding cost and capital efficiency is economic and real, but the resulting liability must remain in the solvency/working-capital stress and customer-cancellation analysis.

## 4. Source correction: old equity + long-term debt − all cash proxy omitted financial debt

The 2025 full balance R6 reports:
- Shareholder equity **EUR 19,612.2m**.
- Short-term borrowings/current debt **EUR 1,681.9m**, noncurrent long-term debt **EUR 2,709.0m**.
- [2025 lease note R25](https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/R25.htm) **EUR 60.8m current + EUR 190.6m noncurrent** lease obligations (these are embedded in accrued liabilities).
- Cash and short-term investments **EUR 13,321.9m**.

The prior simple 2025 denominator `equity + noncurrent LT debt − cash/ST` was **EUR 8,999.3m**, but adding current debt and lease financing changes that arithmetic proxy to **EUR 10.933bn** (+EUR **1.9333bn**). **Even this corrected financing-side expression is NOT a fully defined invested capital.** It nets **all liquidity**, when some may be operational or related to advances, omits strategically financed assets and does not reconcile all lease effects and non-operating capital. It cannot be applied across quarters where current debt and lease amounts are undisclosed.

### 4.1 Capital denominator sensitivity — pure mathematical stress, NOT measured range

Hold NOPAT proxy EUR 12.631bn fixed (hypothetical 17% tax) and vary *hypothetical, unverified net invested capital*: 

| Assumed net operating capital (EUR bn) | NOPAT / capital | Status |
|---:|---:|---|
| 20.000 | 63.15% | Hypothetical denominator only |
| 25.000 | 50.52% | Hypothetical denominator only |
| 30.000 | 42.10% | Hypothetical denominator only |
| 35.000 | 36.09% | Hypothetical denominator only |
| 40.000 | 31.58% | Hypothetical denominator only |

These are **conditional calculations, not empirical ROIC estimates or lower/upper confidence limits**. The relevant purpose is to identify which balance-sheet classifications would materially shift the assessment; we have no defensible net-capital interval at this cutoff.

## 5. Quality controls and Research next action

**Reconciled:** five official quarterly US GAAP asset tables, operating EBIT arithmetic, the complete FY2025 current liability ledger and lease obligations, the exact distinction between FY2025 total contract liabilities and FY2026 Q3 *noncurrent-only* contract liability. Computed values are machine-checkable.

**Not certified:** Q3 2026 current contract liabilities; Q3 2026 current debt maturity/current lease liabilities/AP/accrual categories; operating vs non-operating cash and mixed other assets; economic commitment of strategic Zeiss loans; daily average invested capital; defensible tax on core EBIT. Historical 2025 client deposits are a **leading clue**, not a valid substitute for 2026 US GAAP denominators.

**Research action without blocking on G001:** keep the identified gross-asset productivity and capital-perimeter diagnostic as working evidence; continue operating quality, cash and cyclicality investigation. Seek post-Q3 published **US GAAP** line-by-line split of contract liabilities, debt and leases only when documented with source date ≤2026-10-30; otherwise mark **UNDISCLOSED_AT_CUTOFF** rather than endlessly repeating searches. Do not silently import IFRS numbers into this US GAAP model.

**Governance:** `G001/G006_INDEPENDENT_OF_OPERATING_CAPITAL_WORK`; `ROIC_UNCERTIFIED`; `PILOTAGE_GATE_NOT_EVALUATED`; `NO_REGISTRY_ARTIFACT`; `NO_FUNDAMENTALS_BOOTSTRAP`; `NO_FINAL_INVESTMENT_VERDICT`.
