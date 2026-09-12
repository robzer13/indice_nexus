# RATIONAL AG — I4-B Fresh Reanalysis Pilot

## OroTitan Equity Research V1

**Status:** I4-B fresh reanalysis  
**Execution mode:** `ANALYZE`  
**Issuer:** RATIONAL AG / RATIONAL Aktiengesellschaft  
**Security:** RAA / Xetra / ISIN DE0007010803  
**Data cutoff:** 2026-09-12  
**Reference price date:** 2026-09-11  
**Reference price:** EUR 587.50  
**Research policy:** fresh evidence only. Prior/legacy OroTitan reports, legacy scores, legacy fair values and legacy verdicts were excluded as analytical inputs.

> This is an I4-B analyst research artifact. It is **not** a canonical Phase-4 `researchSnapshot`, does not call the I3-B persistence RPC, and does not move `research_dossiers.current_snapshot_id`.

---

# 1. Executive summary

RATIONAL is a very strong, unusually focused industrial franchise. The evidence supports a durable moat based on product/software differentiation, operator workflow standardisation, training/service/parts depth, reputation, and a direct consultative commercial model. The counterfactual is real rather than theoretical: Unox and other suppliers can deliver materially cheaper combi ovens, and in simpler kitchen workflows the economic case for paying RATIONAL's premium can disappear. The moat therefore does not rest on customer captivity alone; it rests on a premium product and process proposition that must keep earning its premium.

The business economics remain exceptional. FY2025 sales were EUR 1,259.6m, EBIT EUR 332.6m and net income EUR 253.8m. Operating cash flow was EUR 253.1m and company-defined free cash flow EUR 219.4m. The balance sheet carried EUR 539.4m of cash, cash equivalents and deposits, no residual long-term financing liabilities, and an 80% equity ratio. H1 2026 sales rose 6% reported / 8% organically to EUR 641.5m and EBIT rose 11% to EUR 169.9m, although the reported 26.5% EBIT margin benefited from a roughly EUR 14m US tariff reimbursement; without it, the H1 margin was roughly 24.3%.

The principal current weakness is not business quality but the price-to-return setup. At EUR 587.50, a central five-year shareholder-return model produces about 8.8% annualised return with a 24x terminal P/E and about 7.9% with a 23x mature-normalisation terminal P/E. A fresh DCF produces a broad bear/base/bull range of approximately EUR 365 / EUR 580 / EUR 725 per share. This implies no robust margin of safety at the reference price.

The provisional quality score is `OQS = 85.5` (`VERY_STRONG`). Using a **provisional**, not-yet-policy-locked required return `H = 10%`, the valuation score is about `OVS = 54.6` and the constrained Investment Score about `69.6` (`UNATTRACTIVE`). At `H = 12%`, those fall to about `38.4` and `53.4` respectively. The configured required-return policy must therefore be locked before a canonical snapshot is generated.

**Current conclusion:** high-quality Prepared opportunity, but not an OroTitan configuration and not yet operationally READY for canonical activation because the required-return policy and reference-price source convention are not yet locked.

---

# 2. Certification status

```text
BUSINESS_RESEARCH_STATUS      = CERTIFIED
INVESTMENT_CONCLUSION_STATUS  = CERTIFIED_WITH_LIMITATIONS
SCORE_PERMISSION              = ALLOWED

FORENSIC_RELIABILITY          = CLEAN
VALUATION_RELIABILITY         = MEDIUM

HARD_BLOCKERS                 = NONE
RED_TEAM_COMPLETED            = YES
PRE_MORTEM_COMPLETED          = YES
```

Material limitations:

1. `CONFIGURED_REQUIRED_RETURN` is not yet a production-locked policy input. `H = 10%` is used only as a provisional pilot setting; `H = 12%` is shown as sensitivity.
2. Public market-data vendors disagree immaterially on the exact 11 September 2026 Xetra close (roughly EUR 584.5–588.5). EUR 587.50 is the locked pilot reference price. Before canonical persistence, the project's authoritative market-price source must be selected/reconciled.
3. Exact 30 June 2026 all-in excess liquidity including deposits is not extracted in this pilot from a primary H1 balance-sheet table. Valuation uses a deliberately bounded excess-cash assumption, and this is reflected in `VALUATION_RELIABILITY = MEDIUM`.
4. Several market-share and penetration claims originate from the issuer. They are not treated as independent moat proof. Independent evidence is used for competitive structure and substitution.

Hard-blocker audit:

```text
B1 IDENTITY LOCK FAILURE                         = PASS / NOT TRIGGERED
B2 MATERIAL CALCULATION FAILURE                  = PASS / NOT TRIGGERED
B3 MATERIAL EVIDENCE INTEGRITY FAILURE           = PASS / NOT TRIGGERED
B4 FORENSIC RELIABILITY FAIL                     = PASS / NOT TRIGGERED
B5 CRITICAL DEFINITION / METHOD FAILURE          = PASS / NOT TRIGGERED
B6 CRITICAL INFORMATION GAP                      = PASS / NOT TRIGGERED
B7 MATERIAL CROSS-BLOCK CONTRADICTION            = PASS / NOT TRIGGERED
B8 MATERIAL PIT / VERSION / FRESHNESS FAILURE    = PASS / NOT TRIGGERED
```

---

# 3. Identity / data lock

```text
ISSUER_DISPLAY_NAME       = RATIONAL AG
ISSUER_LEGAL_NAME         = RATIONAL Aktiengesellschaft
ISSUER_COUNTRY            = Germany
REPORTING_CURRENCY        = EUR

TICKER                    = RAA
ISIN                      = DE0007010803
PRIMARY_EXCHANGE          = Xetra / Frankfurt regulated market context
TRADING_CURRENCY          = EUR
ECONOMIC_SHARE_COUNT      = 11,370,000

DATA_CUTOFF               = 2026-09-12
REFERENCE_PRICE_DATE      = 2026-09-11
REFERENCE_PRICE           = EUR 587.50
MARKET_CAP                = EUR 6,679.875m
```

RATIONAL's August 2026 corporate announcement independently reconfirms the issuer identity, ISIN and Frankfurt Prime Standard listing. The company investor-relations material states 11.37 million shares.

Price-source conflict: multiple public Xetra-labelled vendors report slightly different 11 September observations. The spread is below 1% and does not change any quality, valuation-class or decision conclusion in this pilot. It remains a source-governance item to resolve before I4-C.

---

# 4. Business model

RATIONAL develops, manufactures and sells professional cooking systems for commercial kitchens. Its economic model is concentrated rather than conglomerate-like. iCombi is the dominant product family; iVario is smaller but currently growing faster. The company also sells accessories, care products, service parts, training and technical services.

FY2025:

```text
Sales                       EUR 1,259.6m
EBIT                        EUR   332.6m
EBIT margin                       26.4%
Net income                  EUR   253.8m
R&D expense                 EUR    75.8m
R&D / sales                        ~6.0%
```

H1 2026:

```text
Sales                       EUR 641.5m   +6% reported / +8% organic
Gross margin                     59.8%
EBIT                        EUR 169.9m   +11%
Reported EBIT margin             26.5%
EBIT margin ex tariff refund     ~24.3%
iCombi sales                EUR 562.2m   +5%
iVario sales                EUR  79.4m   +14%
```

The H1 margin must not be extrapolated mechanically because the roughly EUR 14m US tariff reimbursement is non-recurring. Management maintained mid- to high-single-digit revenue growth guidance and a 25–26% full-year EBIT-margin range.

---

# 5. Economic quality synthesis

The franchise combines:

- high gross margins near 60%;
- mid-20s operating margins through different economic environments;
- low physical capital intensity relative to earnings;
- high organic reinvestment through R&D, sales/service coverage and product development;
- no structural reliance on financial leverage;
- high cash conversion over a multi-year period;
- a concentrated category focus that allows deep product/process specialisation.

The main economic caveat is that a material part of growth investment is expensed through R&D and customer-facing operating costs. A simple tangible invested-capital denominator therefore overstates precision in marginal-return analysis. The economic conclusion remains strong, but the ROIIC proof is deliberately downgraded for interpretability/attribution rather than rewarded for an explosive denominator.

---

# 6. Moat proof

```text
PRIMARY_MOAT             = product/process differentiation + workflow ecosystem + service/brand depth
MOAT_EVIDENCE_STATE      = STRONGLY_SUPPORTED
MOAT_TREND               = STABLE
MOAT_DURABILITY          = LONG
NEGATIVE_EVIDENCE_SEARCH = COMPLETE
MOAT_SCORE               = 90
MOAT_ELITE               = PASS
```

Causal chain:

```text
specialised cooking know-how + sustained R&D
→ differentiated automation / consistency / usability
→ operator and process standardisation
→ training, parts, service and installed workflow familiarity
→ customer productivity / reliability benefits
→ ability to sustain a premium price and high margins
→ high returns and reinvestment capacity
```

Independent / behavioural evidence:

- An independent commercial service operator that services both RATIONAL and Unox describes RATIONAL as materially more expensive but superior in cooking precision, operator interface and parts depth. It explicitly says Unox can produce comparable plates for simpler steam/regen/roast use cases. This is valuable because it proves both the source of the premium and the existence of a genuine substitute.
- A California Energy Commission proceeding identifies RATIONAL, Alto-Shaam and Unox as the largest US combi-oven brands and expects RATIONAL to account for more than half of California oven sales. It also notes that combi ovens themselves cost materially more than conventional convection ovens, confirming that customers must perceive economic utility to adopt the category.
- RATIONAL spent EUR 75.8m, roughly 6% of sales, on R&D in 2025 and reports more than 600 patents, patent applications and registered designs. Patents are not treated as moat proof by themselves; they support the observed innovation mechanism.

Counterfactual / substitution test:

The moat is not absolute. Unox and other competitors can be much cheaper. If a kitchen mainly needs basic steam/regeneration/roasting, the premium can be uneconomic. China provides live negative evidence: Yum China has increasingly sourced locally, and local manufacturers compete at lower prices. This means RATIONAL's moat requires continued product/process differentiation and service value; it is not a lock-in monopoly.

Why `STRONGLY_SUPPORTED` despite substitution: the competing products demonstrate a valid counterfactual, but the evidence still supports durable differentiation in more complex/high-utilisation workflows, while group margins, innovation intensity, installed workflow benefits and global customer economics remain consistent with that mechanism.

---

# 7. Growth / runway

```text
RUNWAY_EVIDENCE_STATE = SUPPORTED
RUNWAY_MAGNITUDE      = LARGE
RUNWAY_HORIZON        = EXTENDED
RUNWAY_SCORE          = 85
RUNWAY_ELITE          = FAIL
```

Core drivers:

1. continued conversion from conventional professional cooking equipment toward multifunctional/combi systems;
2. geographic penetration, especially North America and selected underpenetrated markets;
3. faster iVario growth and broader product-family cross-selling;
4. replacement/upgrade cycles and monetisation of service, accessories, care products and parts;
5. labour, energy and kitchen-space pressure increasing the value of automation and throughput.

Evidence:

- H1 2026 organic sales grew 8% despite weak China.
- North America grew slightly reported but more than 10% organically in H1 due FX translation.
- iVario grew 14% in H1 2026, materially faster than iCombi.
- Independent Morningstar research continues to describe a lengthy runway despite RATIONAL's already high category share.

Constraints:

- RATIONAL already has a very high issuer-reported global share, limiting share-gain arithmetic.
- China shows that local lower-cost substitution can bind.
- premium pricing creates a ceiling where use cases do not require RATIONAL's full functionality.
- continued growth requires R&D, sales/service coverage and selected capacity investments; it is not capital-free.

Why not Elite: the evidence is strong enough for `SUPPORTED`, but the most aggressive penetration/TAM claims remain substantially issuer-originated, and China is a live example of a competitive constraint. The frozen Elite gate requires `RUNWAY_EVIDENCE_STATE = STRONGLY_SUPPORTED`; that bar is not met.

---

# 8. Return quality

```text
ECONOMIC_SPREAD       = CLEARLY_POSITIVE
STANDARD_ROIC         = ~48% conservative operating approximation
COMPANY_ROCE_2025     = 36.7%
ROIIC_3Y_DIAGNOSTIC   = ~90–100% tangible-capital diagnostic
INTERPRETABILITY      = INTERPRETABLE_WITH_LIMITATIONS
ATTRIBUTABILITY       = MEDIUM
RETURN_TREND          = STABLE
RETURN_QUALITY_SCORE  = 85
RETURN_QUALITY_ELITE  = FAIL
```

Operating ROIC approximation:

```text
2025 normalized NOPAT
≈ EBIT 332.6 × (1 - 25.6%)
≈ EUR 247.4m

Operating IC proxy
≈ working capital + fixed assets
2024 ≈ 245 + 249 = EUR 494m
2025 ≈ 283 + 251 = EUR 534m
Average ≈ EUR 514m

STANDARD_ROIC ≈ 247.4 / 514 ≈ 48%
```

This is intentionally presented as an approximation/range rather than false precision. RATIONAL's own ROCE was 36.7% in 2025, but its denominator includes average equity and financing items and is not identical to the OroTitan operating-invested-capital definition.

A three-year tangible-capital ROIIC diagnostic is extremely high because operating profit rose materially while tangible operating capital rose relatively little. However, R&D and sales-force growth investment are expensed, so a denominator based only on booked operating assets understates full economic reinvestment. Accordingly, marginal returns are judged strong but not terminal-Elite proof.

---

# 9. FCF / forensic

```text
FORENSIC_RELIABILITY = CLEAN
CASH_ECONOMICS_SCORE = 85
CASH_ECONOMICS_ELITE = PASS
```

FY2025:

```text
Operating cash flow            EUR 253.1m
Cash capex                     EUR  33.7m
Standardized FCF               EUR 219.4m
FCF margin                          17.4%
Cash + equivalents + deposits  EUR 539.4m
Long-term financing residuals       none
```

FY2024 FCF was EUR 251.4m. Multi-year cash generation is structurally strong; 2025 was temporarily reduced by inventory, receivables and tax-payment effects. There is no material SBC/dilution leakage and no acquisition-adjustment machinery obscuring cash generation.

A normalized 2026 owner-earnings range of roughly EUR 230–260m is supportable for pilot valuation. This is a range rather than a point because the exact maintenance/growth split of current facility investment is not separately disclosed. The whole range remains consistent with high cash realization relative to operating profit.

---

# 10. Capital allocation

```text
CAPITAL_ALLOCATION_SCORE = 85
CAPITAL_ALLOCATION_ELITE = FAIL
```

Positive evidence:

- growth is financed internally;
- no material bank-debt dependence;
- R&D remains around 6% of sales and is expensed;
- organic capacity, product development and customer-facing capabilities have supported high returns;
- ordinary dividend policy has historically returned a substantial share of earnings;
- 2025 proposal combined a EUR 16 regular dividend with a EUR 4 special dividend because liquidity was high;
- no structural share issuance or M&A roll-up obscures per-share economics.

Why terminal Elite fails: the balance sheet retains very substantial liquidity by explicit policy (`security before return`). That policy is defensible and strengthens resilience, but the terminal Elite allocation gate asks for repeated proof that capital generated is deployed into strong value-creating returns across material uses. The large excess-liquidity pool prevents us from calling the entire capital-allocation system Elite.

---

# 11. Management / governance

```text
MANAGEMENT_GOVERNANCE_SCORE = 80
MANAGEMENT_GOVERNANCE_ELITE = FAIL
```

Strengths:

- strategic consistency around product specialisation, R&D and financial independence;
- management guidance is framed around sales growth and operating margin rather than promotional adjusted metrics;
- conservative funding policy and long history of shareholder distributions;
- no material related-party transaction concern identified in the fresh review.

Governance limitations:

- co-founder Walter Kurtz has served on the Supervisory Board since 1998 and chairs it; the company itself says he is not independent;
- RATIONAL explicitly departs from German Code recommendations on a nomination committee, an age limit and publication of Supervisory Board rules;
- family/founder influence is material and free float is constrained;
- founder-family continuity can support long-termism but does not satisfy the stricter terminal Elite requirement for demonstrably strong minority-governance architecture.

---

# 12. Outside view / base rates

The outside view is favourable but not a substitute for primary analysis.

- Independent Morningstar research maintained a EUR 650 fair value estimate after Q1 2026 and in August described the shares as fairly valued while continuing to see a lengthy runway.
- Independent service evidence shows a real premium over Unox, not an imagined one, but also demonstrates that a cheaper substitute is economically adequate for simpler use cases.
- Public market data show a material 2026 de-rating from the year's highs, but a lower stock price is not itself evidence of undervaluation.

The relevant industrial base-rate risk is duration: a high-quality niche leader can remain an excellent company while shareholder returns disappoint if growth fades and the terminal multiple normalises.

---

# 13. Risks / resilience

```text
RESILIENCE_RISK_SCORE = 85
RESILIENCE_ELITE      = PASS
MATERIAL_WEAK_LINK    = NO
```

Key risks and transmission mechanisms:

1. **China / local substitution.** Local suppliers and customer localisation can reduce share and pricing. Yum China is live evidence. Mitigation: diversified geography, locally adapted iCombi One, local sales/dealer expansion.
2. **Premium-price substitution.** Unox and others are materially cheaper. If automation/service differentiation narrows, premium economics can compress.
3. **US tariffs / input inflation.** Management indicated roughly EUR 28–29m tariff burden at current volume assumptions and elevated steel/electronic/logistics costs. Price increases and efficiency are the mitigation, not certainty.
4. **Sales-force productivity.** Growth requires customer-facing investment. If incremental sales productivity falls, ROIIC fades before accounting ROIC necessarily reveals it.
5. **Working capital / capacity.** Inventories and receivables reduced 2025 cash conversion; new facilities can temporarily depress FCF.
6. **Governance / low float.** Concentrated control can reduce minority influence and limits buyback flexibility.
7. **Valuation duration.** At roughly 25x pilot 2026E EPS, even good operating delivery can produce sub-10% shareholder returns if the multiple normalises.

None is currently judged an unresolved material weak link capable of irreversible thesis damage without adequate mitigation. Several are nevertheless explicit invalidation monitors.

---

# 14. Red Team / pre-mortem

Five-year failure scenario:

```text
Revenue growth falls to 3–4%
+ China/value-tier competition persists
+ US tariffs/input costs hold operating margin around 22–23%
+ category penetration matures faster than expected
+ terminal P/E falls to 18–20x
→ strong company, poor entry price, weak shareholder return
```

What would falsify the quality thesis rather than only the valuation thesis:

- sustained loss of premium pricing without offsetting unit growth;
- repeated share loss in mature core markets, not only China localisation;
- gross margin deterioration not explained by temporary tariffs/FX/input shocks;
- persistent sales-force/R&D growth with weak incremental revenue/profit;
- FCF conversion structurally falling despite stable reported EBIT;
- evidence that independent users no longer perceive material operational superiority over lower-cost alternatives.

---

# 15. Valuation / expected return

## 15.1 Locked pilot inputs

```text
Reference price                  EUR 587.50
Shares                           11.37m
Market cap                       EUR 6,679.9m
2026E sales                      EUR 1,341.4m
2026E EPS                        EUR 23.60
2026E normalized FCF             EUR 255m central
German 10Y reference             ~3.50%
```

2026E figures are analyst estimates, not company guidance. The revenue assumption is consistent with management's mid- to high-single-digit growth guidance. The EPS and normalized FCF estimates are constructed from the fresh business/cash analysis.

## 15.2 DCF

```text
Bear FV   ≈ EUR 365/share
Base FV   ≈ EUR 580/share
Bull FV   ≈ EUR 725/share
```

Base assumptions:

- 2027–2036 revenue growth fades from 7% toward 3%;
- FCF margin rises modestly from 19% toward 20%;
- 8.0% discount rate;
- 2.5% terminal growth;
- EUR 400m excess-financial-liquidity bridge.

Bear assumptions:

- 4% growth fading toward 2%;
- 17.5–18% FCF margin;
- 9.0% discount rate;
- 2.0% terminal growth;
- EUR 350m excess cash.

Bull assumptions:

- 9% initial growth fading toward 3.5%;
- 19.5–20.5% FCF margin;
- 7.5% discount rate;
- 2.5% terminal growth;
- EUR 450m excess cash.

Cross-check: Morningstar maintained EUR 650 fair value in May 2026 and described the shares as fairly valued following H1 2026. This supports the broad order of magnitude but does not determine OroTitan fair value.

## 15.3 Expected return

Five-year primary model:

```text
2026E EPS                  EUR 23.60
2027–2031 EPS growth       7.5%, 7.0%, 6.5%, 6.0%, 5.5%
Ordinary payout assumption 70%
Terminal P/E primary       24x
Terminal P/E normalized    23x

PRIMARY_EXPECTED_RETURN_5Y       ≈ 8.8%
MATURE_NORMALIZATION_RETURN_5Y   ≈ 7.9%
SAME_MULTIPLE_RETURN_5Y (~24.9x) ≈ 9.5%
```

At the current price, return quality is therefore highly dependent on continuing business execution and retaining a premium terminal multiple. The investment does not require multiple expansion, but it also does not offer much return headroom.

## 15.4 Margin of safety

```text
MARGIN_OF_SAFETY = NONE
```

The reference price is essentially at the fresh DCF base value, materially above the bear value, and the mature-normalisation expected return is below the provisional 10% hurdle. A wide range of plausible assumptions still produces a valuable business; that is not the same as a robust entry-price margin of safety.

## 15.5 Valuation reliability

```text
VALUATION_RELIABILITY = MEDIUM
```

The operating business is visible, but value remains sensitive to terminal multiple/growth, long-duration margin assumptions, and the exact excess-liquidity bridge. The current German 10-year government yield around 3.50% also argues against an unrealistically low discount rate.

---

# 16. Scoring

## 16.1 Business quality

| Dimension | Score |
|---|---:|
| MOAT | 90 |
| RUNWAY | 85 |
| RETURN_QUALITY | 85 |
| CASH_ECONOMICS | 85 |
| CAPITAL_ALLOCATION | 85 |
| MANAGEMENT_GOVERNANCE | 80 |
| RESILIENCE_RISK | 85 |

```text
OQS_RAW
= 0.20×90 + 0.15×85 + 0.20×85 + 0.10×85
+ 0.15×85 + 0.10×80 + 0.10×85
= 85.5

WEAK_LINK_CAP = min(80) + 25 = 105 → capped at 100
OQS = min(85.5, 100) = 85.5
QUALITY_CLASS = VERY_STRONG
```

## 16.2 Valuation score — provisional H = 10%

```text
PRIMARY_EXPECTED_RETURN      = 8.788%
MATURE_NORMALIZATION_RETURN  = 7.944%

C = expected-return score(primary vs H) ≈ 57.88
N = expected-return score(normalized vs H) ≈ 49.58

RETURN_COMPONENT
= min(0.60×C + 0.40×N, N+15)
≈ 54.56

MOS_CAP (NONE)               = 55
VALUATION_RELIABILITY_CAP    = 95  (MEDIUM)

OVS = 54.56
```

## 16.3 Investment score — provisional H = 10%

```text
INVESTMENT_RAW
= 0.70×85.5 + 0.30×54.56
≈ 76.22

INVESTMENT_SCORE
= min(76.22, 85.5, 54.56+15)
≈ 69.56

INVESTMENT_CLASS = UNATTRACTIVE
```

## 16.4 Required-return sensitivity

At provisional `H = 12%`:

```text
OVS              ≈ 38.38
INVESTMENT_SCORE ≈ 53.38
INVESTMENT_CLASS = UNATTRACTIVE
```

This sensitivity is decision-relevant. The production required-return policy must be fixed before I4-C.

---

# 17. OroTitan terminal gate

```text
CERTIFICATION_GATE            = FAIL
MOAT_ELITE                     = PASS
RUNWAY_ELITE                   = FAIL
RETURN_QUALITY_ELITE           = FAIL
CASH_ECONOMICS_ELITE           = PASS
CAPITAL_ALLOCATION_ELITE       = FAIL
MANAGEMENT_GOVERNANCE_ELITE    = FAIL
RESILIENCE_ELITE               = PASS
MATERIAL_WEAK_LINK_GATE        = PASS   (MATERIAL_WEAK_LINK = NO)
VALUATION_ELITE                = FAIL

OROTITAN_STATUS                = NO
```

Rationale:

- Certification fails terminally because the investment conclusion is `CERTIFIED_WITH_LIMITATIONS`, not fully `CERTIFIED`.
- Runway is `SUPPORTED`, not `STRONGLY_SUPPORTED`.
- Marginal return is economically excellent but does not have the high-attribution / strongly supported proof required for Elite ROIIC.
- Capital allocation is strong but very large deliberate liquidity retention prevents a clean Elite deployment verdict.
- Governance does not meet the strict independence/minority-alignment Elite standard.
- Valuation is not Elite: expected return is below the provisional hurdle, mature-normalisation return is not strong, MOS is `NONE`, and valuation reliability is `MEDIUM`.

Decision-useful gaps:

```text
OROTITAN_GAPS = [
  RESEARCH_NOT_FULLY_CERTIFIED,
  RUNWAY_NOT_ELITE,
  RETURN_QUALITY_NOT_ELITE,
  CAPITAL_ALLOCATION_NOT_ELITE,
  MANAGEMENT_GOVERNANCE_NOT_ELITE,
  VALUATION_NOT_ELITE
]
```

---

# 18. Price ladder / readiness

Primary five-year model, 24x terminal P/E:

| Required return | Maximum entry price |
|---:|---:|
| 8% | EUR 608 |
| 10% | EUR 558 |
| 12% | EUR 512 |
| 15% | EUR 452 |

Mature-normalisation overlay, 23x terminal P/E:

| Required return | Maximum entry price |
|---:|---:|
| 8% | EUR 586 |
| 10% | EUR 537 |
| 12% | EUR 494 |
| 15% | EUR 436 |

With provisional `H = 10%`:

```text
PRICE_FOR_REQUIRED_RETURN_H  ≈ EUR 558
PRICE_FOR_STRONG_RETURN      ≈ EUR 512   (12% analytical level)
PRICE_FOR_EXCEPTIONAL_RETURN ≈ EUR 452   (15% analytical level)
```

Operational state:

```text
DOSSIER_READINESS              = PARTIALLY_READY
OPPORTUNITY_PATH               = PREPARED
PRICE_LADDER_STATUS            = CURRENT_FOR_PILOT / NOT YET CANONICAL
NEXT_ACTION                    = WAIT_FOR_PRICE

INVESTABLE_PRICE_ZONE          ≲ EUR 558 under provisional H=10%
STRONG_OPPORTUNITY_ZONE        ≲ EUR 512
POTENTIAL_OROTITAN_PRICE_ZONE  = NOT_AVAILABLE
```

`POTENTIAL_OROTITAN_PRICE_ZONE` is unavailable because non-valuation Elite gates already fail; price cannot repair those gates.

Why `PARTIALLY_READY` instead of `READY`: the full Deep Dive is substantially complete, but the configured required-return policy and canonical market-price source convention are operational inputs needed for rapid activation and I4-C canonical projection.

---

# 19. Final verdict / next action

```text
BUSINESS QUALITY        = VERY_STRONG
OQS                     = 85.5

CURRENT INVESTMENT SETUP
OVS (H=10% provisional) = 54.6
INVESTMENT_SCORE        = 69.6
INVESTMENT_CLASS        = UNATTRACTIVE

OROTITAN_STATUS         = NO
DOSSIER_READINESS       = PARTIALLY_READY
OPPORTUNITY_PATH        = PREPARED
NEXT_ACTION             = WAIT_FOR_PRICE
```

RATIONAL passes the test that matters most for the I4-B pilot: the fresh method can separate an excellent business from an inadequate current price without degrading the business-quality judgment. It also demonstrates why the terminal OroTitan state cannot be inferred from an OQS in the mid-80s.

Before I4-C canonical snapshot generation:

1. ratify/configure the `REQUIRED_RETURN_H` policy to be used by I2/OVS;
2. select the canonical current-market-price source and resolve the small 11 September vendor discrepancy;
3. optionally extract/reconcile exact H1 2026 all-in excess liquidity from a primary interim balance-sheet table to tighten valuation reliability;
4. rerun deterministic valuation/scoring from the locked inputs;
5. only then map the certified dossier into the patched Phase-4 schema and validate it through I3-B.

No canonical snapshot should be persisted from this I4-B artifact alone.

---

# 20. Evidence ledger

All source use below is fresh to this reanalysis. No prior internal RATIONAL investment report is an evidence item.

| ID | Class | Root source | Main claim fit |
|---|---|---|---|
| E01 | PRIMARY | RATIONAL FY2025 Annual Report, https://www.rational-online.com/media/investor-relations/veroeffentlichungen-gj-2025/rational-ag---annual-report-fy-2025-%28single-pages%29.pdf | FY2025 financials, FCF, liquidity, R&D, ROCE, dividend, capital structure |
| E02 | PRIMARY | RATIONAL H1 2026 corporate news, https://www.rational-online.com/en_xx/company/investor-relations/announcements/ir-full-news/index.php?feedId=b8dae00f-d530-4d9c-80e5-01e2c698c68d&feed_template=news&format=html&id=9f336b2c-811d-406c-b7c7-18e2ab5076cc&lang=en&view=detail | H1 sales, EBIT, product/geographic growth, guidance, tariff refund |
| E03 | PRIMARY | RATIONAL Q1 2026 corporate news, https://www.rational-online.com/en_ca/company/investor-relations/announcements/ir-full-news/index.php?feedId=b8dae00f-d530-4d9c-80e5-01e2c698c68d&feed_template=news&format=html&id=aec666d1-3e82-4751-bc2a-d022c8e67d10&lang=en&view=detail | Q1 trajectory, service offering, issuer market-share claim |
| E04 | PRIMARY | RATIONAL FY2025 Declaration of Corporate Governance, https://www.rational-online.com/media/investor-relations/corporate-gouvernance/rational-ag---declaration-of-corporate-governance-fy-2025.pdf | Board independence and governance-code deviations |
| E05 | TRANSCRIPT / MANAGEMENT | H1 2026 call transcript, https://uk.investing.com/news/transcripts/earnings-call-transcript-rational-posts-solid-h1-2026-growth-as-china-weakens-93CH-4816302 | tariff normalization, China/Yum/local competition, dividend policy, cost outlook |
| E06 | INDEPENDENT OPERATING | Berne Commercial RATIONAL vs Unox, https://www.berne-commercial.com/compare/rational-vs-unox | substitution, price gap, product/service differentiation |
| E07 | INDEPENDENT REGULATORY/INDUSTRY | California Energy Commission filing, https://efiling.energy.ca.gov/GetDocument.aspx?DocumentContentId=96480&tn=260264 | US competitive structure and category pricing |
| E08 | INDEPENDENT ANALYST | Morningstar May/Aug 2026 RATIONAL notes, https://www.morningstar.com/company-reports/1480029-keeping-rationals-fair-value-estimate-at-eur-650-following-a-record-first-quarter | outside-view fair value and runway cross-check |
| E09 | MARKET DATA | Investing Xetra history, https://ng.investing.com/equities/rational-ag-historical-data?cid=1246601 | pilot reference price; vendor discrepancy retained as limitation |
| E10 | MARKET / RISK-FREE CROSS-CHECK | Germany 10Y historical data, https://ng.investing.com/rates-bonds/germany-10-year-bond-yield-historical-data | current euro risk-free environment |

Issuer-originated global-share and penetration claims are explicitly not promoted into independent proof merely because external analyst commentary repeats them.

---

# 21. Calculation references

```text
C01 MARKET_CAP
= 587.50 × 11.37m
= EUR 6,679.875m

C02 2025 NOPAT APPROX
= 332.6 × (1 - 25.6%)
≈ EUR 247.4m

C03 STANDARD_ROIC APPROX
= 247.4 / ((494 + 534)/2)
≈ 48%

C04 OQS_RAW
= 85.5

C05 WEAK_LINK_CAP
= min(90,85,85,85,85,80,85) + 25
= 105 → 100

C06 OQS
= min(85.5,100)
= 85.5

C07 EXPECTED_RETURN_5Y_PRIMARY
= IRR(entry 587.50, 70% payout, EPS path, terminal 24x)
≈ 8.788%

C08 MATURE_NORMALIZATION_RETURN_5Y
= same model, terminal 23x
≈ 7.944%

C09 OVS at provisional H=10%
≈ 54.56

C10 INVESTMENT_SCORE
= min(0.70×85.5 + 0.30×54.56, 85.5, 54.56+15)
≈ 69.56
```

---

# 22. Version lock

```text
DISCOVERY_VERSION          = DISCOVERY_FREEZE_v1.0
MOAT_VERSION               = MOAT_PROOF_FREEZE_v1.0
RUNWAY_VERSION             = RUNWAY_FREEZE_v1.0
ROIC_VERSION               = ROIC_FREEZE_v1.0
FCF_VERSION                = FCF_FORENSIC_FREEZE_v1.0
VALUATION_VERSION          = VALUATION_FREEZE_v1.0
CERTIFICATION_VERSION      = RESEARCH_CERTIFICATION_FREEZE_v1.0
SCORING_VERSION            = SCORING_FREEZE_v1.0
METHOD_VERSION             = OROTITAN_EQUITY_RESEARCH_V1
REPORT_VERSION             = RATIONAL_I4B_FRESH_v0.1
EVIDENCE_LEDGER_VERSION    = RATIONAL_2026-09-12_EVIDENCE_v0.1
CALCULATION_VERSION        = RATIONAL_2026-09-12_CALC_v0.1
DATA_CUTOFF                = 2026-09-12
```
