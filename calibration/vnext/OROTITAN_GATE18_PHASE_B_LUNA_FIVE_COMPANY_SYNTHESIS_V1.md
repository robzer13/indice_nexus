# OROTITAN_GATE18_PHASE_B_LUNA_FIVE_COMPANY_SYNTHESIS_V1

**Gate:** 18  
**Phase:** B company calibration  
**Status:** SYNTHESIS COMPLETE — NOT FROZEN  
**Methodology change:** NO  
**Prompt/schema change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Model winner:** NONE  
**Routing freeze:** NONE  
**Further paid execution:** STOPPED PENDING EXPLICIT CONTRAST AUTHORIZATION  

## 1. Scope

This synthesis closes the planned five-company LUNA sweep under the controlled Gate 18 moat-evidence audit contract.

```text
module = MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3
prompt template = GATE18_MOAT_EVIDENCE_AUDIT_V0_3
generation schema = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_3
maxOutputTokens = 4096
model = openai/gpt-5.6-luna
reasoning = minimal
automatic retry = none
```

The five roles are functional calibration archetypes, not a statistical sample.

## 2. Company-level results

| Company | Role | Engineering valid | Comparison-admissible | Cost USD | Latency ms | Input | Output | Reasoning |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STMicroelectronics | CYCLICAL_SEGMENTED | PASS | PASS | 0.00176170 | 6,375 | 2,437 | 648 | 75 |
| RATIONAL AG | SIMPLE_CLEAN_COMPOUNDER | PASS | PASS | 0.00343145 | 7,014 | 8,948 | 683 | 56 |
| Constellation Software | SERIAL_ACQUIRER | PASS | PASS | 0.00179110 | 16,878 | 2,761 | 605 | 53 |
| Brookfield Corporation | ACCOUNTING_HEAVY | PASS | FAIL | 0.00189505 | 16,859 | 2,980 | 646 | 51 |
| Adyen | DIFFICULT_MOAT_CONFLICTING_EVIDENCE | PASS | FAIL | 0.00203380 | 6,073 | 3,439 | 666 | 55 |

Aggregate:

```text
engineering-valid observations = 5/5 = 100%
comparison-admissible observations = 3/5 = 60%
exact five-company LUNA cost = 0.01091310 USD
input tokens = 20,565
output tokens = 3,248
reasoning tokens = 290
total tokens = 23,813
aggregate latency = 53,199 ms
mean latency = 10,639.8 ms
```

The last rounded Vercel budget snapshot after Adyen remained:

```text
project limit = 4.00 USD
project displayed spend = 1.41 USD
team limit = 4.00 USD
team displayed spend = 1.41 USD
```

## 3. What LUNA did reliably

Across all five cases:

```text
schemaValid = true
semanticValid = true
finishReason = stop
retryCount = 0
provider cost evidence present
response hash present
```

This is a strong engineering-reliability result for the current runner and output contract.

Across the three admissible cases, LUNA also showed useful analytical discipline:

```text
STMicroelectronics:
- preserved scoped market-share and forward-looking distinctions
- retained SiC and MCU conflicts
- did not convert investment or customer relationships into unsupported moat certainty

RATIONAL:
- handled the largest packet in the sample
- retained dealer/service economics uncertainty
- separated workflow functionality from switching-cost proof
- separated resource savings from representative payback

Constellation Software:
- separated recurring revenue from permanent lock-in
- used independent replacement cases to bound longevity claims
- retained the consolidated retention/churn disclosure gap
- kept seller-reputation evidence below quantified sourcing proof
```

## 4. Human-quality failure mode

Brookfield and Adyen did not fail because of missing or malformed evidence IDs.

They failed because valid evidence IDs were assigned the wrong analytical role.

### Brookfield

The output placed E-039 and E-042 in `counterevidence_ids` for a finding that large alternative-asset platforms demonstrate replicability.

But:

```text
E-039 = KKR scaled insurance/alternative-platform evidence
E-042 = Brookfield participating alongside Apollo, BlackRock, Blackstone, Goldman Sachs and KKR in a large AI-infrastructure financing initiative
```

Those observations reinforce peer replicability / competing-capital evidence rather than counter it.

### Adyen

The output correctly stated that product uplift is reported but mainly issuer-measured or selected.

It then placed E-046 in `counterevidence_ids`.

But E-046 is itself another issuer-reported uplift observation with no independent audit. It reinforces the finding and its limitation rather than contradicting either.

### Failure class

```text
ENGINEERING ID VALIDITY = PASS
EXACT GROUNDING = PASS
EVIDENCE ROLE CLASSIFICATION = FAIL
COUNTEREVIDENCE SEMANTICS = FAIL
```

This distinction is material because the automated semantic validator currently verifies canonical references but does not establish that a cited item is correctly classified as support versus counterevidence.

## 5. Packet-density inference

The observed failures do not support a simple packet-size explanation.

```text
RATIONAL = 56 evidence items / 9 conflicts -> PASS
Constellation = 11 / 1 -> PASS
Brookfield = 12 / 2 -> FAIL quality
Adyen = 16 / 3 -> FAIL quality
```

RATIONAL was materially denser than every other non-ST case and remained admissible.

Therefore:

```text
packet density alone is not a sufficient explanation
```

The more plausible calibration hypothesis is that LUNA becomes less reliable when the task requires precise polarity/role classification of evidence in analytically ambiguous settings, especially:

```text
peer replicability versus differentiation
support versus counterevidence
reported benefit versus validation limitation
conflicting mechanisms that coexist rather than negate each other
```

This remains a hypothesis, not a frozen routing rule.

## 6. Relationship to higher-reasoning observations

STMicroelectronics showed:

```text
LUNA  = admissible
TERRA = admissible
SOL   = admissible
ASTRA = admissible
```

RATIONAL showed:

```text
LUNA  = admissible
TERRA = semantic failure
SOL   = semantic failure + incomplete
ASTRA = semantic failure + incomplete
```

Thus the current evidence does not support either of these simplistic rules:

```text
higher reasoning is always better
LUNA is always better
```

The next contrast must answer a specific calibration question rather than repeat a full four-model sweep.

## 7. Recommended next contrast

The highest-information-per-dollar next contrast is:

```text
Adyen x TERRA
role = DIFFICULT_MOAT_CONFLICTING_EVIDENCE
reasoning = medium
dry-run conservative ceiling = 0.061458 USD
```

Reason:

1. Adyen directly targets the failure dimension: conflicting moat evidence and support/counterevidence polarity.
2. TERRA is the lowest-cost higher-reasoning candidate.
3. Adyen's packet is much smaller than RATIONAL's, so the call helps distinguish model reliability from the dense-packet failure observed on RATIONAL.
4. The comparison asks one concrete question: does medium reasoning improve evidence-role classification without introducing semantic-ID failure?

The expected observation is not whether TERRA writes a longer answer. The acceptance question is:

```text
Does TERRA produce canonical references AND correctly classify support/counterevidence roles on Adyen?
```

## 8. Contrast stop rule

No TERRA/SOL/ASTRA call is authorized by this synthesis itself.

A separate explicit authorization record is required before spending.

If Adyen x TERRA is later authorized and:

```text
semanticValid = false
OR evidence-role classification fails
```

then no automatic retry and no automatic SOL/ASTRA escalation occurs.

If it passes, the result may justify one further targeted contrast, but still does not freeze routing.

## 9. Gate state

```text
Phase A = PASS
Phase B = IN PROGRESS
ST four-model benchmark = COMPLETE
RATIONAL four-model benchmark = COMPLETE
LUNA five-company sweep = COMPLETE
LUNA engineering-valid rate = 5/5
LUNA comparison-admissible rate = 3/5
next contrast proposed = Adyen x TERRA
next contrast authorized = NO
Gate 18 = IN PROGRESS / NOT FROZEN
model winner = NONE
routing freeze = NONE
```
