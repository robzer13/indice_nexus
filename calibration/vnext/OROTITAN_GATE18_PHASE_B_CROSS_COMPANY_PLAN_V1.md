# OROTITAN_GATE18_PHASE_B_CROSS_COMPANY_PLAN_V1

**Gate:** 18  
**Phase:** B company calibration  
**Status:** EXECUTION PLAN — NOT FROZEN  
**Methodology change:** NO  
**Prompt/schema change:** NO  
**Physical model identity change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Routing freeze:** NONE  
**Model winner:** NONE  

## 0. Trigger

The first complete four-model company benchmark is now available for:

```text
STMicroelectronics
role = CYCLICAL_SEGMENTED
LUNA  = comparison-admissible
TERRA = comparison-admissible
SOL   = comparison-admissible
ASTRA = comparison-admissible
```

The STMicroelectronics case establishes that all four physical candidates can produce admissible outputs under the controlled Phase B execution profiles.

It does not establish cross-company generalization, a routing winner, or a quality-cost rule.

## 1. Remaining pinned pilot cases

The original Gate 18 pilot contains five functionally diversified companies.

The four remaining companies are executed without changing the pinned pilot:

```text
RATIONAL AG
role = SIMPLE_CLEAN_COMPOUNDER

Constellation Software
role = SERIAL_ACQUIRER

Brookfield Corporation
role = ACCOUNTING_HEAVY

Adyen
role = DIFFICULT_MOAT_CONFLICTING_EVIDENCE
```

STMicroelectronics remains the completed fifth role:

```text
STMicroelectronics
role = CYCLICAL_SEGMENTED
```

This is a functional calibration sample only. It is not statistical validation.

## 2. Comparison contract

Every remaining paid observation must preserve the current controlled contract:

```text
module = MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3
scope = MOAT_INPUTS
prompt template version = 0.3
GenerationSchema version = 0.3
maxOutputTokens = 4096

LUNA  reasoning = minimal
TERRA reasoning = medium
SOL   reasoning = medium
ASTRA reasoning = high
```

For a given company, the following must remain identical across candidates:

```text
company
DATA_CUTOFF
verified Evidence Packet
module question
system instruction
prompt/schema hashes
max-output contract
source availability
tool availability
```

No per-model packet, prompt, schema or output-cap exception is allowed.

## 3. Dry-run gate

Before any new paid inference, all four remaining cases must be dry-run locally with no model call.

The dry-runs must expose:

```text
case identity
role
DATA_CUTOFF
source evidence count
source conflict count
filtered evidence count
filtered conflict count
packet SHA-256
evidence-ledger SHA-256
conflict-ledger SHA-256
serialized packet size
selected model identities
reasoning profiles
live conservative cost ceilings
```

A case is not eligible for paid execution if:

```text
artifact verification fails
case identity is ambiguous
packet construction fails
prompt/schema hash differs
the pinned artifact lineage cannot be verified
```

## 4. Spend discipline

Budget snapshot after the completed STMicroelectronics four-model benchmark:

```text
project monthly limit = 4.00 USD
project displayed spend = 1.07 USD
team monthly limit = 4.00 USD
team displayed spend = 1.07 USD
```

The Vercel CLI budget display is rounded and is not exact per-call cost evidence.

Before authorizing the first new paid call:

```text
SUM(all remaining dry-run conservative ceilings)
<= 2.00 USD
```

If the aggregate conservative ceiling exceeds 2.00 USD, execution stops and the plan must be revised before spending.

This leaves a minimum planning reserve against the 4.00 USD monthly project budget.

Every paid call remains:

```text
one company
one physical model
explicit --model
explicit --max-case-spend-usd
no --allow-multi-model
no automatic retry
private gitignored provider output
```

A failed provider call is recorded as an observation. It is not automatically rerun.

## 5. Paid execution order

Subject to the dry-run gate and spend guard, companies are executed in increasing analytical difficulty:

```text
1. RATIONAL AG
2. Constellation Software
3. Brookfield Corporation
4. Adyen
```

Within each company:

```text
1. LUNA
2. TERRA
3. SOL
4. ASTRA
```

The order is operational only. It is not a ranking of models.

A later model call must not be used to overwrite or reinterpret the earlier model's raw observation.

## 6. Observation admissibility

Each model observation requires engineering checks:

```text
schemaValid = true
semanticValid = true
finishReason = stop
retryCount = 0 unless separately explained
provider cost evidence present
response hash present
```

Human quality adjudication remains separate from engineering validity and checks:

```text
narrative completeness
grounding to supplied E-/C- references
material conflict handling
counter-evidence treatment
weak-link usefulness
unresolved-point usefulness
human-judgment boundary
priority selection
```

A schema-valid output is not automatically comparison-admissible.

## 7. Cross-company synthesis

No model winner or routing policy is selected during per-call execution.

After all planned observations are recorded, synthesis compares:

```text
engineering completion rate
comparison-admissibility rate
grounding failures
conflict-handling failures
boundary violations
priority-selection variance
material incremental analytical value
latency
gateway cost
output tokens
reasoning tokens
```

A routing proposal requires repeated evidence across distinct company roles.

A single exceptional or poor case is insufficient to establish a global routing rule.

## 8. Stop conditions

Paid execution stops immediately if any of the following occurs:

```text
aggregate conservative spend plan exceeds 2.00 USD before execution
a pinned artifact cannot be verified
prompt/schema comparability is broken
runner behavior changes materially
production/publication authority becomes implicated
monthly project budget safety reserve is threatened
```

A single model-quality failure does not automatically terminate the whole calibration sample, but it must be recorded before proceeding.

## 9. Gate state

```text
Phase A = PASS
Phase B = IN PROGRESS
ST four-model benchmark = COMPLETE
cross-company calibration = PLANNED
Gate 18 = IN PROGRESS / NOT FROZEN
model winner = NONE
routing freeze = NONE
```
