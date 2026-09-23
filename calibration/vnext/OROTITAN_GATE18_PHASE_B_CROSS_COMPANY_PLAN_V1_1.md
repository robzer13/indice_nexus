# OROTITAN_GATE18_PHASE_B_CROSS_COMPANY_PLAN_V1_1

**Gate:** 18  
**Phase:** B company calibration  
**Status:** EXECUTION PLAN — NOT FROZEN  
**Supersedes execution order only:** `OROTITAN_GATE18_PHASE_B_CROSS_COMPANY_PLAN_V1`  
**Methodology change:** NO  
**Prompt/schema change:** NO  
**Physical model identity change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  
**Routing freeze:** NONE  
**Model winner:** NONE  

## 0. Trigger for revision

Two complete four-model company observations are now available.

### STMicroelectronics

```text
role = CYCLICAL_SEGMENTED
LUNA  = comparison-admissible
TERRA = comparison-admissible
SOL   = comparison-admissible
ASTRA = comparison-admissible
```

### RATIONAL AG

```text
role = SIMPLE_CLEAN_COMPOUNDER
packet evidence items = 56
packet conflicts = 9

LUNA  = comparison-admissible
TERRA = not comparison-admissible
SOL   = not comparison-admissible
ASTRA = not comparison-admissible
```

RATIONAL paid observations consumed:

```text
LUNA  = 0.00343145 USD
TERRA = 0.03714350 USD
SOL   = 0.05769200 USD
ASTRA = 0.23156750 USD
TOTAL = 0.32983445 USD
```

The three higher-cost RATIONAL candidates failed the semantic evidence-reference contract. SOL and ASTRA were also materially incomplete.

This does not establish LUNA as a global winner. It does establish that automatically paying for all four candidates on every remaining company is no longer the highest-information-per-dollar execution strategy.

## 1. Unchanged comparison contract

The controlled request contract remains unchanged:

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

No model-specific packet, prompt, schema, source-access or output-cap exception is introduced.

## 2. Remaining pinned cases

```text
Constellation Software
role = SERIAL_ACQUIRER
evidence items = 11
conflicts = 1

Brookfield Corporation
role = ACCOUNTING_HEAVY
evidence items = 12
conflicts = 2

Adyen
role = DIFFICULT_MOAT_CONFLICTING_EVIDENCE
evidence items = 16
conflicts = 3
```

All pinned hashes and dry-run admission checks from `OROTITAN_GATE18_PHASE_B_CROSS_COMPANY_DRYRUN_ADMISSION_001.json` remain authoritative.

## 3. Revised paid execution sequence

The former automatic within-company sequence:

```text
LUNA -> TERRA -> SOL -> ASTRA
```

is suspended for the three remaining companies.

The next sequence is a low-cost cross-role LUNA sweep:

```text
1. Constellation Software x LUNA
2. Brookfield Corporation x LUNA
3. Adyen x LUNA
```

Each call remains separately authorized, executed, adjudicated, recorded, CI-validated and merged before the next call.

No TERRA, SOL or ASTRA call is authorized during this sweep.

## 4. Spend guard

Dry-run conservative LUNA ceilings:

```text
Constellation Software = 0.0067830 USD
Brookfield Corporation = 0.0068432 USD
Adyen                  = 0.0070458 USD
TOTAL                  = 0.0206720 USD
```

Budget snapshot after RATIONAL ASTRA:

```text
project monthly limit = 4.00 USD
project displayed spend = 1.40 USD
team monthly limit = 4.00 USD
team displayed spend = 1.40 USD
displayed remaining = 2.60 USD
```

The Vercel CLI budget display remains rounded and is not exact per-call evidence.

## 5. Per-observation admissibility

Engineering admissibility remains:

```text
schemaValid = true
semanticValid = true
finishReason = stop
retryCount = 0 unless separately explained
provider cost evidence present
response hash present
```

Human adjudication remains separate:

```text
narrative completeness
exact E-/C- grounding
material conflict handling
counter-evidence treatment
weak-link usefulness
unresolved-point usefulness
human-judgment boundary
priority selection
```

No malformed reference may be repaired post hoc for comparison admission.

## 6. Post-sweep decision gate

After all three remaining LUNA observations are recorded, stop paid execution and synthesize the five-company LUNA evidence first.

The synthesis must determine:

```text
semantic-validity rate across roles
comparison-admissibility rate across roles
packet-density sensitivity
conflict-density sensitivity
priority-selection stability
weak-link usefulness
latency
cost
token usage
```

Only then may higher-cost contrast calls be proposed.

A contrast call must answer a specific unresolved calibration question. Examples include:

```text
Does a higher-reasoning candidate materially improve a conflict-heavy case?
Does a higher-reasoning candidate restore or degrade exact evidence-ID discipline?
Is a failure associated with packet density rather than company archetype?
```

No contrast model is preselected in this plan.

## 7. Higher-cost contrast cap

After the LUNA sweep, any TERRA/SOL/ASTRA observation requires a new explicit authorization record.

Constraints:

```text
one company
one model
one call
no automatic retry
no multi-model execution
explicit hard spend cap
private gitignored raw output
record + CI + merge before another paid call
```

The default is no escalation unless the expected calibration value is material.

## 8. Interpretation boundary

This revision is execution economics, not model routing.

It does not mean:

```text
LUNA is frozen as winner
higher-reasoning models are rejected globally
RATIONAL generalizes to every company
the prompt/schema is defective
Gate 18 is complete
```

It means only that sequential evidence now supports collecting cheap cross-role LUNA observations before purchasing more expensive contrasts.

## 9. Gate state

```text
Phase A = PASS
Phase B = IN PROGRESS
ST four-model benchmark = COMPLETE
RATIONAL four-model benchmark = COMPLETE
remaining cross-company strategy = LUNA SWEEP
Gate 18 = IN PROGRESS / NOT FROZEN
model winner = NONE
routing freeze = NONE
```
