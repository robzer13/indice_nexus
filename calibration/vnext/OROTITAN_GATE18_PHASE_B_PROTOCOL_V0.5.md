# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.5

**Gate:** 18  
**Phase:** B company calibration  
**Status:** EXECUTION CANDIDATE — NOT FROZEN  
**Methodology change:** NO  
**Prompt/schema change:** NO  
**Physical model identity change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  

## 0. Trigger

SOL at `reasoning=high` exhausted the complete Phase B output budget twice on the same STMicroelectronics MOAT audit:

```text
v0.3 envelope
maxOutputTokens = 1536
reasoningTokens = 1536
visible output = 0
finishReason = length
gatewayCostUsd = 0.043277

v0.4 envelope
maxOutputTokens = 4096
reasoningTokens = 4096
visible output = 0
finishReason = length
gatewayCostUsd = 0.094477
```

No analytical-quality observation exists for either SOL attempt because no visible structured answer was emitted.

A third blind max-output increase is not justified under the Gate 18 cost-discipline rule.

## 1. Allowed variable

Gate 18 section 7 permits model-variable reasoning profiles while requiring the following to remain identical for one benchmark task:

```text
company
DATA_CUTOFF
Evidence Packet
module question
system instruction
GenerationSchema
max output contract
tool availability
source availability
```

Therefore protocol v0.5 changes only the Phase B SOL reasoning effort.

## 2. Phase B execution profiles

```text
LUNA  -> minimal
TERRA -> medium
SOL   -> medium
ASTRA -> high
```

Physical model IDs remain unchanged.

SOL remains:

```text
openai/gpt-5.6-sol
T3_PREMIUM_REASONING candidate
reference-quality baseline candidate
```

Changing the company-level execution profile from high to medium is not:

```text
a model substitution
a routing promotion
a quality verdict
a methodology change
```

It is an empirically motivated bounded-execution correction.

## 3. Common transport contract

The v0.4 common max-output envelope remains unchanged:

```text
maxOutputTokens = 4096
```

No per-model output-cap exception is allowed.

Prompt, GenerationSchema, evidence packet and semantic validation remain unchanged from v0.4.

## 4. Existing LUNA/TERRA evidence

The accepted LUNA v0.4 and TERRA v0.4 observations remain comparison-admissible.

They do not require paid reruns because their complete invocation contract remains unchanged under v0.5:

```text
same physical model
same reasoning profile
same packet
same prompt
same schema
same 4096 max-output contract
same source access
```

Only SOL's explicitly permitted model-specific reasoning profile changes.

## 5. Next execution sequence

Before any paid SOL v0.5 call:

```text
1. free dry-run
2. live conservative cost ceiling
3. explicit single-model cap
4. SOL only
5. inspect engineering receipt + output
```

ASTRA remains unauthorized until SOL v0.5 is inspected.

No automatic retry is authorized.

## 6. Cost discipline

The following remain mandatory:

```text
explicit --model
explicit --max-case-spend-usd
single paid model per command
pre-call conservative live-price guard
no --allow-multi-model unless separately authorized
private gitignored model output
```

## 7. Gate state

```text
Phase A = PASS
Phase B = IN PROGRESS
Gate 18 = IN PROGRESS / NOT FROZEN
model winner = NONE
routing freeze = NONE
```
