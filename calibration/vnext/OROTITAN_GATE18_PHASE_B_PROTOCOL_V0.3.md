# OROTITAN_GATE18_PHASE_B_PROTOCOL_V0.3

**Gate:** 18  
**Phase:** B company calibration  
**Status:** EXECUTION CANDIDATE — NOT FROZEN  
**Methodology change:** NO  
**Publication authority:** DISABLED  
**Production mutation:** FORBIDDEN  

## 0. Trigger

The first paid v0.2 canary was deliberately limited to:

```text
STMicroelectronics
LUNA only
max case spend = 0.004 USD
```

Engineering result:

```text
schema_valid = true
semantic_valid = true
finish_reason = stop
retry_count = 0
gateway_cost_usd = 0.00191305
input_tokens = 2380
output_tokens = 786
reasoning_tokens = 116
```

The sampled evidence and conflict references were grounded in the pinned packet and the human-judgment boundary was respected.

However, multiple narrative fields ended at or near the v0.2 hard character ceiling or terminated mid-thought.

Therefore:

```text
v0.2 canary = engineering pass
v0.2 canary = human-quality hold
v0.2 output = not comparison-admissible
```

This is treated as an output-contract defect, not as a proven LUNA quality defect.

## 1. v0.3 change

The evidence packet remains the same deterministic MOAT_INPUTS-scoped packet.

The provider output-token cap remains:

```text
1536
```

The only intended changes are:

```text
prompt/schema version -> 0.3
narrative hard ceiling -> 180 characters
weak-link candidate ceiling -> 140 characters
prompt target -> <= 120 characters
complete self-contained narrative required
terminal punctuation required
boundary saturation rejected deterministically
```

The extra schema headroom exists to allow a thought to finish. It is not an instruction to generate longer prose.

## 2. Cost discipline

No TERRA, SOL or ASTRA call is authorized by this change.

The next paid inference, if separately approved after deterministic CI and dry-run, is:

```text
same company = STMicroelectronics
same model = LUNA
same evidence packet
v0.3 prompt/schema
single model only
explicit spend cap required
```

This rerun is necessary because model comparisons require the same prompt/schema contract. The v0.2 LUNA output cannot be compared directly with a later v0.3 TERRA/SOL/ASTRA output.

## 3. Gate state

```text
Gate 18 = IN PROGRESS
Phase A = PASS
Phase B = IN PROGRESS
model winner = NONE
routing freeze = NONE
```
