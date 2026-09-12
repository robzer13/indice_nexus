# OROTITAN_EXECUTION_CONTRACT_PATCH_V1.0.1

## N_SELECTION_RULE

```text
IF valid numeric MATURE_NORMALIZATION_RETURN exists
→ use it as N basis

ELSE IF MATURE_NORMALIZATION_RETURN is legitimately
NOT_ASSESSABLE or NOT_AVAILABLE after required work
AND valid numeric NO_MULTIPLE_EXPANSION_RETURN exists
→ use NO_MULTIPLE_EXPANSION_RETURN as N basis

ELSE
→ preserve the applicable Mature-Normalization unavailable semantic state
→ numeric OVS prohibited

INVALID / UNRECONCILED MATURE_NORMALIZATION_RETURN
→ FAIL CLOSED
→ no Same-Multiple fallback
```

## Economic rationale

`MATURE_NORMALIZATION_RETURN` has precedence because it directly tests whether the investment still provides an adequate normalized return without relying on an economically aggressive mature valuation state. `NO_MULTIPLE_EXPANSION_RETURN` remains a diagnostic and fallback only when Mature Normalization was legitimately performed but cannot be numerically assessed.

The selector is not `MIN`, `MAX`, analyst discretion, or a conservatism override. When both valid returns are numeric, Mature Normalization is selected regardless of whether it is above or below Same-Multiple.

## Fail-closed behavior

A calculation failure, definition failure, basis mismatch, unresolved critical assumption, invalid value, or deterministic reconciliation failure affecting Mature Normalization does not create a fallback condition. Existing certification and reconciliation blockers remain authoritative and the affected numeric OVS must not be published.

```text
NEW_METHODOLOGY
= NO
```
