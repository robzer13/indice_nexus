# OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.2

**Status:** EMERGENCY COMPATIBILITY PATCH — BLOCKING CONTRACT / CANONICAL REPRESENTABILITY DEFECT  
**Authority:** `OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0` §15 emergency-patch rule  
**Patch class:** TARGETED CONTRACT / CANONICAL SEMANTIC COMPATIBILITY PATCH  
**Methodology change:** NO  
**Scoring change:** NO  
**Valuation change:** NO  
**Certification change:** NO  
**Terminal-gate change:** NO  
**Historical snapshot rewrite:** NO  
**Analytical judgment change:** NO  
**Contract Pin Pack V2 modification:** NO

## 1. Defect

The frozen V1 Integration semantic vocabulary explicitly distinguishes:

```text
UNKNOWN
NOT_APPLICABLE
NOT_ASSESSABLE
MISSING
NOT_AVAILABLE
```

and requires schema-required analytical fields to preserve the exact state supported by the authoritative analytical dossier.

The V2 Integration contract is non-analytical and forbids reinterpretation of upstream judgments.

However, the V1 core field:

```text
l2_research_fundamentals.fundamental_states.roic_trend
```

admits only:

```text
IMPROVING
STABLE
DECLINING
VOLATILE
UNCLEAR
```

This makes the legitimate authoritative value:

```text
NOT_APPLICABLE
```

unrepresentable for a sector where industrial ROIC trend is structurally inapplicable.

This is a canonical semantic representability defect, not an analytical contradiction.

## 2. Exact semantic correction

The only new admissible value for `fundamental_states.roic_trend` is:

```text
NOT_APPLICABLE
```

The patched V2 compatibility enum is exactly:

```text
IMPROVING
STABLE
DECLINING
VOLATILE
UNCLEAR
NOT_APPLICABLE
```

No other enum is broadened.

Exact mapping rule:

```text
SOURCE
canonical_fundamental_verdicts.ROIC_TREND = NOT_APPLICABLE

TARGET
l2_research_fundamentals.fundamental_states.roic_trend = NOT_APPLICABLE

MAPPING_TYPE = EXACT_SEMANTIC_PRESERVATION
SEMANTIC_LOSS = NO
```

## 3. Forbidden substitutions

The patch does not authorize:

```text
NOT_APPLICABLE -> UNCLEAR
NOT_APPLICABLE -> STABLE
NOT_APPLICABLE -> null
NOT_APPLICABLE -> omitted required field
NOT_APPLICABLE -> free text
```

It does not authorize a company-specific exception.

## 4. Immutability firewall

The frozen V1 schema file remains unchanged and historically hash-resolvable.

V2 uses the additive compatibility schema:

```text
contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.2.json
```

That compatibility schema is a V2-only validation copy of the frozen V1 core with one targeted semantic correction: `roic_trend += NOT_APPLICABLE`.

Existing V1 validators and V1 historical snapshots continue to resolve the frozen V1 schema unchanged.

None of the 13 bytesets in `OROTITAN_CONTRACT_PIN_PACK_V2.json` is modified by this patch.

Therefore:

```text
PINNED_AUTHORITY_BYTES_CHANGED = NO
CONTRACT_SET_CHANGED = NO
CONTRACT_SET_SHA256 = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
```

## 5. Existing-run compatibility rule

This emergency patch may be consumed by an already-running blocked V2 run without rebinding its 13-pin Contract Pin Pack only when all of the following are true:

```text
1. run contract_set_sha256
   = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

2. run remains BLOCKED at INTEGRATION

3. blocker is exactly a semantic representability defect for:
   canonical_fundamental_verdicts.ROIC_TREND = NOT_APPLICABLE
   ->
   l2_research_fundamentals.fundamental_states.roic_trend

4. authoritative upstream analytical artifacts remain unchanged and hash-valid

5. no analytical, valuation, scoring, certification or terminal-gate output is reopened

6. patch file, compatibility schema and implementation commit are exact and hash-verified

7. required regression suites pass

8. resume occurs only through the controlled Registry RPC using exact optimistic-concurrency versions
```

This rule is a compatibility admission mechanism authorized by the pinned V2 process emergency-patch clause. It does not mutate `orotitan_runs.contract_pins` or `contract_set_sha256`.

Any mismatch requires fail-closed handling and a successor run.

## 6. Deterministic non-effect

`roic_trend` is a stored analytical judgment and is not an input to canonical I2 score formulas.

This patch therefore does not alter:

```text
OQS_RAW
WEAK_LINK_CAP
OQS
OVS
INVESTMENT_RAW
INVESTMENT_SCORE
OROTITAN_STATUS
```

for an otherwise identical valid payload.

## 7. Acceptance requirements

At minimum prove:

```text
A. existing roic_trend values remain valid
B. NOT_APPLICABLE is valid in the V2 core compatibility layer
C. unsupported values remain invalid
D. full V2 ANALYZE snapshot validates with NOT_APPLICABLE
E. exact value is preserved after validation
F. I2 deterministic outputs are unchanged
G. V2 persistence admission accepts the payload
H. frozen V1 validation remains unchanged
I. existing V2 regression remains PASS
J. no historical V1 rewrite is required
K. no score or terminal status changes solely from this patch
L. INSURANCE/BANKING sector use requires no issuer-specific code path
```

## 8. Scope boundary

```text
NEW_METHODOLOGY = NO
SCORING_CHANGE = NO
VALUATION_CHANGE = NO
CERTIFICATION_CHANGE = NO
TERMINAL_GATE_CHANGE = NO
HISTORICAL_SNAPSHOT_REWRITE = NO
ANALYTICAL_JUDGMENT_CHANGE = NO
CANONICAL_SEMANTIC_COMPATIBILITY_DEFECT = YES
```
