# OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.8

STATUS =
EMERGENCY COMPATIBILITY PATCH

AUTHORITY =
OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0 §15

PATCH_CLASS =
TARGETED CANONICAL REPRESENTATION COMPATIBILITY PATCH

BLOCKING_DEFECT_CLASS =
CONTRACT_CONTRADICTION + DETERMINISTIC_CANONICAL_OUTPUT_REPRESENTABILITY_DEFECT

NEW_METHODOLOGY =
NO

SCORING_CHANGE =
NO

VALUATION_CHANGE =
NO

CERTIFICATION_CHANGE =
NO

TERMINAL_GATE_CHANGE =
NO

I2_CHANGE =
NO

I3B_CHANGE =
NO

HISTORICAL_SNAPSHOT_REWRITE =
NO

ANALYTICAL_JUDGMENT_CHANGE =
NO

CONTRACT_PIN_PACK_MODIFICATION =
NO

CANONICAL_SEMANTIC_COMPATIBILITY_DEFECT =
YES

## 1. Blocking defect

The frozen analytical and valuation authorities can legitimately produce a shareholder expected-return horizon that is a positive fractional number of years.

The frozen V1 core representation, and the deployed V2.0.6 compatibility schema inherited by V2.0.7, currently constrain:

```text
l3_investment_valuation.valuation.return_horizon
```

to:

```text
integer >= 1
OR
existing permitted specialState
```

A legitimately fractional horizon therefore cannot be projected without rounding, truncation or state substitution.

Integration is a non-analytical projection boundary. Rounding, truncating or replacing a known fractional horizon with a special state changes or discards analytical semantics and is forbidden.

This is a blocking canonical representation defect. It is not a methodology, valuation-policy, scoring, Certification, terminal-gate, I2, I3-B or issuer-specific analytical exception.

## 2. Analytical validity boundary

V2.0.8 does not create or select an analytical horizon.

It only permits exact canonical projection when the authoritative upstream analytical artifact has already established a legitimate positive fractional return horizon under the governing valuation methodology.

If the upstream horizon is analytically invalid, accidental, contradictory or requires a valuation rerun, V2.0.8 does not apply.

## 3. Exact additive correction

V2.0.8 preserves all semantic corrections authorized by V2.0.2 through V2.0.6 and all existing-run lifecycle admission rules authorized by V2.0.7.

It adds exactly one representation correction:

```text
TARGET FIELD
= l3_investment_valuation.valuation.return_horizon

PREVIOUS NUMERIC FORM
= integer >= 1

V2.0.8 NUMERIC FORM
= number > 0

UNIT
= years

SERIALIZATION
= JSON number

SPECIAL-STATE BRANCH
= unchanged
```

The field-local schema definition becomes:

```json
{
  "oneOf": [
    {
      "type": "number",
      "exclusiveMinimum": 0
    },
    {
      "$ref": "#/$defs/specialState"
    }
  ],
  "x-unit": "years"
}
```

No hidden rounding or decimal-place reduction is authorized.

JSON / Ajv finite-number semantics apply. NaN and Infinity are not canonical JSON numbers and remain rejected.

No arbitrary maximum horizon is introduced because no governing analytical contract establishes one. If a later analytical authority establishes an upper bound, that requires its own governed change.

## 4. Field-local semantic firewall

V2.0.8 does not broaden:

```text
$defs.specialState
$defs.returnValue
any other integer field
any other duration field
any other year-count field
any analytical-metric special state
any score field
any price field
any certification state
any terminal-gate state
```

Existing ROIIC compatibility remains exactly:

```text
l2_research_fundamentals.analytical_metrics.roiic
+= NOT_INTERPRETABLE
```

No additional ROIIC rule is created.

## 5. Issuer independence

The rule is global and deterministic.

It contains no:

```text
issuer identifier
security identifier
RUN_ID
company name
ticker
Microsoft-specific branch
```

Any OroTitan V2 run with an analytically legitimate positive fractional return horizon may use the same field-local representation.

## 6. Exact production schema / validator successors

The compatibility schema successor is:

```text
NAME
= 04_SCREENER_SCHEMA_V1_COMPAT_V2.0.8

PATH
= contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.8.json

SHA256
= 30ffb6648845bdafebe282ccdca3c6a33ff9e0ca8dd926fe61a3186a9ab6426c
```

The V2 validator successor is:

```text
PATH
= lib/orotitan-equity/v2/research-snapshot-schema.ts

SHA256
= 16957bec96f973833660c114b794814802f8cd500f91e92a127432b57de18fcb
```

The validator continues to compose:

```text
04_SCREENER_SCHEMA_V2
+
V1 core compatibility schema
+
existing V2 product semantic checks
+
existing I2 reconciliation boundary
```

No frozen V1 authority is modified.

## 7. Contract Set firewall

The compatibility layer remains outside the frozen 13-pin Contract Pin Pack.

For eligible V2 runs:

```text
CONTRACT_SET_SHA256
= 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

CONTRACT_SET_CHANGED
= NO

SOURCE_RUN_CONTRACT_SET_MUTATION
= NO

SOURCE_RUN_CONTRACT_PIN_MUTATION
= NO
```

If resolution would require rebinding the run's Contract Set, V2.0.8 does not authorize same-run recovery.

## 8. Existing-run admission

V2.0.8 extends the finite compatibility projection set used by the V2.0.7 existing-run admission controller with exactly:

```text
targetField
= l3_investment_valuation.valuation.return_horizon

sourceValue
= finite number > 0

targetValue
= exact same finite number > 0

SEMANTIC_LOSS
= NONE
```

For all string-valued compatibility rules, the V2.0.2-V2.0.6 finite rule set remains unchanged.

For a blocked Integration checkpoint, V2.0.7 lifecycle requirements remain mandatory, including a durable active CHECKPOINT, compatible Contract Set, no admitted canonical snapshot, READY_TO_PUBLISH=NO, no publication authorization/event, hash-valid upstream artifacts, exact compatibility bytes, PASS regressions, no analytical mutation and no unrelated blocker.

After PASS, the canonical recovery mechanism remains:

```text
public.resume_orotitan_stage(...)
```

No reopen operation is authorized for a BLOCKED Integration stage.

## 9. Deterministic non-effect

`return_horizon` is a stored valuation calculation output. The compatibility change does not alter:

```text
OQS_RAW
WEAK_LINK_CAP
OQS
OVS
INVESTMENT_RAW
INVESTMENT_SCORE
OROTITAN_STATUS
```

For an otherwise identical canonical payload, changing only the representation capability from integer-only to exact positive numeric years must leave deterministic I2 outputs byte/value-equivalent.

## 10. I3-B boundary

The validated V2 snapshot writer remains unchanged.

V2.0.8 requires proof that I3-B receives the exact validated JSON number without coercion, truncation, rounding or schema downgrade.

The production snapshot writer is not invoked merely to test this patch.

## 11. Exact implementation boundary

Patch base:

```text
REPOSITORY
= robzer13/indice_nexus

BASE_COMMIT
= 79f03bd4a2576c61c181a4c0df41dda48014e610
```

The V2.0.8 compatibility correction is valid only if the implementation diff from that base is limited to:

```text
A. contracts/orotitan-equity/v2/OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.8.md
   new immutable compatibility authority

B. contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.8.json
   strict field-local successor of V2.0.6
   return_horizon integer >= 1 -> number > 0
   no other semantic schema change

C. lib/orotitan-equity/v2/research-snapshot-schema.ts
   compatibility schema path / URN only

D. lib/orotitan-equity/v2/existing-run-compatibility-admission.ts
   add exact positive-finite numeric return_horizon compatibility projection
   retain all V2.0.7 lifecycle controls

E. tests/equity-v2-roic-not-applicable.test.ts
   schema, precision, I2 and I3-B regression matrix

F. tests/equity-v2-compatibility-admission.test.ts
   existing-run admission regression for exact fractional return_horizon
```

No frozen V1 file and no frozen V2 13-pin authority file may be modified.

Because an authority file cannot embed its own digest without recursion, its exact SHA-256, Git blob and production commit are verified externally after merge.

## 12. Mandatory regression matrix

V2.0.8 requires PASS for:

```text
ROIIC numeric legacy forms
ROIIC permitted legacy special states
ROIIC NOT_INTERPRETABLE
NOT_INTERPRETABLE in unauthorized fields -> REJECT

return_horizon integers:
1
3
5
10

return_horizon fractions:
0.25
0.5
1.5
4.7835616438356166
9.999999

invalid return_horizon:
0
negative
NaN
Infinity
string
malformed / non-numeric form

JSON stringify/parse:
4.7835616438356166
-> exact same numeric value and decimal round-trip representation

I2 deterministic outputs unchanged

I3-B validated writer input preserves exact fractional value

legacy integer-horizon V2 runs remain valid
unrelated V2 outputs remain valid
V1 frozen validator remains unchanged

existing-run compatibility admission:
ROIIC NOT_INTERPRETABLE exact + fractional return_horizon exact
-> admissible only when every V2.0.7 common/checkpoint control passes

complete prior unit regression
PostgreSQL Registry regression
lint
typecheck
production build
```

Any failed regression fails the patch closed.

## 13. Deployment / publication boundary

V2.0.8 authorizes no analytical artifact rewrite, no canonical snapshot persistence and no publication by itself.

Until production compatibility deployment and same-run admission pass:

```text
DO NOT RESUME INTEGRATION
DO NOT PERSIST CANONICAL SNAPSHOT
DO NOT SET READY_TO_PUBLISH = YES
DO NOT RECORD PUBLICATION AUTHORIZATION
```

Publication remains separately gated by:

```text
GO PUBLISH <COMPANY>
```

V2.0.8 never grants publication authorization.
