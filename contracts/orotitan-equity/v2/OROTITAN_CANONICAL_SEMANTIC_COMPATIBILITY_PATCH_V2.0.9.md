# OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.9

STATUS =
EMERGENCY COMPATIBILITY PATCH

AUTHORITY =
OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0 §15

PATCH_CLASS =
TARGETED CANONICAL REPRESENTATION COMPATIBILITY PATCH

BLOCKING_DEFECT_CLASS =
DETERMINISTIC_CANONICAL_OUTPUT_REPRESENTABILITY_DEFECT

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

## 1. Blocking defect

The frozen analytical methodology can legitimately produce:

```text
l2_research_fundamentals.analytical_metrics.rd_adjusted_roic
= NOT_INTERPRETABLE
```

when the R&D-adjusted return calculation is not economically interpretable.

The deployed V2.0.8 compatibility schema still maps that field only to:

```text
#/$defs/returnValue
```

and `returnValue` does not include `NOT_INTERPRETABLE`.

Substituting a number or another state would change analytical semantics and is forbidden.

## 2. Exact additive correction

V2.0.9 preserves every V2.0.2-V2.0.8 compatibility correction and adds exactly one field-local rule:

```text
TARGET_FIELD
= l2_research_fundamentals.analytical_metrics.rd_adjusted_roic

ADDITIONAL_PERMITTED_VALUE
= NOT_INTERPRETABLE
```

The field-local schema becomes:

```json
{
  "oneOf": [
    { "$ref": "#/$defs/returnValue" },
    { "const": "NOT_INTERPRETABLE" }
  ]
}
```

## 3. Semantic firewall

V2.0.9 does not broaden:

```text
$defs.specialState
$defs.returnValue
all_in_roic
any other analytical metric
any score field
any valuation field
any certification field
any terminal-gate field
```

Existing field-local permissions for `standard_roic`, `roic_ex_goodwill` and `roiic` remain unchanged.

## 4. Issuer independence

The correction is global and deterministic. It contains no issuer, security, ticker, company or RUN_ID branch.

## 5. Contract Set firewall

The compatibility layer remains outside the frozen 13-pin V2 Contract Pin Pack.

```text
CONTRACT_SET_SHA256
= 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

CONTRACT_SET_CHANGED
= NO

SOURCE_RUN_CONTRACT_PIN_MUTATION
= NO
```

## 6. Existing-run admission

The finite compatibility projection set is extended with exactly:

```text
targetField
= l2_research_fundamentals.analytical_metrics.rd_adjusted_roic

sourceValue
= NOT_INTERPRETABLE

targetValue
= NOT_INTERPRETABLE

SEMANTIC_LOSS
= NONE
```

Blocked Integration checkpoints remain subject to all V2.0.7/V2.0.8 lifecycle, hash, lineage, regression, publication and snapshot-state controls.

After PASS, the only authorized same-stage recovery operation is:

```text
public.resume_orotitan_stage(...)
```

No Deep Dive reopen is authorized.

## 7. Deterministic non-effect

The representability correction must not change:

```text
OQS_RAW
WEAK_LINK_CAP
OQS
OVS
INVESTMENT_RAW
INVESTMENT_SCORE
OROTITAN_STATUS
```

## 8. I3-B boundary

The validated snapshot writer remains unchanged. Regression must prove that an already validated `NOT_INTERPRETABLE` value reaches the I3-B persistence boundary without coercion. The production snapshot writer must not be invoked as a regression test.

## 9. Mandatory regression matrix

V2.0.9 requires PASS for:

```text
rd_adjusted_roic legacy numeric/range/special-state forms
rd_adjusted_roic NOT_INTERPRETABLE
arbitrary rd_adjusted_roic strings -> REJECT
NOT_INTERPRETABLE in unauthorized all_in_roic -> REJECT
$defs.returnValue unchanged
$defs.specialState unchanged
all V2.0.2-V2.0.8 compatibility rules preserved
I2 deterministic outputs unchanged
I3-B boundary preserves exact NOT_INTERPRETABLE
existing blocked Integration checkpoint admission remains fail-closed
V1 frozen validator remains unchanged
lint
typecheck
unit regression
production build
```

Any failed regression fails the patch closed.

## 10. Deployment / publication boundary

V2.0.9 authorizes no analytical artifact rewrite, no canonical snapshot persistence and no publication by itself.

Until production deployment and same-run compatibility admission pass:

```text
DO NOT RESUME INTEGRATION
DO NOT PERSIST CANONICAL SNAPSHOT
DO NOT SET READY_TO_PUBLISH = YES
DO NOT RECORD PUBLICATION AUTHORIZATION
```

Publication remains separately gated by `GO PUBLISH <COMPANY>`.
