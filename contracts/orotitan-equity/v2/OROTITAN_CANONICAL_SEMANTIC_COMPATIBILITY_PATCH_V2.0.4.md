# OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.4

STATUS =
EMERGENCY COMPATIBILITY PATCH

AUTHORITY =
OROTITAN_EXECUTION_PROCESS_V2_FREEZE_V2.0 §15

PATCH_CLASS =
TARGETED CANONICAL SEMANTIC COMPATIBILITY PATCH

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

HISTORICAL_SNAPSHOT_REWRITE =
NO

ANALYTICAL_JUDGMENT_CHANGE =
NO

CONTRACT_PIN_PACK_MODIFICATION =
NO

CANONICAL_SEMANTIC_COMPATIBILITY_DEFECT =
YES

## 1. Defect

The frozen analytical state can validly establish:

```text
canonical_fundamental_verdicts.return_quality.standard_roic
= NOT_INTERPRETABLE
```

The V2.0.3 compatibility schema cannot represent that exact state because:

```text
l2_research_fundamentals.analytical_metrics.standard_roic
→ #/$defs/returnValue
```

and the global `returnValue` / `specialState` definitions deliberately do not admit `NOT_INTERPRETABLE`.

Integration is a projection boundary. It may not coerce a valid upstream analytical state merely to satisfy schema validation.

This is therefore a blocking canonical semantic representability defect under the V2 §15 emergency mechanism. It is not a methodology, scoring, valuation, certification or terminal-gate change.

## 2. Exact additive correction

V2.0.4 is a strict additive successor to V2.0.3.

It retains unchanged:

```text
V2.0.2:
fundamental_states.roic_trend += NOT_APPLICABLE

V2.0.3:
analytical_metrics.roiic += NOT_INTERPRETABLE
```

It adds exactly:

```text
V2.0.4:
analytical_metrics.standard_roic += NOT_INTERPRETABLE
```

Canonical mapping:

```text
SOURCE
canonical_fundamental_verdicts.return_quality.standard_roic
= NOT_INTERPRETABLE

TARGET
l2_research_fundamentals.analytical_metrics.standard_roic
= NOT_INTERPRETABLE

MAPPING_TYPE
= EXACT_SEMANTIC_PRESERVATION

SEMANTIC_LOSS
= NO
```

No issuer-specific condition is authorized or implemented.

## 3. Field-local firewall

V2.0.4 does not add `NOT_INTERPRETABLE` to:

```text
$defs.specialState
$defs.returnValue
all_in_roic
roic_ex_goodwill
rd_adjusted_roic
share_count_cagr
economic_spread
valuation expected returns
mature-normalization returns
no-multiple returns
score fields
price fields
growth fields
any unrelated numerical field
```

The already-authorized ROIIC exception remains field-local and unchanged.

Forbidden substitutions remain forbidden:

```text
NOT_INTERPRETABLE -> UNKNOWN
NOT_INTERPRETABLE -> NOT_ASSESSABLE
NOT_INTERPRETABLE -> NOT_APPLICABLE
NOT_INTERPRETABLE -> MISSING
NOT_INTERPRETABLE -> NOT_AVAILABLE
NOT_INTERPRETABLE -> numeric value
NOT_INTERPRETABLE -> range
NOT_INTERPRETABLE -> null
NOT_INTERPRETABLE -> omission
NOT_INTERPRETABLE -> free text
```

## 4. Compatibility schema

The V2 compatibility schema is:

```text
contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.4.json
```

It is derived from the exact V2.0.3 compatibility schema.

For `analytical_metrics.standard_roic`, it admits exactly:

```text
#/$defs/returnValue
OR
const "NOT_INTERPRETABLE"
```

For `analytical_metrics.roiic`, it retains exactly the V2.0.3 field-local behavior:

```text
#/$defs/returnValue
OR
const "NOT_INTERPRETABLE"
```

The global definitions remain unchanged.

The frozen V1 schema remains byte-immutable and continues to reject V2-only compatibility states where V1 did not admit them.

## 5. Contract-pin firewall

This emergency compatibility layer is not inserted into the frozen 13-pin Contract Pin Pack.

Existing run `contract_pins` and `contract_set_sha256` are immutable.

For the production V2.0 contract set:

```text
CONTRACT_SET_SHA256
= 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

CONTRACT_SET_CHANGED
= NO
```

No pinned authority bytes are modified by this patch.

## 6. Deterministic non-effect

Neither `analytical_metrics.standard_roic` nor `analytical_metrics.roiic` is an I2 scoring input.

For otherwise identical canonical payloads, changing only either field from a valid numeric return value to `NOT_INTERPRETABLE` must not change:

```text
OQS_RAW
WEAK_LINK_CAP
OQS
OVS
INVESTMENT_RAW
INVESTMENT_SCORE
OROTITAN_STATUS
```

The I3-B / persistence boundary must receive the exact field tokens without coercion.

## 7. Existing-run compatibility rule

An already-created V2 run may consume V2.0.4 without rebinding the 13-pin Contract Pin Pack only if ALL conditions below pass:

```text
1. run.contract_set_sha256
   = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

2. Integration has NOT produced an admitted canonical snapshot.

3. READY_TO_PUBLISH is NO.

4. No publication authorization exists.

5. The representability blocker is limited to an exact authoritative analytical state supported by V2.0.4:
   standard_roic = NOT_INTERPRETABLE
   and/or the already-supported V2.0.3 state:
   roiic = NOT_INTERPRETABLE

6. Authoritative upstream analytical artifacts remain unchanged and hash-valid.

7. No Research / Fundamentals / Valuation / Certification / scoring / terminal artifact is reopened or rewritten.

8. V2.0.4 patch file, compatibility schema, implementation and deployment commit are exact and hash-verified.

9. All required regressions pass.

10. Existing run contract_pins remain untouched.

11. Existing run contract_set_sha256 remains untouched.

12. Integration resumes only from authoritative Registry state using optimistic concurrency.

13. There is no other unrelated unresolved blocker.
```

Any failure:

```text
FAIL CLOSED
DO NOT COERCE SEMANTICS
DO NOT MUTATE THE CURRENT BLOCKED RUN ANALYTICALLY
CREATE A CONTROLLED SUCCESSOR RUN ACCORDING TO THE FROZEN PROCESS
```

## 8. Wise Group plc applicability record

This section records applicability only. It does not create issuer-specific implementation logic.

```text
COMPANY
= Wise Group plc

RUN_ID
= 18c2a3e0-cb05-4c36-89bc-ba3fa6691329

CONTRACT_SET_SHA256
= 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

AUTHORITATIVE DEEP DIVE FINAL MANIFEST
= 6b30634e-111f-48ff-93f3-3e4421d40037@1

DEEP DIVE FINAL MANIFEST SHA256
= d7b26f046b8c43b1965ff31394416bce6f87979b2e33273c6cd6e02d8c99524d

FUNDAMENTALS_LOCK
= 6dd4d1d6-8bd5-4036-a073-dfaadc899a97@1

FUNDAMENTALS_LOCK SHA256
= 7bb5dfb1f4d1d90201d6cafd5fbdea637155e12810d54f6896310aeeec13d2ce

AUTHORITATIVE STATES
standard_roic = NOT_INTERPRETABLE
roiic = NOT_INTERPRETABLE
```

Wise may use the existing-run rule only after post-deployment Registry and artifact revalidation proves every condition in §7. The patch itself does not authorize resume, snapshot persistence or publication.

## 9. Acceptance requirements

V2.0.4 requires regression proof for:

```text
A. pre-existing STANDARD_ROIC returnValue forms
B. STANDARD_ROIC NOT_INTERPRETABLE exact preservation
C. ROIIC NOT_INTERPRETABLE regression
D. simultaneous STANDARD_ROIC + ROIIC NOT_INTERPRETABLE
E. unsupported STANDARD_ROIC string rejection
F. unsupported ROIIC string rejection
G. frozen V1 rejection
H. field locality
I. unchanged global specialState / returnValue
J. V2.0.2 ROIC_TREND NOT_APPLICABLE regression
K. I2 deterministic non-effect
L. I3-B exact persistence-boundary preservation
M. full V2 / I2 / I3-B / OroTitan regression + lint + typecheck + build
N. historical non-effect and zero snapshot rewrite
```

## 10. Production and publication boundary

This patch authorizes no production snapshot write and no publication.

During compatibility resolution:

```text
persist_orotitan_research_snapshot_v2
= FORBIDDEN

record_orotitan_publish_authorization
= FORBIDDEN

record_orotitan_publish_result
= FORBIDDEN

READY_TO_PUBLISH
= NO
```

A blocked Integration stage, if and only if the existing-run rule passes after deployment verification, is resumed through the legal BLOCKED-stage resume path. It is not reopened as COMPLETE work.
