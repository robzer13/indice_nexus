# OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.6

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

The authoritative analytical state can validly establish:

```text
RETURN_QUALITY.canonical_verdict_fields.roic_ex_goodwill
= NOT_INTERPRETABLE
```

The active V2.0.5 compatibility schema cannot represent that exact state because:

```text
l2_research_fundamentals.analytical_metrics.roic_ex_goodwill
```

is bound directly to:

```text
$defs.returnValue
```

and global `returnValue` / `specialState` do not admit `NOT_INTERPRETABLE`.

Integration is a projection boundary and may not replace the authoritative analytical judgment with `UNKNOWN`, `NOT_AVAILABLE`, `NOT_APPLICABLE`, `null`, omission, free text, a numeric value, or any other value merely to satisfy schema validation.

This is a blocking canonical semantic representability defect. It is not a methodology, scoring, valuation, certification, terminal-gate, I2, I3-B, or issuer-specific analytical exception.

## 2. Exact additive correction

V2.0.6 is a strict additive successor to V2.0.5.

It retains unchanged:

```text
V2.0.2:
fundamental_states.roic_trend += NOT_APPLICABLE

V2.0.3:
analytical_metrics.roiic += NOT_INTERPRETABLE

V2.0.4:
analytical_metrics.standard_roic += NOT_INTERPRETABLE

V2.0.5:
fundamental_states.roic_trend += UNKNOWN
```

It adds exactly:

```text
V2.0.6:
analytical_metrics.roic_ex_goodwill += NOT_INTERPRETABLE
```

Canonical mapping:

```text
SOURCE
RETURN_QUALITY.canonical_verdict_fields.roic_ex_goodwill
= NOT_INTERPRETABLE

TARGET
l2_research_fundamentals.analytical_metrics.roic_ex_goodwill
= NOT_INTERPRETABLE

MAPPING_TYPE
= EXACT_SEMANTIC_PRESERVATION

SEMANTIC_LOSS
= NO
```

No issuer-specific condition is authorized or implemented.

## 3. Field-local semantic firewall

The only new admissible value introduced by V2.0.6 is:

```text
l2_research_fundamentals.analytical_metrics.roic_ex_goodwill
+= NOT_INTERPRETABLE
```

The resulting field-local definition is:

```json
{
  "oneOf": [
    { "$ref": "#/$defs/returnValue" },
    { "const": "NOT_INTERPRETABLE" }
  ]
}
```

V2.0.6 does not broaden:

```text
$defs.specialState
$defs.returnValue
analytical_metrics.all_in_roic
analytical_metrics.rd_adjusted_roic
analytical_metrics.share_count_cagr
any fundamental-state enum
any score field
any valuation field
any price field
any certification state
any terminal-gate state
```

The V2.0.3, V2.0.4 and V2.0.5 field-local compatibility corrections remain unchanged.

Forbidden substitutions include:

```text
NOT_INTERPRETABLE -> UNKNOWN
NOT_INTERPRETABLE -> NOT_AVAILABLE
NOT_INTERPRETABLE -> NOT_APPLICABLE
NOT_INTERPRETABLE -> NOT_ASSESSABLE
NOT_INTERPRETABLE -> MISSING
NOT_INTERPRETABLE -> null
NOT_INTERPRETABLE -> omission
NOT_INTERPRETABLE -> empty string
NOT_INTERPRETABLE -> free text
NOT_INTERPRETABLE -> numeric value
```

## 4. Exact file boundary

Patch base:

```text
REPOSITORY
= robzer13/indice_nexus

BASE_COMMIT
= 52837e2b04a2eaf40b1739600320c3c636e78bfa
```

The V2.0.6 compatibility correction is valid only if the implementation diff from that base is limited to exactly these paths:

```text
A. contracts/orotitan-equity/v2/OROTITAN_CANONICAL_SEMANTIC_COMPATIBILITY_PATCH_V2.0.6.md
   new compatibility authority

B. contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.6.json
   strict JSON successor of V2.0.5
   metadata version 2.0.5 -> 2.0.6
   analytical_metrics.roic_ex_goodwill gains exactly NOT_INTERPRETABLE
   no other semantic schema difference

C. lib/orotitan-equity/v2/research-snapshot-schema.ts
   exact implementation change:
   04_SCREENER_SCHEMA_V1_COMPAT_V2.0.5.json
   -> 04_SCREENER_SCHEMA_V1_COMPAT_V2.0.6.json

   exact compiled core URN change:
   v1-v2-compat-2.0.5#/$defs/researchSnapshot
   -> v1-v2-compat-2.0.6#/$defs/researchSnapshot

D. tests/equity-v2-roic-not-applicable.test.ts
   V2.0.6 field-local compatibility regressions
```

No frozen V1 file and no frozen V2 13-pin authority file may be modified.

Because an authority file cannot embed its own content digest without recursion, exact bytes are identified by the production Git commit and per-file Git blob identifiers, and are additionally SHA-256 verified from the deployed commit after merge. Existing-run admission is forbidden until that external hash record is complete.

## 5. Contract-pin firewall

This compatibility layer is not inserted into the frozen 13-pin Contract Pin Pack.

Existing run `contract_pins` and `contract_set_sha256` are immutable.

For the production V2.0 contract set:

```text
CONTRACT_SET_SHA256
= 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

CONTRACT_SET_CHANGED
= NO
```

## 6. Deterministic non-effect

`analytical_metrics.roic_ex_goodwill` is a stored analytical calculation/judgment output and is not an I2 deterministic scoring input.

For otherwise identical canonical payloads, changing only:

```text
roic_ex_goodwill = 12
```

to:

```text
roic_ex_goodwill = NOT_INTERPRETABLE
```

must not change:

```text
OQS_RAW
WEAK_LINK_CAP
OQS
OVS
INVESTMENT_RAW
INVESTMENT_SCORE
OROTITAN_STATUS
```

The I3-B persistence boundary must receive the exact `NOT_INTERPRETABLE` token without coercion.

## 7. Existing-run compatibility rule

An already-created V2 run may consume V2.0.6 without rebinding the frozen 13-pin Contract Pin Pack only if ALL conditions pass:

```text
1. run.contract_set_sha256 = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
2. Integration has NOT produced an admitted canonical snapshot.
3. READY_TO_PUBLISH = NO.
4. No publication authorization exists.
5. The representability blocker is limited to an exact authoritative state admitted by the compatibility chain.
6. Authoritative upstream analytical artifacts remain unchanged and hash-valid.
7. No Research / Fundamentals / Valuation / Certification / scoring / terminal artifact is reopened or rewritten merely to satisfy representation.
8. V2.0.6 authority file, compatibility schema, implementation and deployment commit are exact and hash-verified.
9. All required regressions pass.
10. Existing run contract_pins remain untouched.
11. Existing run contract_set_sha256 remains untouched.
12. Integration resumes only from authoritative Registry state using optimistic concurrency.
13. The active Integration artifact remains a durable CHECKPOINT for the same stage revision.
14. There is no other unrelated unresolved blocker.
```

Any failure:

```text
FAIL CLOSED
DO NOT COERCE SEMANTICS
DO NOT MUTATE THE CURRENT BLOCKED RUN ANALYTICALLY
```

## 8. Thales existing-run applicability record

```text
COMPANY
= THALES

RUN_ID
= ffbb1518-149a-4224-81a3-628b92e9a151

CONTRACT_SET_SHA256
= 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e

ACTIVE INTEGRATION CHECKPOINT
= e92e915b-f9cb-4373-8c8f-055bbc7bf753@1

ACTIVE INTEGRATION CHECKPOINT SHA256
= 2bd2360a6848c2330a4a6aa340acf8db01deb928803db1807320834985cab4b2

AUTHORITATIVE SOURCE ARTIFACT
= f8dfa8cb-d8f4-4cd5-9518-00af8395717f@1

AUTHORITATIVE SOURCE PATH
= RETURN_QUALITY.canonical_verdict_fields.roic_ex_goodwill

AUTHORITATIVE SOURCE VALUE
= NOT_INTERPRETABLE

CORROBORATING CALCULATION ARTIFACT
= 89386e1b-262e-4d73-a59d-63ce4b3ded2b@2

CORROBORATING CALCULATION ID
= CALC-ROIC-XGW-004

PRE-PATCH OBSERVED RUN STATE_VERSION
= 10

PRE-PATCH OBSERVED INTEGRATION STAGE STATE_VERSION
= 2
```

The state versions above are audit observations only and MUST be re-read immediately before the resume RPC.

## 9. Required regression matrix

V2.0.6 requires proof for:

```text
A. all pre-existing roic_ex_goodwill returnValue forms remain valid
B. ROIC_EX_GOODWILL NOT_INTERPRETABLE exact preservation
C. unsupported ROIC_EX_GOODWILL string rejection
D. frozen V1 still rejects ROIC_EX_GOODWILL NOT_INTERPRETABLE
E. V2.0.6 schema is a strict field-local additive successor to V2.0.5
F. unchanged global specialState / returnValue
G. unchanged V2.0.3 ROIIC compatibility
H. unchanged V2.0.4 STANDARD_ROIC compatibility
I. unchanged V2.0.5 ROIC_TREND UNKNOWN compatibility
J. NOT_INTERPRETABLE remains invalid at non-authorized fields
K. I2 deterministic non-effect
L. I3-B exact NOT_INTERPRETABLE persistence-boundary preservation
M. full V2 / I2 / I3-B / OroTitan unit regression
N. PostgreSQL migration / Registry regression
O. lint + typecheck + production build
P. historical non-effect and zero snapshot rewrite
Q. zero contract-pin mutation
R. no upstream analytical artifact rewrite
```

A failed regression fails the patch closed.

## 10. Deployment state machine

```text
AUTHORIZED
= V2.0.6 authority bytes committed

IMPLEMENTED
= V2.0.6 schema + validator switch + regression bytes committed

REGRESSION_PASSED
= required CI and build matrix succeeds for the exact branch head / merge candidate

PREVIEW_DEPLOYED
= Vercel preview for the exact tested commit is READY

PRODUCTION_MERGED
= exact reviewed patch is merged to main

PRODUCTION_DEPLOYED
= Vercel production deployment is READY and reports the exact main merge commit

HASH_VERIFIED
= authority/schema/validator/tests bytes from the production commit are re-read
  + Git blob identifiers are recorded
  + SHA-256 values are computed from those exact UTF-8 bytes
  + production deployment commit identity equals the verified Git commit

EXISTING_RUN_ADMITTED
= every §7 condition is re-read from live Registry/artifact state and passes

RESUMED
= public.resume_orotitan_stage succeeds with live optimistic-concurrency versions
```

## 11. Registry optimistic-concurrency resume

For an admitted blocked Integration run, the authorized lifecycle transition is:

```text
public.resume_orotitan_stage(
  p_run_id,
  p_stage_code,
  p_expected_run_state_version,
  p_expected_stage_state_version,
  p_idempotency_key,
  p_request_fingerprint_sha256
)
```

Immediately before invocation:

```text
RE-READ run.state_version
RE-READ INTEGRATION.state_version
VERIFY lifecycle_status = BLOCKED
VERIFY active_manifest_kind = CHECKPOINT
VERIFY active_manifest_artifact_id/version = e92e915b-f9cb-4373-8c8f-055bbc7bf753@1
VERIFY active manifest SHA256 = 2bd2360a6848c2330a4a6aa340acf8db01deb928803db1807320834985cab4b2
VERIFY blocker_summary contains only the admitted compatibility blocker
VERIFY READY_TO_PUBLISH = NO
VERIFY no PUBLISH_AUTHORIZED event
VERIFY current_snapshot_id IS NULL
```

The resume RPC changes lifecycle state only. It does not alter analytical artifacts, contract pins, the checkpoint body, I2, I3-B, or publication state.

## 12. Production and publication boundary

Until:

```text
HASH_VERIFIED = YES
AND
EXISTING_RUN_ADMITTED = YES
```

the following remain forbidden:

```text
persist_orotitan_research_snapshot_v2
record_orotitan_publish_authorization
record_orotitan_publish_result
READY_TO_PUBLISH = YES
```

After legal resume, Integration may continue from the authoritative checkpoint under the unchanged Integration contract. Canonical snapshot persistence may occur only through unchanged I3-B after complete validation. Production publication remains separately gated by:

```text
GO PUBLISH THALES
```

V2.0.6 itself never grants publication authorization.
