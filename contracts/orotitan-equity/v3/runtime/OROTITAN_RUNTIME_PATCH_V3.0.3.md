# OROTITAN_RUNTIME_PATCH_V3.0.3

**Status:** PRODUCTION RUNTIME SUCCESSOR
**Scope:** new controlled-run admission + valuation-date alignment + blocked-defect methodology successor
**Methodology change:** VALUATION DATE ALIGNMENT ONLY
**Prior Contract Set rewrite:** FORBIDDEN
**Historical run rewrite:** FORBIDDEN
**Canonical publication authorization:** NO

## 1. Active authority

For new runs:

```text
OROTITAN_RUNTIME_BOOTSTRAP_V3.0.3
CONTRACT_SET_SHA256 = 3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2
VALUATION_DATE_ALIGNMENT_AUTHORITY = OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0
DCF_TIMING_AUTHORITY = OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0
```

The prior active Contract Set remains immutable:

```text
257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c
```

No historical run is rebound.

## 2. Runtime invariants

```text
ACTIVE_FOR_NEW_RUNS = YES
HISTORICAL_RUN_REBINDING = NO
PRIOR_CONTRACT_SETS_PRESERVED = YES
METHODOLOGY_SUCCESSOR_REPLAY_SUPPORTED = YES
CANONICAL_PUBLICATION_AUTHORIZED = NO
```

## 3. Controlled blocked-parent repair route

The successor RPC may admit a historical `BLOCKED` parent only when the database proves the exact V3 valuation timing/denominator authority defect under the prior Contract Set.

The route requires transactional locks on the parent run, Deep Dive stage and dossier, exact parent run-state CAS, same cutoff, exact identity/routing, unpublished/non-cancelled state, no current canonical snapshot and the exact blocker tuple.

The route must not change the parent to `ACTIVE`, remove the blocker or modify any parent artifact.

## 4. Replay semantics

`SUCCESSOR` is lineage, not a new legal `RUN_TYPE`.

```text
RUN_TYPE = INITIAL
PARENT_RUN_ID = exact historical run
BASELINE_SNAPSHOT_ID = NULL
DATA_CUTOFF = parent DATA_CUTOFF
FIRST_REGISTRY_STAGE = RESEARCH
RESEARCH_SCOPE = SAME_CUTOFF_NON_VALUATION_REVALIDATION_ONLY
FIRST_ANALYTICAL_PHASE = VALUATION_AFTER_REVALIDATION
```

Only exact persisted non-valuation artifacts may cross via governed revalidation lineage. Parent valuation outputs and parent `VALUATION_LOCK` remain historical-only.

## 5. Publication boundary

Runtime activation authorizes no publication and no `CURRENT_SNAPSHOT` pointer movement.
