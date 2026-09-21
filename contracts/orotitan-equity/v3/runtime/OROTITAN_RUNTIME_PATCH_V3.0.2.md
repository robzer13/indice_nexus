# OROTITAN_RUNTIME_PATCH_V3.0.2

**Status:** PRODUCTION RUNTIME SUCCESSOR  
**Scope:** new controlled run admission + DCF timing authority + methodology-replay CAS  
**Methodology change:** DCF TIMING ONLY  
**Prior V3 Contract Set rewrite:** FORBIDDEN  
**V2 Contract Set rewrite:** FORBIDDEN  
**Historical run rewrite:** FORBIDDEN  
**Canonical publication authorization:** NO

## Purpose

This patch activates the global DCF timing authority for new runs and closes the transactionality gap in controlled methodology-replay successor creation.

The active new-run authority is:

```text
OROTITAN_RUNTIME_BOOTSTRAP_V3.0.2
CONTRACT_SET_SHA256 = 257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c
DCF_TIMING_AUTHORITY = OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0
```

Historical V2 runs remain pinned to:

```text
1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
```

Historical pre-DCF V3 runs remain pinned to:

```text
8d9596911b98d8c2125e5a0a19997f1620cc9034efc99bf8b5a763450bf9c0cf
```

No historical run is rebound.

## Controlled methodology-replay successor

`SUCCESSOR` remains lineage, not a legal `RUN_TYPE`.

For a same-cutoff methodology replay with no canonical snapshot:

```text
RUN_TYPE = INITIAL
PARENT_RUN_ID = exact historical run
BASELINE_SNAPSHOT_ID = NULL
DATA_CUTOFF = parent DATA_CUTOFF
FIRST_REGISTRY_STAGE = RESEARCH
RESEARCH_SCOPE = exact same-cutoff non-valuation revalidation only
FIRST_ANALYTICAL_PHASE = VALUATION after successful revalidation
```

Creation must use `create_orotitan_methodology_successor_run`. The RPC locks the parent and dossier transactionally and verifies the expected parent state version, ACTIVE status, DEEP_DIVE stage, exact cutoff, exact identity, exact historical Contract Set, unpublished/non-cancelled state and absence of a current canonical snapshot before insertion.

The generic `create_orotitan_run` path remains authoritative for ordinary new runs.

## Cross-run authority

Only exact persisted non-valuation artifacts may cross through explicit hash-verified lineage. Parent valuation outputs and parent `VALUATION_LOCK` are historical-only and may not be revalidated into the successor.

No post-cutoff evidence, fundamental judgment change or calibration to old valuation outputs is authorized.

## Production boundary

This patch and its migration create no company run and authorize no publication.

```text
GO PUBLISH <COMPANY>
```

remains separately required.
