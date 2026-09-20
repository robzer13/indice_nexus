# OROTITAN_RUNTIME_PATCH_V3.0.1

**Status:** PRODUCTION RUNTIME SUCCESSOR  
**Scope:** new controlled run admission only  
**Methodology change:** NO  
**V3 Contract Pin Pack mutation:** NO  
**V2 Contract Pin Pack mutation:** NO  
**Historical run rewrite:** FORBIDDEN  
**Canonical publication authorization:** NO

## Purpose

This patch makes the frozen V3 Contract Set production-admissible for **new controlled runs** while preserving every historical run under its persisted Contract Set.

The active new-run authority is:

```text
OROTITAN_RUNTIME_BOOTSTRAP_V3.0.1
CONTRACT_SET_SHA256 = 8d9596911b98d8c2125e5a0a19997f1620cc9034efc99bf8b5a763450bf9c0cf
```

V2 remains immutable historical authority for runs already pinned to:

```text
1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
```

## Registry admission

No Registry schema migration is introduced by this patch.

The live Registry admission boundary remains the controlled `create_orotitan_run` RPC plus the existing contract-pin completeness and deterministic Contract Set hash functions. V3 is admitted only when the exact 13-pin set is complete and recomputes to the frozen V3 Contract Set hash.

No existing `orotitan_runs.contract_pins`, `contract_set_sha256`, `data_cutoff`, stage row, artifact, event, snapshot or canonical pointer is rewritten.

## Controlled methodology-replay successor

For a pure methodology replay of an earlier run:

```text
RUN_TYPE = INITIAL
PARENT_RUN_ID = historical run_id
BASELINE_SNAPSHOT_ID = NULL
DATA_CUTOFF = parent DATA_CUTOFF
FIRST_STAGE = RESEARCH
```

The parent is read-only. The successor receives the V3 pins at creation. Identity is bound through the existing controlled Registry path.

## Production boundary

This runtime activation does not create a company run and does not authorize publication.

```text
GO PUBLISH <COMPANY>
```

remains a separate authorization boundary.
