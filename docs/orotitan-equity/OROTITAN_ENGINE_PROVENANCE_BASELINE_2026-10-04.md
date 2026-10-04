# OROTITAN_ENGINE_PROVENANCE_BASELINE_2026-10-04

STATUS = OBSERVED_REGISTRY_BASELINE  
OBSERVED_AT = 2026-10-04  
PURPOSE = establish the initial provenance baseline before future engine generations

## Current engine tuple

```text
PROCESS_VERSION = 2.0
PILOTAGE_CONTRACT_VERSION = 2.0
CONTRACT_SET_SHA256 = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
ENGINE_STATUS = CURRENT
```

## Current published corpus

Observed current canonical snapshot inventory:

```text
04_SCREENER_SCHEMA_V2 / 2.0.0 = 34 dossiers
04_SCREENER_SCHEMA_V1 / 1.0.0 = 1 historical snapshot
```

The 34 currently published V2 analyses resolve to the current engine tuple above.

Examples verified:

- Intuitive Surgical
- Visa
- Microsoft
- TotalEnergies
- Booking Holdings

All use:

```text
process_version = 2.0
pilotage_contract_version = 2.0
contract_set_sha256 = 1116ca12...
```

## Legacy baseline

A prior Qualys run remains historically identifiable as:

```text
PROCESS_VERSION = 1.0
PILOTAGE_CONTRACT_VERSION = 1.0.1
CONTRACT_SET_SHA256 = 34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63
ENGINE_STATUS = LEGACY
```

The later/current Qualys publication is V2 and must not erase the historical V1 identity.

## Governance rule established

```text
SCHEMA VERSION != ENGINE VERSION
```

Engine identity is based on the authoritative contract fingerprint tuple, not on UI labels or snapshot schema alone.

Future methodology changes must create a new engine fingerprint and existing snapshots remain attached to their original engine.
