# OROTITAN_ENGINE_PROVENANCE_SPEC_V0.1

STATUS = IMPLEMENTATION_METADATA_SPEC  
METHODOLOGY_CHANGE = NO

## 1. PURPOSE

Make three independent notions visible:

1. engine generation;
2. research freshness;
3. market-price freshness.

These must never be conflated.

## 2. ENGINE IDENTITY

Current engine identity is an exact tuple:

```text
PROCESS_VERSION
PILOTAGE_CONTRACT_VERSION
CONTRACT_SET_SHA256
```

Current tuple:

```text
PROCESS_VERSION = 2.0
PILOTAGE_CONTRACT_VERSION = 2.0
CONTRACT_SET_SHA256 = 1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e
```

Classification:

- exact tuple → CURRENT;
- same/newer process family but other fingerprint → PREVIOUS;
- process major lower than current → LEGACY;
- missing authority metadata → UNKNOWN.

Schema version alone is never sufficient to classify the engine.

## 3. RESEARCH FRESHNESS

Research freshness is operational metadata only.

Initial UI thresholds:

- RECENT: age <= 90 days;
- AGING: 91–180 days;
- STALE: >180 days;
- UNKNOWN: no valid data cutoff.

This status does not change scores or certification.

## 4. MARKET FRESHNESS

Market-price freshness remains separately derived from market-price timestamps.

A fresh market price cannot make stale research current.

A recent research analysis cannot make an old market price current.

## 5. DISPLAY

Company pages and screener should expose:

- Moteur actuel / précédent / legacy;
- short engine fingerprint;
- data cutoff;
- analysis age;
- research freshness;
- market freshness.

## 6. HISTORY

Historical snapshots and runs retain their original engine identity.

No migration may overwrite an old fingerprint with the current one.
