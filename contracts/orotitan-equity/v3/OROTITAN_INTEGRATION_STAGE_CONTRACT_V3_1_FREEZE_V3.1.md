# OROTITAN_INTEGRATION_STAGE_CONTRACT_V3.1 - FREEZE V3.1

**Status:** FROZEN - V3.1
**Base incorporated by reference:** OROTITAN_INTEGRATION_STAGE_CONTRACT_V3_FREEZE_V3.0 @ SHA256 09534d88e3ac4d37cfa0f55b7e762692cd198b7baff424ed169396bdd6851759
**Depends on:** V3.1 Process + V3.1 Pilotage + V3.1 Deep Dive
**Projection authority:** 04_INTEGRATION_SPEC_V3.1 + 04_SCREENER_SCHEMA_V3.1
**Admission authority:** I3B_VALIDATED_SNAPSHOT_WRITER_V1.2
**Methodology change:** NO

All V3.0 Integration Stage rules remain unchanged except for successor dependency pins.

Integration is admitted only after the V3.1 Deep Dive is complete and `READY_FOR_INTEGRATION = YES`.

Integration must preserve the exact V3.1 timing tuple and denominator representation supplied by the certified Deep Dive. It may not:
- change `DATA_CUTOFF`, `REFERENCE_PRICE_DATE` or `VALUATION_DATE`;
- move a share-count bound between dates;
- create an EV bridge;
- scalarize a range;
- change valuation or Certification judgments.

Historical snapshots remain immutable. Publication still requires separate authorization.
