# OROTITAN_EXECUTION_PROCESS_V3.1 - FREEZE V3.1

**Status:** FROZEN - V3.1
**Base incorporated by reference:** OROTITAN_EXECUTION_PROCESS_V3_FREEZE_V3.0 @ SHA256 8bf1817d4a3d3655b386e452b54eb50be1025ce56d4933bd1cbb5f0ef8c04dd7
**Methodology delta:** OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0 only
**Historical run rewrite:** FORBIDDEN
**Scoring formula change:** NO
**Investment-policy change:** NO
**Certification change:** NO
**Terminal-gate change:** NO
**Production activation:** NOT AUTHORIZED BY THIS DOCUMENT ALONE

All V3.0 execution rules remain unchanged except the authority binding below.

## 1. Successor authority binding

V3.1 runs pin both:
- `OROTITAN_ECONOMIC_SHARE_COUNT_METHODOLOGY_V1_FREEZE_V1.0`; and
- `OROTITAN_VALUATION_DATE_ALIGNMENT_AUTHORITY_V1_FREEZE_V1.0`.

For valuation-date alignment, the alignment authority supersedes only the date clauses explicitly listed in its supersession scope. All non-date share-count rules remain controlled by the share-count authority. DCF temporal mechanics remain controlled by `OROTITAN_DCF_TIMING_AUTHORITY_V1_FREEZE_V1.0`.

## 2. Valuation admission

Valuation may start only when the canonical timing tuple is deterministic and affected required inputs are governed at their required dates.

A denominator exact count or bound from an earlier date is not admitted at `VALUATION_DATE` without an explicit exact/hard-bounded bridge. An unresolved material FCFF EV bridge blocks FCFF equity-value closure. `UNKNOWN` remains a valid fail-closed result.

## 3. Historical firewall

No existing run is rebound to V3.1.

A same-cutoff methodology replay uses a new controlled successor run with `parent_run_id` pointing to the historical run. Cross-run reuse is limited to exact persisted artifacts admitted by explicit revalidation lineage.

## 4. Blocked-parent defect recovery

A historical parent in `BLOCKED` state may be admitted to the successor route only when the production Registry proves all of the following atomically:
- exact prior Contract Set affected by the repaired conflict;
- `CURRENT_STAGE = DEEP_DIVE`;
- exact valuation timing/denominator conflict blocker;
- Deep Dive lifecycle `BLOCKED`;
- unpublished and non-cancelled;
- no current canonical snapshot;
- exact parent state-version CAS;
- exact issuer/security/dossier/cutoff/routing equality.

This does not generally make `BLOCKED` parents successor-eligible.

## 5. Publication

This process version creates no publication authorization and never moves `CURRENT_SNAPSHOT`.
