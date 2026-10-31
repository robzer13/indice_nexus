# P0-E L’Oréal — SHADOW INITIAL run receipt
Date: 2026-10-31
Status: CREATED AND INDEPENDENTLY READ BACK
Authority: user GO P0-E, L’Oréal, SHADOW, V3.1, 2026-10-30.
No approval to start Research stage or publish.

## Fixed identity
- ISIN: FR0000120321; ticker OR; MIC XPAR; issuer L’Oréal S.A.
- Issuer UUID: c431698b-3b5f-40c4-bf24-8bf4612db876
- Security UUID: 44bf0acd-9420-42f3-bdd1-1e725740154d
- Dossier UUID: e30bd3e4-618c-41a2-93b5-015fc2ea708e
- RUN UUID: **204390a1-5cba-4b9c-9821-ca0c7df43eca**
- SHADOW Supabase: awgsurdyvsyolcgpnygh
- Production Supabase: cugpgtzygqqlxetyeven — NOT TOUCHED

## Frozen methodology
- Analytic profile: OROTITAN_V3_1
- Runtime bootstrap: 3.0.3
- Frozen pin pack: 3.0.2
- Contract Set SHA-256: 3644e501909326af04d66730fa30b1ac3da0d82fb6b717202af6d948f3211fe2
- All 15 original pinned Git blobs independently SHA-1/ SHA-256 verified.
- Total 70,759 canonical bytes; latest verification 2026-10-31T11:28:28.147Z.
- process/pilotage versions: 3.1 / 3.1.
- Deterministic creation idempotency key:
  `orotitan:v3:create:c702919d93c9dd1d8d5d38eec397795af580196a9d4ef60743ccd9c09f147c6d`.

## Actual persisted state
- `run_type=INITIAL`, `run_status=CREATED`, `state_version=2`
- `current_stage=NULL`, `parent_run_id=NULL`, `baseline_snapshot_id=NULL`
- `published_at=NULL`, `cancelled_at=NULL`
- Event `RUN_CREATED`: 1
- Event `RUN_IDENTITY_BOUND`: 1
- No duplicate runs for issuer/cutoff (count = 1).
- Stage rows = 0; run artifacts = 0; dossier snapshots = 0.
- `orotitan_p0e_run_admission_receipts_v1`: 1 for the run.
- Receipt preserves 15 original-byte verification checks.
- `publication_authorized=false`.
- Replayed exact request in SHADOW: response `REPLAYED` with same run UUID and `state_version=2`.
- A prior real SHADOW `BEGIN/ROLLBACK` smoke test established no surviving
  run or receipt before the authorized commit.

## Engineering and audit
- Root SHADOW atomic writer:
  `public.orotitan_shadow_create_initial_run_v3_1`.
- Official Registry writers reused in same transaction:
  `create_orotitan_run` and `bind_orotitan_run_identity`.
- The SQL wrapper and original byte verifier are SHADOW-only.
- Server role required; anon/authenticated execute denied.
- Required helper grants are limited to the two immutable pinned-contract predicates.
- The Registry's historical `orotitan_generation` V1/V2 schema label is
  not an analytic methodology marker. Analytic V3.1 is proven by the persisted
  full contract pins, SHA-256 and version fields; no historical schema mutation.

### Next boundary
Stage initiation and Research evidence acquisition are not included.
Do not start a stage, claim Research findings, create an artifact or publish
without a separately authorized GO RESEARCH and its stage preflight.
