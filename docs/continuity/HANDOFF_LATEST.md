# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-POST-C7-20261001-068`

## Current state

ChatGPT operating protocol:

`OROTITAN_CHATGPT_OPERATING_PROTOCOL_V1_FREEZE_V1.0`

Status:

`FROZEN_V1_0`

Post-C7 architectural planning:

`CLOSED`

## Regression battery

Baseline protocol battery executed against:

- merged CI baseline;
- frozen 25-scenario acceptance matrix;
- live Supabase registry in read-only preflight mode.

Result:

`FOUNDATION_PASS_WITH_IMPLEMENTATION_GAPS`

Verified:

- VNext CI baseline = PASS;
- Screener CI baseline = PASS;
- RLS enabled on core registry tables;
- critical registry RPCs are service-role only;
- checkpoint/finalize/reopen version guards exist;
- idempotency controls exist;
- artifact resolver checks SHA-256 + authority class;
- publication remains separate from SAVE.

Security debt recorded:

- RLS-without-policy posture must be explicitly documented as service-only;
- 3 legacy functions have mutable `search_path` and must be hardened before final production hardening.

## Vertical slice

`VERTICAL_SLICE_READY = FALSE`

15 scenarios require implementation closure before vertical slice.

They cluster into:

1. Data Contracts V2
2. Process Engine V2
3. ChatGPT ↔ Supabase bridge

## Active execution action

`RUN_DATA_CONTRACTS_V2_CI_AND_REVIEW`

## Product invariant

OroTitan product surface / vitrine remains:

`FRENCH_FIRST_UI = TRUE`

Machine contracts and internal canonical vocabulary may remain English.

## Data Contracts V2 candidate

Drafted on branch `post-c7-data-contracts-v2-design-001`.

Current package:
- composable analytical JSON Schema;
- semantic validator;
- targeted regression tests;
- no live Supabase mutation.

Exact next action:

`RUN_DATA_CONTRACTS_V2_CI_AND_REVIEW`
