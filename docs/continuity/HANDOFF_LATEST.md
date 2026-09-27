# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-009`

## Standing execution authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` is ACTIVE.

The user has permanently authorized OroTitan zero-cost execution steps without repeated confirmation.

Operational rule:

- zero-cost local or repository execution: proceed without reprompt;
- nonzero external monetary cost: obtain explicit user authorization before spending.

This includes zero-cost model downloads, memory/load preflights, local inference, methodologically justified bounded retries, candidate/model changes, diagnostics, tests, PR/CI/merge operations, and continuity maintenance.

Protocol guards remain in force: historical results are immutable, no retroactive pass, private evidence remains private, identity/contract mismatches fail closed, and methodology changes remain versioned.

## Current Gate 18 state

Phi-4 C4 expansion has been stopped and Phi-4 is retained as calibration evidence.

Qwen3.5 4B has been downloaded and identity-pinned:

- model: `qwen3.5:4b-q4_K_M`
- digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`
- size: 3,389,983,735 bytes
- quantization: `Q4_K_M`
- Ollama reported parameter size: `4.7B`

No Qwen3.5 inference has occurred.

## Current authorized step

A single context-4096 load-only memory preflight is authorized under the standing zero-cost authorization.

The runner must:

- verify exact model identity;
- load without prompt or semantic generation;
- measure RAM/VRAM and `ollama ps`;
- unload explicitly;
- stop.

## Exact next action

```text
EXECUTE_QWEN3_5_4B_CONTEXT4096_LOAD_ONLY_MEMORY_PREFLIGHT
```

No further authorization prompt is required for subsequent zero-cost steps. Only a proposed nonzero external cost requires explicit user approval.
