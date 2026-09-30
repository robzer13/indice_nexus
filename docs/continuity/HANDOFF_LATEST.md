# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260930-053`

## Llama 3.2 3B — context8192 load-only preflight passed

Pinned identity:
- model: `llama3.2:3b-instruct-q4_K_M`;
- digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`;
- quantization: Q4_K_M.

Observed context8192 load-only result:
- status: `PASS_LOAD_ONLY_MEASURED`;
- no prompt and no semantic inference;
- loaded free RAM: 0.47 GiB;
- loaded VRAM: 2751 MiB used / 1212 MiB free;
- processor split: 32% CPU / 68% GPU;
- Ollama reported resident size: 3.4 GB;
- explicit unload completed;
- post-unload `ollama ps` empty.

Disposition:

`PASS_WITH_CRITICAL_RAM_PRESSURE`

The context8192 single-run authorization is consumed.

The RAM snapshots were again non-monotonic around the run (0.39 GiB before, 0.47 GiB loaded, 1.67 GiB after). The loaded-state free-RAM measurement remains the controlling risk signal.

This result establishes loadability at 8192 only. It does not establish inference fit, production fit, candidate admission, winner status or routing.

## Context16384 final diagnostic authorization

Exactly one context16384 load-only final diagnostic is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

Purpose:
measure exact loadability at the 16384 context used for direct C4 protocol comparability before any inference decision.

No prompt, semantic inference, automatic retry, model switch, context growth beyond 16384, production mutation, ranking, routing or publication is authorized.

No inference is pre-authorized by this step.

## Current exact next action

`EXECUTE_LLAMA3_2_3B_CONTEXT16384_LOAD_ONLY_MEMORY_PREFLIGHT`
