# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260930-051`

## Llama 3.2 3B — context4096 load-only preflight passed

Pinned identity:
- model: `llama3.2:3b-instruct-q4_K_M`;
- digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`;
- quantization: Q4_K_M.

Observed context4096 load-only result:
- status: `PASS_LOAD_ONLY_MEASURED`;
- no prompt and no semantic inference;
- loaded free RAM: 0.34 GiB;
- loaded VRAM: 2683 MiB used / 1280 MiB free;
- processor split: 20% CPU / 80% GPU;
- Ollama reported resident size: 2.9 GB;
- explicit unload completed;
- post-unload `ollama ps` empty.

Disposition:

`PASS_WITH_CRITICAL_RAM_PRESSURE`

The context4096 single-run authorization is consumed.

This result establishes loadability at 4096 only. It does not establish inference fit, production fit, model admission, winner status or routing.

## Context8192 diagnostic authorization

Exactly one context8192 load-only diagnostic is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

No prompt, semantic inference, automatic retry, model switch, context growth beyond 8192, production mutation, ranking, routing or publication is authorized.

Context16384 is not pre-authorized.

## Current exact next action

`EXECUTE_LLAMA3_2_3B_CONTEXT8192_LOAD_ONLY_MEMORY_PREFLIGHT`
