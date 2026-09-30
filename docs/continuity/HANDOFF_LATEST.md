# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260930-054`

## Llama 3.2 3B — context16384 load-only preflight passed

Pinned identity:
- model: `llama3.2:3b-instruct-q4_K_M`;
- digest: `a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72`;
- quantization: Q4_K_M.

Observed context16384 load-only result:
- status: `PASS_LOAD_ONLY_MEASURED`;
- no prompt and no semantic inference;
- loaded free RAM: 0.52 GiB;
- loaded VRAM: 2746 MiB used / 1217 MiB free;
- processor split: 46% CPU / 54% GPU;
- Ollama reported resident size: 4.4 GB;
- explicit unload completed;
- post-unload `ollama ps` empty.

Disposition:

`PASS_WITH_HIGH_RAM_PRESSURE`

The context16384 single-run load-only authorization is consumed.

Context growth stops at 16384.

This result proves loadability at the target C4 context only. It does not prove inference fit, production fit, candidate admission, winner status or routing.

## First bounded Constellation C4 authorization

Exactly one zero-cost local inference is authorized under `OROTITAN-STANDING-TECHNICAL-AUTH-002` on the same pinned Constellation Software packet.

Fixed execution contract:
- context: 16384;
- max output: 1024;
- temperature: 0;
- timeout: 600000 ms;
- keep_alive: 0s;
- transport: loopback only;
- minimum baseline free RAM before inference: 1.0 GiB;
- raw generated content: private only;
- human adjudication required if engineering validation passes.

No automatic retry, prompt change, schema change, packet change, context change, output-budget change, temperature change, timeout change, model switch, production mutation, ranking, routing or publication is authorized.

## Current exact next action

`EXECUTE_FIRST_BOUNDED_LLAMA3_2_CONSTELLATION_C4_INFERENCE_V1_1`
