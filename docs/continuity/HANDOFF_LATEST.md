# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-042`

## Post-Ministral local candidate registry refresh

The 2026 registry was refreshed after the Ministral 3 3B human-quality critical failure.

Selected next candidate:

`GRANITE4_1_3B_OLLAMA_Q4_K_M`

Exact Ollama target:

`granite4.1:3b-q4_K_M`

Public identity:
- digest prefix: `6fd349357287`;
- artifact class: ~2.1GB;
- quantization: Q4_K_M;
- context: 128K;
- input: text;
- license: Apache-2.0;
- release date: 2026-04-28;
- current Ollama documentation explicitly lists structured JSON output support.

Selection logic:
- same low-memory artifact class as the already measured Granite 4 3B candidate;
- current 2026 successor release;
- targeted discriminator for contract/structured-output behavior without increasing hardware pressure;
- public capability claims are not treated as OroTitan semantic-quality evidence.

Deferred next option:

`LLAMA3_2_3B_OLLAMA_Q4_K_M`

Llama 3.2 remains the next different-family candidate if Granite 4.1 is stopped.

## Current authority

Authorized:
- one exact zero-cost `ollama pull` of `granite4.1:3b-q4_K_M`;
- local identity verification after download.

Not authorized:
- model inference;
- prompt/messages;
- load smoke;
- retry;
- context change;
- model switch;
- production mutation;
- publication.

## Exact next action

```text
EXECUTE_GRANITE4_1_3B_PINNED_DOWNLOAD_AND_IDENTITY_VERIFY
```
