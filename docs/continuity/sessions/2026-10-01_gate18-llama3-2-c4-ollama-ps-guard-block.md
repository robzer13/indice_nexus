# Gate 18 — Llama 3.2 C4 Ollama residency guard block

Resume ID: `VNEXT-G18-C-20261001-057`

Observed status: `BLOCKED_BEFORE_INFERENCE`.

Observed error:

`VNEXT_GATE18_PHASE_C_C4_LLAMA3_2_3B_OLLAMA_PS_CHECK_FAILED`

The failure occurred at the residency guard before model verification, RAM guard, packet construction and provider generation.

No semantic inference occurred.

The single inference authorization remains unconsumed with one run remaining.

Remediation:
replace subprocess `ollama ps` with loopback HTTP `GET /api/ps`.

All C4 inference parameters remain unchanged.
