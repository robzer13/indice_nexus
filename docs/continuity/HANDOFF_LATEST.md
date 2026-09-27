# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-035`

## Qwen3 8B Constellation C4

Engineering:

`CLEAN_ENGINEERING_PASS`

Human adjudication:

`CRITICAL_FAILURE`

Key adjudication findings:
- E-045 is still overstated from a replacement RFP/process into an eventual replacement outcome;
- E-063 is re-asked as an unresolved low-retention/transparency question even though the packet already fixes the epistemic boundary: nondisclosure is not evidence of low retention;
- C-005 is surfaced correctly and remains unresolved;
- no evidence/conflict IDs are invented;
- priority selection remains weak: direct switching-cost / serial-acquirer inputs such as E-043 and E-067 are omitted from priority findings, while E-066 is relegated to an unresolved point.

Disposition:

`STOP_QWEN3_8B_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE`

No global Qwen3-family failure is inferred. Qwen3 8B is not admitted for production from current evidence. No retry or further Qwen3 8B inference is authorized.

## Next candidate — Ministral 3 3B

Candidate:

`MINISTRAL3_3B_INSTRUCT_2512_Q4_K_M`

Exact Ollama tag:

`ministral-3:3b-instruct-2512-q4_K_M`

Pinned public identity:
- digest prefix: `f04aa1c738f6`;
- artifact: ~3.0 GB;
- parameters: 3.85B;
- quantization: Q4_K_M;
- context: 256K;
- license: Apache-2.0.

Authorization:

`G18-PHASEC-MINISTRAL3-3B-DOWNLOAD-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-ministral3-3b-download-verify.ts`

Boundaries:
- one exact-model download only;
- identity verification only;
- no load smoke;
- no inference;
- no automatic retry;
- no model switch;
- no paid execution;
- no production mutation.

## Exact next action

```text
EXECUTE_MINISTRAL3_3B_PINNED_DOWNLOAD_AND_IDENTITY_VERIFY
```
