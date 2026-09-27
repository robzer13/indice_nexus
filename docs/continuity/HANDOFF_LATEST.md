# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-017`

## Standing authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE for zero-cost technical execution. Nonzero external monetary cost still requires explicit user authorization.

## Qwen3.5 disposition

The clean `think:false` Constellation retry remains an engineering PASS, but human adjudication found a critical evidence-grounding failure.

Material defect:
- `E-045` states a replacement RFP;
- the model output stated a completed replacement after approximately 20 years.

Additional quality defects included duplicated weak-link candidates, low-information unresolved points, and priority selection that underweighted serial-acquirer-specific moat evidence.

Disposition:

`STOP_QWEN3_5_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE`

No global Qwen3.5-family failure is inferred. No model winner or routing decision exists.

## Gemma 3 4B static review

Candidate:

`gemma3:4b-it-q4_K_M`

Public identity checked 2026-09-27:
- Ollama public digest prefix `a2af6cc3eb7f`;
- approximately 3.3GB;
- 128K context;
- Q4_K_M target;
- text and image input.

Gemma Terms of Use last modified 2026-04-01 were reviewed. The user explicitly accepted the Gemma Terms for this OroTitan calibration path.

Static hardware fit:

`PLAUSIBLE_BUT_TIGHT_UNPROVEN`

## Authorized next action

Exactly one pinned local download is authorized:

`G18-PHASEC-GEMMA3-4B-DOWNLOAD-AUTH-001`

Runner:

`scripts/vnext-gate18-phase-c-gemma3-4b-download-verify.ts`

Scope:
- pull only `gemma3:4b-it-q4_K_M`;
- verify exact local tag;
- capture full digest, size and Ollama metadata;
- require digest prefix `a2af6cc3eb7f`;
- verify Q4_K_M if exposed;
- no load smoke;
- no prompt;
- no model inference;
- no automatic retry;
- no production mutation.

## Exact next action

```text
EXECUTE_GEMMA3_4B_PINNED_DOWNLOAD_VERIFY
```

If and only if the pinned download passes, prepare a separate context-4096 load-only memory preflight using the observed full digest.
