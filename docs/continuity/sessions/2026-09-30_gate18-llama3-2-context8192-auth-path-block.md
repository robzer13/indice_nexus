# Gate 18 — Llama 3.2 context8192 auth-path tooling block

Resume ID: `VNEXT-G18-C-20260930-052`

## Starting state

One context8192 load-only diagnostic was authorized and unconsumed.

## Observed command result

`BLOCKED`

Error:

`LLAMA3_2_3B_CONTEXT8192_LOAD_SMOKE_NOT_AUTHORIZED`

## Root cause

The context8192 runner referenced the consumed generic context4096 authorization artifact rather than the dedicated unconsumed context8192 authorization artifact.

## Execution boundary

The failure occurred at the local authorization guard before any Ollama request.

No model load, prompt, semantic inference or context execution occurred.

## Remediation

- Correct the runner authorization path.
- Add an anti-regression test pinning the dedicated context8192 artifact.
- Preserve the context8192 authorization as unconsumed with one run remaining.

## Next action

`EXECUTE_SAME_LLAMA3_2_3B_CONTEXT8192_LOAD_ONLY_PREFLIGHT_WITH_CORRECTED_RUNNER`
