# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-024`

## Gemma 3 final disposition

Historical Constellation run remains:

`FAIL_DETERMINISTIC_SEMANTIC_CONTRACT`

Deterministic forensic chain:
1. all three priority findings had non-null `counterevidence_link` with empty `counterevidence_ids`;
2. after narrow in-memory normalization, the validator exposed `VNEXT_GATE18_V10_UNKNOWN_CONFLICT_REF`;
3. the only canonical conflict ID is `C-005`, while the model invented `C-006`;
4. `C-006` appeared in one material-conflict entry and two unresolved points;
5. after removing only those unknown references on the in-memory diagnostic copy, the frozen v1.0 semantic validator passed.

Therefore:
- two deterministic substantive defect classes are established;
- no third known deterministic defect remains;
- the historical run is not reclassified;
- engineering PASS was not reached;
- formal human-quality adjudication was not reached;
- no Gemma retry or additional Gemma C4 cell is authorized.

Disposition:

`STOP_GEMMA3_C4_EXPANSION_RETAIN_CALIBRATION_EVIDENCE`

This does not imply global Gemma-family failure.

## Candidate-registry refresh

The previous fallback `qwen3:8b` remains on hold because its ~5.2GB artifact is a poor fit for the observed 7.84 GiB RAM / 4 GiB VRAM envelope.

Refreshed zero-cost candidates:
1. `granite4:3b` — ~2.1GB, 3.4B, Q4_K_M, 128K, Apache-2.0;
2. `ministral-3:3b-instruct-2512-q4_K_M` — ~3.0GB, 256K, Apache-2.0;
3. `llama3.2:3b` — ~2.0GB, 128K, older family option;
4. `qwen3:8b` — retained on hardware-fit hold.

Selected next candidate:

`GRANITE4_3B_OLLAMA_Q4_K_M`

Selection basis:
- new model family;
- materially smaller artifact than recent candidates;
- Apache-2.0;
- current Granite 4 documentation emphasizes instruction following and tool calling;
- highest expected information gain per memory cost among refreshed options.

## Granite 4 3B download authorization

Authorization:

`G18-PHASEC-GRANITE4-3B-DOWNLOAD-AUTH-001`

Exact public target:
- tag: `granite4:3b`;
- expected digest prefix: `89962fcc7523`;
- expected quantization: `Q4_K_M`;
- expected artifact class: ~2.1GB;
- no load smoke;
- no prompt;
- no inference;
- no automatic retry;
- no automatic model switch.

Runner:

`scripts/vnext-gate18-phase-c-granite4-3b-download-verify.ts`

## Exact next action

```text
EXECUTE_GRANITE4_3B_PINNED_DOWNLOAD_AND_IDENTITY_VERIFY
```
