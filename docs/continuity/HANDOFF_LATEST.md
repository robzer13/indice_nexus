# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-C-20260927-022`

## Standing authority

`OROTITAN-STANDING-TECHNICAL-AUTH-002` remains ACTIVE for zero-cost technical execution. Nonzero external monetary cost still requires explicit user authorization.

## Gemma 3 first Constellation C4 result

Historical run status:

`FAIL_DETERMINISTIC_SEMANTIC_CONTRACT`

Identity:
- company: Constellation Software;
- model: `gemma3:4b-it-q4_K_M`;
- digest: `a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a`;
- packet SHA256: `9a47bcf15d0c90da55349cbb3fbb2da8e9645b859a535dc8909501e89a20b6d8`;
- prompt SHA256: `0891fa34d02a47c83e8342d5da5e653a3566d90f1c63c1bfc662f4e6097c10b8`.

Execution:
- wall clock: 142518 ms;
- done reason: `stop`;
- prompt eval count: 3558;
- eval count: 683;
- output token margin: 341;
- runtime error: none;
- schema valid: true;
- semantic valid: false;
- semantic error: `VNEXT_GATE18_V10_COUNTEREVIDENCE_LINK_WITHOUT_IDS`.

Validation v1.1:
- raw presentation compliant: false;
- normalized path count: 13;
- substantive status: FAIL.

Interpretation:
The run is complete and schema-valid. The failure is deterministic and substantive after v1.1 presentation normalization. It is not an output-budget exhaustion or runtime failure.

The historical run remains FAIL. No retry is authorized.

## Deterministic forensic

Prepared runner:

`scripts/vnext-gate18-phase-c-c4-constellation-gemma3-counterevidence-link-forensic.ts`

Purpose:
1. reproduce the frozen v1.1 presentation normalization in memory;
2. identify findings where `counterevidence_ids` is empty while `counterevidence_link` is non-null;
3. set only those links to null on an in-memory diagnostic copy;
4. rerun the frozen v1.0 substantive semantic validator;
5. report whether another deterministic semantic defect remains.

Boundaries:
- no Ollama call;
- no model inference;
- no external network;
- no source artifact mutation;
- no raw narrative text printed;
- no retroactive pass;
- no automatic repair;
- no retry authority.

## Exact next action

```text
RUN_LOCAL_READ_ONLY_GEMMA3_CONSTELLATION_COUNTEREVIDENCE_LINK_FORENSIC
```

Use the private run artifact:
`calibration/vnext/private-runs/2026-09-27T160225776Z__C4_CONSTELLATION_MOAT_EVIDENCE_AUDIT_GEMMA3_4B_V1_1_CONTEXT16384_001.json`.
