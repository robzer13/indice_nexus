# Gate 18 — Post-C7 assisted analysis strategy

Resume ID: `VNEXT-G18-POSTC7-20261001-063`

## User direction

Until a materially stronger computer is purchased:
- do not insist on 100% automation;
- simplify the process as much as possible;
- improve analytical quality.

## Selected strategy

`ASSISTED_HUMAN_IN_THE_LOOP_ANALYSIS_BRIDGE`

Automate deterministic preparation, validation, persistence and controls.

Use a high-capability interactive LLM plus analyst review for judgment-heavy analytical modules.

The model output is not automatically authoritative.

Validated/certified artifacts remain the downstream contract.

## Future migration

The inference step must remain replaceable.

Future stronger hardware should allow:

`INTERACTIVE_LLM → LOCAL_INFERENCE_ADAPTER`

without replacing task bundles, schemas, validators or persistence.

## Next action

`IMPLEMENT_ASSISTED_TASK_BUNDLE_CONTRACT_AND_MINIMAL_PREPARE_IMPORT_BRIDGE_V0_1`
