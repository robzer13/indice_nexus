# OroTitan VNExT — Latest Handoff

Resume ID: `VNEXT-G18-POSTC7-20261001-063`

## Post-C7 interim strategy selected

The current local-model campaign is formally closed at C7 with:

`LOCAL_CANDIDATE_REJECTED`

The user has now selected the interim operating direction:

`ASSISTED_HUMAN_IN_THE_LOOP_ANALYSIS_BRIDGE`

This strategy is intended to remain active until materially stronger local hardware is available and a new local-model qualification campaign becomes worthwhile.

## Objective

Do not optimize for 100% automation.

Optimize for:
- lower operator friction;
- higher analytical quality;
- exact evidence traceability;
- deterministic validation;
- provider-neutral migration path;
- no production/API dependency that would need to be rewritten later.

## Target flow

```text
DETERMINISTIC EVIDENCE PREPARATION
→ ASSISTED TASK BUNDLE
→ HIGH-CAPABILITY INTERACTIVE LLM
→ STRUCTURED RESULT
→ LOCAL VALIDATION
→ TARGETED REPAIR LOOP IF NEEDED
→ HUMAN ANALYTICAL CONFIRMATION
→ CERTIFIED ARTIFACT
→ NORMAL OROTITAN DOWNSTREAM CONTROLS
```

## Automation boundary

Automate:
- packet/prompt build;
- IDs/hashes/schema;
- result import;
- deterministic validation;
- repair-request generation;
- persistence/logging;
- technical control cards.

Keep assisted/human:
- moat judgment;
- conflict interpretation;
- materiality;
- runway;
- capital allocation;
- management/governance;
- final qualitative synthesis;
- refresh impact assessment.

## First implementation target

Pilot module:

`MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3`

Implement:
1. task-bundle contract;
2. prepare CLI;
3. import/validation CLI;
4. minimal repair-prompt generator;
5. first assisted pilot.

No routing freeze.
No production mutation.
No model winner.

## Current exact next action

`IMPLEMENT_ASSISTED_TASK_BUNDLE_CONTRACT_AND_MINIMAL_PREPARE_IMPORT_BRIDGE_V0_1`
