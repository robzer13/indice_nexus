# OroTitan Assisted Analysis Bridge v0.1

## Purpose

This is the interim post-C7 operating model until materially stronger local hardware is available.

The objective is not to maximize automation percentage. The objective is to minimize analyst friction while increasing analytical quality and preserving OroTitan's evidence, schema, certification and publication controls.

## Core principle

```text
AUTOMATE DETERMINISTIC WORK
+
ASSIST HIGH-JUDGMENT WORK
+
VALIDATE EVERYTHING BEFORE PERSISTENCE
```

A language-model answer is not an authoritative research artifact merely because it is well written.

The authoritative object remains the validated and certified persisted artifact.

## Target workflow

```text
RESEARCH SOURCES / EVIDENCE LEDGER
        ↓
DETERMINISTIC TASK BUNDLE BUILDER
        ↓
TASK_PACKET.json
PROMPT.md
EXPECTED_OUTPUT_SCHEMA.json
CONTROL_CARD.md
        ↓
INTERACTIVE HIGH-CAPABILITY LLM
        ↓
STRUCTURED RESULT
        ↓
LOCAL IMPORT + VALIDATION
        ↓
PASS ──────────────→ HUMAN ANALYTICAL CONFIRMATION
  │                              ↓
  │                     CERTIFIED ARTIFACT
  │                              ↓
  │                    NORMAL OROTITAN PIPELINE
  │
  └─ FAIL → MINIMAL REPAIR REQUEST
                   ↓
           INTERACTIVE LLM
                   ↓
             RE-VALIDATE
```

## What should remain automated

Automation should own all operations that are reproducible and objectively checkable:

- source and artifact identity;
- evidence IDs;
- packet hashes;
- prompt versions;
- schema versions;
- task/run IDs;
- JSON parsing;
- evidence-reference existence;
- conflict-reference existence;
- deterministic semantic rules;
- required-field coverage;
- unknown preservation;
- duplicate/reference checks;
- persistence;
- run logs;
- certification metadata;
- pre-publication technical controls.

The analyst should never have to manually recreate evidence IDs or rebuild the prompt from multiple files.

## What should remain assisted / human

Judgment-heavy tasks should not be delegated to a weak local model merely to preserve automation:

- moat strength and durability;
- interpretation of contradictory evidence;
- weak-link prioritization;
- materiality;
- management quality;
- capital allocation;
- runway;
- business-model nuance;
- final qualitative conclusion;
- refresh impact assessment.

The interactive LLM can propose the analysis. The analyst remains responsible for accepting, rejecting or correcting the judgment.

## Operator workflow target

Normal module execution should become:

1. Run one prepare command.
2. Open the generated bundle in ChatGPT.
3. Return the structured result.
4. Run one import/validate command.
5. Review the concise control card.
6. Confirm, or send the generated repair request back to the model.

No repeated prompt engineering. No manual evidence-ID transcription. No silent normalization.

## Proposed CLI surface

Initial names are implementation targets, not frozen public API:

```text
npm run orotitan:assist:prepare -- --company <company> --module <module>
npm run orotitan:assist:import -- --run <run_id> --result <path>
npm run orotitan:assist:status -- --run <run_id>
```

A later convenience layer may place the generated prompt on the clipboard and accept a pasted JSON response, but clipboard automation is not required for v0.1.

## Bundle contract

### TASK_PACKET.json

Contains:

- company;
- run ID;
- module;
- data cutoff;
- packet version;
- packet hash;
- evidence items;
- conflicts;
- explicit unknowns;
- allowed evidence/reference IDs;
- output constraints.

### PROMPT.md

Contains only:

- task instructions;
- frozen analytical definitions;
- judgment boundaries;
- exact response requirements;
- reference to the packet;
- prohibition on inventing evidence;
- handling of conflicts and unknowns.

The evidence itself should not be duplicated into an uncontrolled prose prompt if the packet already contains it.

### EXPECTED_OUTPUT_SCHEMA.json

Machine-readable output contract.

### CONTROL_CARD.md

Human-facing run metadata:

- company;
- module;
- packet hash;
- prompt hash;
- expected schema;
- source cutoff;
- validation status;
- blockers;
- next action.

## Import contract

Import must fail closed.

A result is rejected when:

- run/task identity mismatches;
- packet hash mismatches;
- JSON is invalid;
- schema is invalid;
- an evidence ID does not exist;
- a conflict ID does not exist;
- a deterministic semantic rule fails;
- required UNKNOWN handling is missing;
- the output attempts to bypass judgment boundaries.

No importer may silently fix evidence identities.

## Repair loop

When an import fails, the system should create a minimal repair bundle rather than asking the analyst to diagnose the JSON manually.

The repair request should expose:

- failing JSON paths;
- validator codes;
- expected contract;
- allowed IDs relevant to the failure;
- explicit instruction to preserve all unaffected fields.

It should not alter the packet, prompt contract, evidence or historical failed result.

## Human authority

The analyst remains authoritative for qualitative judgment and materiality.

A validation PASS proves contract compliance. It does not prove that the investment judgment is correct.

Analytical confirmation therefore remains a separate state from technical validation.

## Publication boundary

This bridge does not change OroTitan publication rules.

Validated analysis is not automatically published.

Existing certification, Integration, reconciliation and pre-publication controls remain downstream.

## Why this survives the future hardware upgrade

The bridge deliberately isolates the inference step.

When stronger hardware is purchased:

```text
INTERACTIVE LLM
→ replace with
LOCAL INFERENCE ADAPTER
```

The packet builder, prompt contract, schemas, validators, result importer, persistence and certification flow remain the same.

That means work invested in the assisted workflow is not throwaway work.

## First pilot

Use:

```text
MOAT_EVIDENCE_AUDIT_ASSISTED_V0_3
```

because Gate 18 already generated extensive calibration evidence for this module.

Pilot success should measure:

- operator time;
- number of manual transformations;
- schema pass rate;
- invalid-reference rate;
- deterministic-semantic pass rate;
- human-quality defects;
- repair-loop count;
- analyst confidence;
- readiness for persistence.

Do not expand the bridge to other judgment modules until this pilot is clean enough to justify reuse.
