# OROTITAN_GATE18_PHASE_B_V10_TARGETED_CLAIM_PROBE_V0.1

**Gate:** 18  
**Phase:** B company calibration  
**Status:** CALIBRATION-ONLY TARGETED REGRESSION PROBE — NOT FROZEN  
**Production mutation:** FORBIDDEN  
**Publication authority:** DISABLED  
**Model winner authority:** NONE  
**Routing authority:** NONE

## 0. Purpose

The first Adyen × LUNA v1.0 canary produced human-quality-admissible output, but it did not directly exercise the critical `counterevidence_ids`, `MIXED`, and scope-resolved-conflict paths.

This probe exists only to test those semantic behaviors under controlled instructions.

It is deliberately **not** a normal priority-selection observation and must not be used as model-ranking evidence.

## 1. Preserved contract

The probe preserves:

```text
source company = Adyen
SOURCE_RUN_ID = a8230f34-d23c-4ed8-b1e2-7cbd0ca81fbf
DATA_CUTOFF = 2026-09-19
packet scope = MOAT_INPUTS
packet identity = unchanged
GenerationSchema = GATE18_MOAT_EVIDENCE_AUDIT_OUTPUT_V0_6
GenerationSchema version = 0.6
maxOutputTokens = 4096
v1.0 system evidence-role rules = unchanged
automatic retry = forbidden
automatic model escalation = forbidden
```

Only the module question is replaced by a calibration-only targeted probe question.

## 2. Probe identity

```text
probe_id = ADYEN_CLAIM_TARGET_CORE_001
prompt_template_id = GATE18_ADYEN_CLAIM_TARGET_PROBE_V0_1
prompt_template_version = 0.1
```

## 3. Required finding 1 — switching friction

The model must produce one inferential finding at the switching-friction / relationship-persistence level.

Required directional map:

```text
support_state = MIXED
support = E-036, E-042
counterevidence = E-037, E-041
conflict = C-005
```

This directly tests whether support and counterevidence bear on the same economic proposition.

## 4. Required finding 2 — service incident existence

The model must produce one descriptive existence claim that a documented service disruption occurred.

Required directional map:

```text
support_state = SUPPORTED
support = E-055
counterevidence = []
conflict = C-010
```

`E-056` must not be counterevidence to this exact claim because peak-event uptime can coexist with a separate documented incident.

The material conflict state must be:

```text
C-010 = RESOLVED_IN_PACKET
```

## 5. Required finding 3 — qualification orthogonality

The model must produce one bounded descriptive finding that Adyen reports selected positive commercial outcomes.

Required directional map:

```text
support_state = SUPPORTED
support = E-040, E-043
counterevidence = []
```

Both evidence items require qualifications:

```text
E-040 -> selected enterprise wins do not quantify revenue contribution
E-043 -> issuer-hosted selected case is not representative
```

The limitations must remain qualifications and must not invert evidence direction.

## 6. Material conflicts

The output must include:

```text
C-005 = UNRESOLVED_IN_PACKET
C-010 = RESOLVED_IN_PACKET
```

## 7. Deterministic targeted validator

The targeted probe adds a calibration-only deterministic assertion layer after the normal v1.0 semantic validator.

It verifies exact evidence-role placement required by this probe.

This does not claim that deterministic code can generally infer semantic alignment from arbitrary prose.

## 8. Interpretation boundary

A successful targeted probe means only:

```text
the selected physical model can follow the v1.0 role-alignment contract
when the regression scenario is explicitly exercised
```

It does **not** mean:

```text
Gate 18 PASS
model winner selected
routing frozen
normal-priority behavior proven
repeatability proven
production authorized
```

## 9. Execution sequence

```text
implementation + deterministic tests
-> no-cost dry-run
-> inspect prompt/schema hashes + cost ceiling
-> fresh budget snapshot
-> separate explicit authorization if a paid probe is justified
-> exactly one model / one call / zero retry
```

No paid execution is authorized by this document.
