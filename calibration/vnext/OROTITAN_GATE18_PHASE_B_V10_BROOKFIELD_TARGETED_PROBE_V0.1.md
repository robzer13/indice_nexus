# OROTITAN_GATE18_PHASE_B_V10_BROOKFIELD_TARGETED_PROBE_V0.1

**Gate:** 18  
**Phase:** B company calibration  
**Status:** CALIBRATION-ONLY TARGETED REGRESSION PROBE — NOT FROZEN  
**Production mutation:** FORBIDDEN  
**Publication authority:** DISABLED  
**Model winner authority:** NONE  
**Routing authority:** NONE

## 0. Purpose

The Brookfield Corporation × LUNA v1.0 normal-path canary is human-quality admissible and shows no evidence-role inversion.

However, the two evidence items that caused the historical v0.5 failure, `E-039` and `E-042`, were not selected by the v1.0 normal run.

This probe exists only to test those exact historical evidence-role assignments under the v1.0 semantic contract.

It is not a normal priority-selection observation and is not comparison-admissible model-ranking evidence.

## 1. Preserved contract

```text
source company = Brookfield Corporation
SOURCE_RUN_ID = b765650c-477a-4d34-9143-55db296f6000
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

Only the module question changes to force the historical regression paths.

## 2. Probe identity

```text
probe_id = BROOKFIELD_PEER_ROLE_CORE_001
prompt_template_id = GATE18_BROOKFIELD_PEER_ROLE_PROBE_V0_1
prompt_template_version = 0.1
```

## 3. Required finding 1 — peer-platform replicability

The model must state one bounded descriptive proposition that large-scale alternative-asset-management capability is present across Brookfield's major peers.

Required directional map:

```text
support_state = SUPPORTED
support = E-036, E-037, E-039
counterevidence = []
conflicts = []
```

`E-039` documents KKR insurance / alternative-platform scale.

Its peer-accounting limitation may be expressed as a qualification.

It must not be converted into counterevidence to the exact proposition that major peers possess comparable large-scale platform capability.

## 4. Required finding 2 — competing capital in AI infrastructure

The model must state one bounded descriptive proposition that Brookfield faces competing capital from major peer platforms in large AI-infrastructure financing initiatives.

Required directional map:

```text
support_state = SUPPORTED
support = E-042
counterevidence = []
conflicts = []
```

`E-042` documents Brookfield participating alongside Apollo, BlackRock, Blackstone, Goldman Sachs and KKR in a large AI-infrastructure financing initiative.

For the exact proposition about competing capital, this observation is SUPPORT.

## 5. Why these two items matter

The historical v0.5 Brookfield canary incorrectly placed `E-039` and `E-042` in `counterevidence_ids` against a peer-replicability proposition.

That failure was a semantic direction error:

```text
peer replicability evidence
was mislabeled as
evidence against peer replicability
```

The v1.0 normal Brookfield canary no longer showed a visible role inversion, but it did not select E-039/E-042.

Therefore the exact historical regression remains untested until this probe is exercised.

## 6. Deterministic targeted validator

The targeted validator runs after the normal v1.0 semantic validator and checks:

- exactly two priority findings;
- E-036/E-037/E-039 as SUPPORT for finding 1;
- no counterevidence for finding 1;
- E-042 as SUPPORT for finding 2;
- no counterevidence for finding 2;
- no conflict assignment to either finding;
- no counterevidence link when no counterevidence exists.

This assertion layer is calibration-specific.

It does not claim that deterministic code can infer arbitrary natural-language evidence polarity.

## 7. Interpretation boundary

A successful probe means only:

```text
LUNA under the v1.0 contract can correctly classify the exact
Brookfield E-039 / E-042 historical regression evidence
when those items are explicitly exercised
```

It does not mean:

- model winner selected;
- routing frozen;
- repeatability proven;
- Phase C complete;
- production authorized;
- publication authorized.

## 8. Execution sequence

```text
implementation + deterministic tests
-> no-cost dry-run
-> inspect prompt/schema hashes + cost ceiling
-> fresh budget snapshot
-> separate explicit single-call authorization if justified
-> paid probe
-> human adjudication
-> Phase B closure decision
```

No paid execution is authorized by this document.
