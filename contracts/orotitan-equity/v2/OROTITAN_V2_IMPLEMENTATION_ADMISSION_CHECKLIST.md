# OROTITAN V2 — IMPLEMENTATION & PRODUCTION ADMISSION CHECKLIST

**Status:** FROZEN DESIGN GATE  
**Purpose:** distinguish a frozen V2 design from an executable V2 production authority bundle.

Merging V2 design documents does NOT activate V2 for live runs.

## A. Frozen design authorities

Required:

```text
A01 V2 Execution Process frozen
A02 V2 Pilotage Contract frozen
A03 V2 Research Contract frozen
A04 V2 Deep Dive Contract frozen
A05 V2 Integration Contract frozen
A06 V2 Handoff Prompt Templates frozen
A07 V2 Taxonomy frozen
A08 V2 Canonical Product Extension frozen
```

## B. Authority compatibility

Before activation, prove that unchanged higher analytical authorities remain compatible:

```text
B01 Analysis Standard compatibility PASS
B02 Master Prompt compatibility PASS
B03 Investment Policy compatibility PASS
B04 Execution Patch compatibility PASS
B05 Integration Spec compatibility or explicit V2 successor frozen
B06 Canonical screener schema V2 successor frozen
B07 I2 compatibility PASS
B08 I3-B compatibility PASS
B09 Registry/Stage Manifest compatibility PASS
```

No authority may be silently mixed across versions.

## C. Runtime implementation

Required:

```text
C01 Pilotage can generate/reconstruct all V2 bootstrap prompts
C02 Research final handoff -> Fundamentals verified
C03 Fundamentals CHECKPOINT persistence verified
C04 FUNDAMENTALS_LOCK exact resolution verified
C05 Fundamentals -> Valuation handoff verified
C06 Valuation CHECKPOINT persistence verified
C07 VALUATION_LOCK exact resolution verified
C08 Valuation -> Certification handoff verified
C09 Certification FINAL Deep Dive finalization verified
C10 Certification -> Integration admission verified
C11 Blocked-resolution prompt flow verified
C12 Auto-limited reopen flow verified
C13 supersession/history preservation verified
C14 targeted REFRESH flow verified
C15 V1 runs remain executable/resolvable under V1 pins
```

## D. Canonical schema / product implementation

Required:

```text
D01 canonical snapshot schema accepts V2 classification
D02 taxonomy enums validated against OROTITAN_TAXONOMY_V2.0
D03 business_description_short validation enforced
D04 structured thesis validation enforced
D05 PEA filter state remains descriptive/non-scoring
D06 Integration mapping writes exact admitted V2 fields
D07 I3-B traceability includes new V2 fields where required
D08 frontend reads new canonical fields without becoming scoring authority
D09 frontend filters country/sector/industry/business model from canonical snapshot truth
D10 legacy V1 canonical snapshot rendering remains supported
```

## E. Regression / acceptance

Required:

```text
E01 all 45 V2 process acceptance tests PASS
E02 V1 contract/regression suite remains PASS
E03 Registry PostgreSQL suite remains PASS
E04 I2 deterministic regression PASS
E05 I3-B regression PASS
E06 Next.js lint PASS
E07 TypeScript typecheck PASS
E08 unit tests PASS
E09 production build PASS
E10 Vercel preview PASS
```

## F. Contract Pin Pack V2

A production V2 run MUST pin an immutable authority bundle.

Required:

```text
F01 Contract Pin Pack V2 exists
F02 exact immutable Git locators for V2 authorities
F03 canonical SHA-256 verified byte-for-byte
F04 contract_set_sha256 deterministic
F05 Registry pin-completeness logic supports the V2 set
F06 no existing V1 run is mutated to V2 pins
F07 new V2 run creation fails closed on missing/mismatched V2 pin
```

If Registry pin-completeness currently requires the V1 logical set, activation remains BLOCKED until a reviewed compatibility/migration design exists. Do not bypass that guard.

## G. Production pre-flight

Read-only first:

```text
G01 target Supabase project exact
G02 live Registry constraints exact
G03 no unexpected schema drift
G04 private artifact stores healthy
G05 current canonical dossier pointers preserved
G06 V1 Qualys snapshot preserved
G07 V2 code/contract commit exact and CI green
G08 V2 Contract Pin Pack resolves all authorities
G09 dry-run/bootstrap reconstruction PASS
G10 publication boundary still requires GO PUBLISH
```

## H. Activation command

Only after A–G are all PASS should Pilotage request explicit user authorization:

```text
GO ACTIVATE OROTITAN V2
```

Before that command:

```text
V2_DESIGN_STATUS = FROZEN
V2_PRODUCTION_STATUS = NOT_ACTIVE
```

After authorized activation and successful final postflight:

```text
V2_PRODUCTION_STATUS = ACTIVE_FOR_NEW_RUNS
V1_EXISTING_RUNS = GRANDFATHERED
```

## I. Freeze window after activation

Normal redesign remains closed until BOTH:

```text
>= 10 completed V2 analyses
AND
>= 6 weeks from first V2 production run
```

Only documented blocking defects may produce V2.0.x before that review gate.
