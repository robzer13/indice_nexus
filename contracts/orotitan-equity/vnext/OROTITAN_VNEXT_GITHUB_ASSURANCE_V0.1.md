# OROTITAN_VNEXT_GITHUB_ASSURANCE_V0.1

**Project:** OroTitan Equity Research  
**Status:** CANDIDATE - ENFORCEMENT PENDING  
**Methodology change:** NO  
**Depends on:** Gates 6 through 11  
**Runtime dependency:** NONE  
**Production mutation:** NONE  
**Shadow DB mutation:** NONE

## 0. Purpose

Gate 12 makes deterministic assurance a merge-time property rather than a developer convention.

The target invariant is:

```text
NO MERGE INTO vnext
UNLESS
verify-vnext = PASS
```

GitHub Actions remains a second line of defense. Runtime safeguards remain authoritative at execution time; CI prevents code that weakens those safeguards from entering the protected VNext branch.

## 1. Existing deterministic CI

The VNext workflow is:

```text
.github/workflows/vnext-ci.yml
```

It executes for:

```text
push -> vnext
pull_request -> vnext
```

The required job identity is:

```text
verify-vnext
```

The job currently executes:

```text
dependency install
baseline-manifest JSON validation
lint with zero warnings
TypeScript typecheck
unit / contract / deterministic assurance tests
PostgreSQL migration tests
Next.js production build
```

There are no VNext path filters that can silently skip the job for a pull request targeting `vnext`.

## 2. Assurance coverage

Gate 12 explicitly requires the following assurance categories to remain wired into `verify-vnext`:

```text
unit tests
contract tests
golden fixtures
JSON Schema validation
SHA-256 exact-byte validation
lineage invariants
publication invariants
state-machine invariants
environment isolation
pre-stage admission
post-stage certification
recovery classification
PostgreSQL migration non-regression
production build
```

Relevant deterministic tests include:

```text
tests/vnext-state-machine.test.ts
tests/vnext-state-model.test.ts
tests/vnext-state-model-schema.test.ts
tests/vnext-module-contract.test.ts
tests/vnext-environment-boundary.test.ts
tests/vnext-run-controller.test.ts
tests/vnext-pre-stage-preflight.test.ts
tests/vnext-post-stage-certification.test.ts
tests/vnext-recovery-engine.test.ts
tests/vnext-gate12-assurance.test.ts
```

## 3. Golden fixture

Gate 12 introduces:

```text
tests/vnext/fixtures/gate12-golden-v0.1.json
```

The fixture pins deterministic VNext semantics that must not drift silently:

```text
run-state vocabulary
stage lifecycle vocabulary
handoff-gate mapping
recovery-class vocabulary
shadow project identity
production project identity as forbidden target
publication-disabled expectation
exact UTF-8 hash fixture
required assurance-file inventory
```

Its exact-byte SHA-256 fixture is computed by the test suite, rather than accepted as metadata.

Golden drift requires an explicit versioned fixture change.

## 4. Assurance self-check

```text
tests/vnext-gate12-assurance.test.ts
```

checks that:

```text
golden runtime vocabularies still match code
golden SHA-256 bytes still hash exactly
shadow environment identity remains pinned
production Supabase remains rejected
required deterministic test/schema files still exist
VNext CI still triggers on pull requests to vnext
the verify-vnext job still exists
lint/typecheck/tests/PostgreSQL/build remain wired
package scripts still execute the required assurance layers
the desired GitHub ruleset cannot silently drop merge protections
```

This is defense against accidental CI erosion.

## 5. Required GitHub ruleset

Repository rules are not represented by repository source alone. They are a GitHub control-plane setting.

The frozen desired state is versioned at:

```text
contracts/orotitan-equity/vnext/VNEXT_GITHUB_RULESET_DESIRED_STATE_V0.1.json
```

Required branch target:

```text
refs/heads/vnext
```

Required enforcement:

```text
ACTIVE
```

Required protections:

```text
Require pull request before merging        = ON
Required approving review count            = 0
Require status checks before merging       = ON
Required status check                      = verify-vnext
Require branches to be up to date          = ON
Block force pushes                         = ON
Block branch deletion                      = ON
Bypass actors                              = NONE
```

A review count of zero is intentional for the current single-maintainer development topology. The control objective is deterministic assurance, not artificial self-approval.

## 6. Why strict required checks

GitHub's strict required-check mode requires the pull-request branch to be up to date with the protected base before merging.

For VNext this prevents:

```text
PR A passes against old vnext
PR B changes a shared invariant
PR A merges without retesting against the new base
```

The extra CI run is accepted as the safer default.

## 7. Post-enforcement development workflow

Once the ruleset is ACTIVE, direct development commits to `vnext` stop.

All subsequent VNext work uses:

```text
vnext
  -> create gate-specific working branch
  -> commit implementation/tests/docs there
  -> open PR targeting vnext
  -> verify-vnext must PASS
  -> merge PR
```

This applies to both human and connector-generated changes.

No bypass is part of the normal OroTitan development process.

## 8. Tooling boundary

GitHub Copilot Student and Codespaces may assist development.

They are not runtime authorities and are not dependencies of the VNext execution contract.

A missing Copilot/Codespaces service cannot change or disable the deterministic CI requirements.

## 9. Gate 12 acceptance matrix

```text
G12-01 VNext CI runs on pull_request -> vnext        PASS
G12-02 stable required job name verify-vnext          PASS
G12-03 lint enforced                                   PASS
G12-04 typecheck enforced                              PASS
G12-05 unit/contract tests enforced                    PASS
G12-06 PostgreSQL tests enforced                       PASS
G12-07 production build enforced                       PASS
G12-08 golden fixture present                          PASS
G12-09 exact-byte SHA-256 golden test                  PASS
G12-10 JSON Schema tests present                       PASS
G12-11 lineage invariants present                      PASS
G12-12 publication/environment invariants present      PASS
G12-13 state-machine invariants present                PASS
G12-14 assurance self-check prevents silent CI erosion PASS
G12-15 ruleset desired state versioned                 PASS
G12-16 ruleset ACTIVE on vnext                         PENDING
G12-17 pull request required                           PENDING
G12-18 verify-vnext required before merge              PENDING
G12-19 up-to-date branch required                      PENDING
G12-20 force-push/deletion protection                  PENDING
G12-21 bypass list empty                               PENDING
```

## 10. Gate condition

Gate 12 remains `PENDING` until the live GitHub ruleset is applied and independently reread from GitHub.

The repository-side assurance implementation may be green before then, but that alone does not satisfy:

```text
NO MERGE VNext IF DETERMINISTIC ASSURANCE FAILS
```

## 11. Out of scope

Gate 12 does not:

- modify analytical methodology;
- bind an AI provider;
- mutate Supabase;
- change production publication authority;
- replace runtime deterministic checks.
