# OroTitan Repository Agent Instructions

These instructions govern engineering/execution behavior in this repository. They do **not** modify OroTitan analytical methodology, frozen business contracts, scoring, economic conventions, gates, or investment policy.

## Authority hierarchy

Always apply instructions in this order:

1. System / platform instructions
2. Existing higher-authority OroTitan project contracts
3. This repository execution-control protocol
4. Task-specific user mission
5. Implementation actions

If a real conflict cannot be resolved without guessing:

`FAIL CLOSED -> STOP -> EXPLAIN -> WAIT FOR USER`

## Mandatory protocol read

At the start of **every repository mission**, including read-only VERIFY / CHECK / AUDIT / VALIDATE / INSPECT / REVIEW / CONFIRM missions, read:

`.agent/EXECUTION_CONTROL_PROTOCOL.md`

Read it before any substantive repository action. The detailed protocol is mandatory, not optional. The critical invariants below remain binding even if that file cannot be read.

## Critical execution invariants

### 1. Mission Lock before modification

For every mission, establish before editing:

- `MISSION_ID`
- `MISSION_OBJECTIVE`
- `AUTHORIZED_ACTIONS[]`
- `FORBIDDEN_ACTIONS[]`
- `ACCEPTANCE_CRITERIA[]`
- `DONE_CONDITION`
- `CURRENT_BRANCH`
- `TARGET_PR` when applicable

After execution starts, do not implicitly broaden `MISSION_OBJECTIVE`, `AUTHORIZED_ACTIONS`, `FORBIDDEN_ACTIONS`, or `DONE_CONDITION`.

**A discovered problem is not authorization for a new mission.**

Non-blocking out-of-scope discoveries go to `FOLLOW_UP_ITEMS[]` only.
Blocking out-of-scope discoveries set `MISSION_STATUS = BLOCKED`, then stop and wait for user GO.

### 2. Mandatory preflight

Before modification:

1. identify repository;
2. identify current branch;
3. verify git/worktree status when available;
4. identify the mission's existing PR if any;
5. read the files actually involved;
6. determine current state;
7. determine target state;
8. identify the minimum necessary change;
9. identify only relevant validations;
10. ensure the mission does not duplicate work already in progress.

Operationally establish:

- `CURRENT_STATE`
- `TARGET_STATE`
- `MINIMUM_CHANGE`
- `VALIDATION_PLAN`

### 3. One mission = one branch = one PR by default

Defaults:

- `MAX_ACTIVE_BRANCHES_PER_MISSION = 1`
- `MAX_NEW_PRS_PER_MISSION = 1`

If the mission already has a branch/PR, continue there.

Without explicit user GO, do not:

- create a replacement/second PR;
- abandon a branch to bypass a problem;
- merge;
- force-push;
- perform destructive/shared rebase;
- delete a branch.

CI/review failure is not a reason to create another branch or PR.

### 4. Minimal Change Rule

Every change must be directly necessary to satisfy the current mission acceptance criteria.

Do not perform opportunistic refactors, cleanup, unrelated fixes, formatting sweeps, dependency changes, redesigns, optional hardening, optimization, or production/infrastructure changes outside scope.

Before each change ask:

> Is this change necessary to satisfy the current mission's acceptance criteria?

If no: do not implement; record only as follow-up.

### 5. Controlled execution loop

The only autonomous correction loop is:

`INSPECT -> FORM HYPOTHESIS -> ONE COHERENT CHANGE -> TEST -> EVALUATE`

Each correction must have:

- `HYPOTHESIS`
- `ROOT_CAUSE`
- `CHANGE`
- `VALIDATION`
- `RESULT`

Do not recursively turn each new observation into another implementation task.

### 6. Fix-attempt limit

`MAX_AUTONOMOUS_FIX_ATTEMPTS_PER_ROOT_CAUSE = 2`

After two targeted failed fixes for the same root cause:

`MISSION_STATUS = BLOCKED`

Report the root cause, attempt 1, attempt 2, current evidence, recommended next step, then stop and wait for user GO.

### 7. CI failures are information, not new missions

Classify first:

- `CAUSED_BY_CURRENT_CHANGE` -> scoped correction allowed.
- `PRE_EXISTING_FAILURE` -> report; do not repair unless directly required.
- `FLAKY_OR_INFRASTRUCTURE` -> at most one justified retry; do not alter product code to satisfy flaky infrastructure.
- `UNKNOWN` -> bounded investigation; no speculative correction.

### 8. Verification-only default is read-only

For missions primarily asking to VERIFY / CHECK / AUDIT / VALIDATE / INSPECT / REVIEW / CONFIRM:

`DEFAULT_PERMISSION = READ_ONLY`

Without specific GO, do not modify code/config/infrastructure, create branch/commit/PR, or fix discovered failures.

**A verification failure is a result, not authorization to fix it.**

### 9. Review comments

Classify every review comment:

- `BLOCKING`: actually prevents mission acceptance or proves implementation incorrect.
- `NON_BLOCKING`: suggestion, style, cleanup, optional hardening, refactor, future work.

Only BLOCKING comments may authorize an autonomous scoped correction **when the current Mission Lock already authorizes implementation/modification work**. In a verification-only mission, a BLOCKING finding remains a read-only result and requires explicit user GO before any fix. NON_BLOCKING comments become follow-up only.

### 10. Async polling limit

`MAX_ASYNC_POLL_CYCLES = 2`

Applies to CI, GitHub Actions, review bots, Vercel, deployments, and external services.

After two checks without material change:

`MISSION_STATUS = PENDING_EXTERNAL_RESULT`

`PENDING_EXTERNAL_RESULT = YES`

Stop polling and return control to the user. This is neither COMPLETE nor BLOCKED.

### 11. Action budget / checkpoint

`MAX_SIGNIFICANT_ACTIONS_BEFORE_CHECKPOINT = 12`

Verification-only mode:

- `MAX_SIGNIFICANT_ACTIONS = 8`
- `MAX_CODE_CHANGES = 0`
- `MAX_COMMITS = 0`
- `MAX_NEW_BRANCHES = 0`
- `MAX_NEW_PRS = 0`
- `MAX_POLL_CYCLES = 2`

At checkpoint verify mission, scope, branch, PR, root cause, convergence, and whether DONE is already met. If any critical invariant changed, stop and report.

### 12. Anti-recursion / stop conditions

Never automatically transform a discovered architecture, monitoring, retry, contract, or other secondary issue into a new mission.

Stop immediately if:

- two targeted fixes fail for the same root cause;
- material scope expansion is required;
- a second PR would be required;
- branch/PR identity is ambiguous;
- destructive action is required;
- higher authority conflicts;
- test result is non-interpretable;
- a critical assumption would require guessing;
- repository state materially differs from expectation;
- acceptance criteria become ambiguous;
- a verification task would require modification;
- DONE is already reached.

### 13. Definition of Done

The mission is complete as soon as the **original acceptance criteria** are satisfied and, when applicable:

- relevant tests pass;
- required CI passes;
- blocking review comments = 0.

Then:

`MISSION_STATUS = COMPLETE -> STOP IMMEDIATELY`

Do not continue for extra reassurance, optional review, cleanup, redesign, optimization, future work, or the next mission.

**DONE = STOP.**

### 14. Explicit user GO required

A new user GO is required before:

- a new mission;
- scope expansion;
- a second PR;
- a new unplanned branch;
- merge;
- force-push;
- destructive/shared rebase;
- branch deletion;
- architecture redesign;
- out-of-scope DB migration;
- production/infrastructure change;
- fixing a problem discovered during verification;
- implementing a follow-up item.

Authorization for mission A never authorizes mission B.

### 15. OroTitan execution principles

Preserve:

`PERSISTED STATE > conversational reconstruction`

`FAIL CLOSED > guessing`

`EXACT IDENTITY > filename / branch / PR guessing`

`EXPLICIT TRANSITIONS > implicit continuation`

This execution-control layer must never create analytical scores, business states, investment gates, or methodology changes.

## Standard end-of-mission report

Use the exact compact structure defined in `.agent/EXECUTION_CONTROL_PROTOCOL.md`, then stop.
