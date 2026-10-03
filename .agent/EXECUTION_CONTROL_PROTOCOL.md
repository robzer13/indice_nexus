# OroTitan Execution Control Protocol

Status: repository engineering / execution policy.

This protocol governs development-agent execution in this repository. It is **not** an analytical-methodology document and must not alter OroTitan frozen business contracts, analytical formulas, economic conventions, scoring, gates, state semantics, or investment policy.

The root `AGENTS.md` contains the critical invariants and requires this file to be read before any repository modification.

---

## A. Objective

Install and enforce a permanent execution-control circuit breaker for OroTitan engineering work.

The protocol exists to prevent:

- mission-scope drift;
- recursive creation of sub-missions;
- correction loops;
- repetitive verification;
- excessive CI/review/deployment polling;
- work not requested by the user;
- ambiguous completion;
- opportunistic architecture changes after the requested outcome is already achieved.

It applies to future repository tasks and discussions that operate on this repository.

---

## B. Persistence

The policy is repository-versioned and must not depend on conversational memory.

Persistent instruction locations:

- root `AGENTS.md`: critical invariants;
- `.agent/EXECUTION_CONTROL_PROTOCOL.md`: detailed protocol.

Do not move the essential safeguards into temporary files, chat-only notes, or optional documentation.

---

## C. Authority hierarchy

Apply this order:

1. SYSTEM / PLATFORM INSTRUCTIONS
2. EXISTING HIGHER-AUTHORITY PROJECT CONTRACTS
3. OROTITAN EXECUTION CONTROL PROTOCOL
4. TASK-SPECIFIC USER MISSION
5. IMPLEMENTATION ACTIONS

A task-specific mission may define the work to perform, but it does not implicitly remove execution safeguards.

If a real conflict remains:

`FAIL CLOSED -> STOP -> EXPLAIN -> WAIT FOR USER`

---

## D. Mandatory Mission Lock

At the beginning of every mission, before modification, establish:

```text
MISSION_ID =
MISSION_OBJECTIVE =
AUTHORIZED_ACTIONS[] =
FORBIDDEN_ACTIONS[] =
ACCEPTANCE_CRITERIA[] =
DONE_CONDITION =
CURRENT_BRANCH =
TARGET_PR =
```

`TARGET_PR` may be `NONE` when not applicable.

Once execution begins, do not implicitly broaden:

- `MISSION_OBJECTIVE`;
- `AUTHORIZED_ACTIONS[]`;
- `FORBIDDEN_ACTIONS[]`;
- `DONE_CONDITION`.

Absolute rule:

`A DISCOVERED PROBLEM != AUTHORIZATION FOR A NEW MISSION`

Out-of-scope discovery handling:

### Non-blocking

```text
FOLLOW_UP_ITEMS += discovered issue
DO NOT investigate deeply
DO NOT fix
CONTINUE only the original mission
```

### Blocking

```text
MISSION_STATUS = BLOCKED
BLOCKER = precise description
STOP
WAIT FOR USER GO
```

---

## E. Mandatory preflight

Before any modification:

1. identify the repository;
2. identify the current branch;
3. verify git/worktree status when the execution environment exposes it;
4. identify the existing PR corresponding to the mission;
5. read the files actually involved;
6. determine current state;
7. determine target state;
8. determine the smallest necessary change;
9. identify only the tests/validations relevant to the mission;
10. verify the mission does not duplicate work already in progress.

Establish operationally:

```text
CURRENT_STATE =
TARGET_STATE =
MINIMUM_CHANGE =
VALIDATION_PLAN =
```

If the environment cannot expose a local worktree status, state that limitation and use the authoritative remote branch/ref state available. Do not invent a clean git status.

Only then begin implementation.

---

## F. One mission = one branch = one PR

Default limits:

```text
MAX_ACTIVE_BRANCHES_PER_MISSION = 1
MAX_NEW_PRS_PER_MISSION = 1
```

If a branch/PR already corresponds to the mission, continue there.

A failed CI job, review comment, or correction does **not** justify another PR.

Without explicit user GO, do not:

- create a replacement PR;
- create a second PR for the same mission;
- close/recreate a PR to bypass a problem;
- abandon a branch to bypass a problem;
- merge;
- force-push;
- destructive/shared rebase;
- delete a branch.

---

## G. Minimal Change Rule

Every modification must be directly necessary for the current acceptance criteria.

Forbidden by default:

- opportunistic refactor;
- cleanup not required;
- architecture redesign;
- unnecessary dependency change;
- unrelated bug fix;
- large formatting-only change;
- unrequested optimization;
- optional hardening;
- production or infrastructure change outside mission scope.

Before each change, ask:

> Is this change necessary to satisfy the acceptance criteria of the current mission?

If the answer is no:

```text
DO NOT IMPLEMENT
FOLLOW_UP_ITEMS += item
```

---

## H. Controlled execution loop

The only autonomous implementation loop is:

`INSPECT -> FORM HYPOTHESIS -> ONE COHERENT CHANGE -> TEST -> EVALUATE`

Every corrective cycle must be represented as:

```text
HYPOTHESIS =
ROOT_CAUSE =
CHANGE =
VALIDATION =
RESULT =
```

Forbidden pattern:

`FAIL -> random change -> new problem -> unrelated change -> new idea -> new architecture -> continue`

A new observation must first be classified against the current Mission Lock.

---

## I. Fix-attempt limit

```text
MAX_AUTONOMOUS_FIX_ATTEMPTS_PER_ROOT_CAUSE = 2
```

After two targeted unsuccessful corrections for the same root cause:

```text
MISSION_STATUS = BLOCKED
ROOT_CAUSE =
ATTEMPT_1 =
ATTEMPT_2 =
CURRENT_EVIDENCE =
RECOMMENDED_NEXT_STEP =
```

Then stop and wait for user GO.

A third speculative correction is prohibited.

---

## J. CI rule

A failed CI/test result is information, not a new mission.

Classify first:

### A. CAUSED_BY_CURRENT_CHANGE

A scoped fix is allowed, subject to the Mission Lock and fix-attempt budget.

### B. PRE_EXISTING_FAILURE

Report it. Do not repair unless direct mission completion requires it and the Mission Lock allows that work.

### C. FLAKY_OR_INFRASTRUCTURE

At most one retry when justified. Do not change product code to satisfy flaky infrastructure.

### D. UNKNOWN

Perform bounded investigation only. No speculative fix.

Rule:

`A FAILED TEST IS INFORMATION, NOT A NEW MISSION.`

---

## K. Verification-only mode

When a mission is primarily:

- VERIFY
- CHECK
- AUDIT
- VALIDATE
- INSPECT
- REVIEW
- CONFIRM

then:

```text
DEFAULT_PERMISSION = READ_ONLY
```

Allowed:

- read;
- inspect;
- run existing validations;
- inspect logs;
- inspect CI;
- inspect Vercel;
- inspect GitHub;
- test existing behavior;
- return diagnosis.

Without specific GO, prohibited:

- modify code;
- fix;
- implement;
- refactor;
- create branch;
- commit;
- push;
- create PR;
- change configuration;
- change infrastructure;
- redesign architecture.

Absolute rule:

`A VERIFICATION FAILURE IS A RESULT, NOT AUTHORIZATION TO FIX IT.`

Example:

`GO VERIFY PRODUCTION`

does not imply:

`GO FIX PRODUCTION`

and does not imply:

`GO IMPLEMENT ANTI-LOOP`.

---

## L. Review rule

Classify every review comment:

### BLOCKING

A comment is blocking only if it:

- prevents the current mission acceptance criteria;
- demonstrates a required condition fails;
- proves the current implementation is incorrect.

A blocking finding permits only a scoped correction within the current mission and correction budget.

### NON_BLOCKING

Examples:

- suggestion;
- style;
- cleanup;
- improvement;
- optional hardening;
- refactor;
- future work.

Handling:

```text
FOLLOW_UP_ITEMS += comment
NO CHANGE
```

A new review is not automatically required after every correction. Run only the validations required by the mission or repository gates.

---

## M. Async polling / waiting

```text
MAX_ASYNC_POLL_CYCLES = 2
```

This applies to:

- GitHub Actions;
- CI;
- Codex Review;
- Vercel;
- review bots;
- deployment;
- other external services.

After two checks without material change:

```text
PENDING_EXTERNAL_RESULT = YES
STOP POLLING
RETURN CONTROL TO USER
```

Do not repeatedly wait and recheck indefinitely.

A materially changed result resets interpretation, not the mission scope.

---

## N. Action budget / checkpoint

```text
MAX_SIGNIFICANT_ACTIONS_BEFORE_CHECKPOINT = 12
```

Verification-only mode:

```text
MAX_SIGNIFICANT_ACTIONS = 8
MAX_CODE_CHANGES = 0
MAX_COMMITS = 0
MAX_NEW_BRANCHES = 0
MAX_NEW_PRS = 0
MAX_POLL_CYCLES = 2
```

At checkpoint verify:

```text
MISSION_UNCHANGED?
SCOPE_UNCHANGED?
SAME_BRANCH?
SAME_PR?
ROOT_CAUSE_UNCHANGED?
WORK_CONVERGING?
DONE_CONDITION_NOT_ALREADY_MET?
```

If a critical answer is NO:

`STOP -> STATUS REPORT -> WAIT FOR USER`

---

## O. Anti-recursion

Never automatically convert a discovery into a new mission.

Forbidden examples:

- mission -> architecture discovery -> architecture redesign;
- mission -> monitoring issue -> monitoring system;
- mission -> retry issue -> new retry engine;
- mission -> contract issue -> new contract version;
- mission -> unrelated production issue -> production correction.

Secondary subjects belong in:

`FOLLOW_UP_ITEMS[]`

and require a separate user GO.

---

## P. Stop conditions

Stop immediately if:

- two targeted corrections fail for the same root cause;
- solution requires material scope expansion;
- a second PR would be necessary;
- branch/PR identity is ambiguous;
- destructive action becomes necessary;
- a higher-authority contract conflicts;
- a test result cannot be interpreted reliably;
- a critical hypothesis requires guessing;
- repository state materially differs from expected state;
- acceptance criteria become ambiguous;
- a verification-only task would require modification;
- DONE is already reached.

Do not continue to gather optional certainty after a stop condition.

---

## Q. Definition of Done

The mission is complete when:

`ORIGINAL_ACCEPTANCE_CRITERIA = SATISFIED`

and, when applicable:

- `RELEVANT_TESTS = PASS`;
- `REQUIRED_CI = PASS`;
- `BLOCKING_REVIEW_COMMENTS = 0`.

Then:

`MISSION_STATUS = COMPLETE -> STOP IMMEDIATELY`

Do not continue in order to:

- be more certain;
- repeat a successful validation;
- request optional review;
- clean code;
- improve architecture;
- execute follow-ups;
- optimize;
- anticipate the next mission.

Rule:

`DONE = STOP`

not:

`DONE = FIND MORE WORK`.

---

## R. Explicit authorization

A new user GO is required before:

- a new mission;
- material scope expansion;
- a second PR;
- an unplanned new branch;
- merge;
- force-push;
- destructive/shared rebase;
- branch deletion;
- architecture redesign;
- DB migration outside the current mission;
- production change;
- infrastructure change;
- correction discovered by a VERIFY-only mission;
- implementation of a follow-up.

Authorization for mission A never authorizes mission B.

---

## S. Standard end-of-mission report

Return a short report:

```text
MISSION_STATUS = COMPLETE | BLOCKED

MISSION =
BRANCH =
PR =
LATEST_COMMIT =

FILES_CHANGED =
TESTS =
CI_STATE =
BLOCKING_REVIEWS =

ACCEPTANCE_CRITERIA =
DONE_CONDITION_MET = YES | NO

BLOCKERS =
FOLLOW_UP_ITEMS =

SCOPE_DRIFT = YES | NO
```

Then stop.

Do not append a proposed next implementation unless the user explicitly asks.

---

## T. OroTitan alignment

Always preserve:

`PERSISTED STATE > conversational reconstruction`

`FAIL CLOSED > guessing`

`EXACT IDENTITY > filename / branch / PR guessing`

`EXPLICIT TRANSITIONS > implicit continuation`

An anomaly must never be hidden by inventing a new convention.

This protocol must create:

- no analytical score;
- no business/analytical state;
- no investment gate;
- no economic convention;
- no methodology revision.

It is an engineering/execution-control layer only.
