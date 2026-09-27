# OROTITAN CONTINUITY STANDING AUTHORIZATION V1

Status: ACTIVE
User authorization date: 2026-09-27
Scope: continuity maintenance only

The user authorized a standing continuity mechanism so that context, steps, objectives, decisions, blockers, and next actions are not lost when a discussion reaches its limit.

## Authorized without repeated user confirmation

- update `docs/continuity/CURRENT_STATE.json`;
- update `docs/continuity/HANDOFF_LATEST.md`;
- append material decisions to `docs/continuity/DECISION_LOG.md`;
- update `docs/continuity/OPEN_ISSUES.md`;
- create dated session checkpoint files;
- record Git HEAD, PRs, commits, tests, blockers, authorities, and NEXT_ACTION;
- reconcile continuity metadata after repository changes;
- create continuity-only PRs when needed.

## Not authorized by this standing authorization

- analytical methodology changes;
- frozen contract changes;
- new model inference;
- model downloads;
- paid execution;
- model switching;
- second-cell or matrix expansion;
- production mutation;
- publication;
- secrets or private evidence publication;
- retroactive pass;
- mutation of historical model outputs;
- mutation of authoritative analytical results.

Any action outside the authorized continuity scope still requires its normal OroTitan authority.


## Execution authorization override — 2026-09-27

The user subsequently established `OROTITAN-STANDING-TECHNICAL-AUTH-002`.

For OroTitan technical execution, zero-cost actions no longer require repeated user confirmation. This includes local downloads, load/memory preflights, local inference, bounded retries, model/candidate changes, testing, GitHub PR/CI/merge operations, and other zero-cost protocol steps.

A new user authorization is required only when the next action would incur a nonzero external monetary cost.

This execution authorization does not weaken the protocol guards on historical immutability, private evidence handling, fail-closed identity/contract checks, CI-before-merge, or versioned methodology changes.
