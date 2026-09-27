# OROTITAN CONTINUITY PROTOCOL V1

Status: ACTIVE
Scope: VNExT engineering continuity
Methodology change: NO
Analytical authority: NONE
Production mutation authority: NONE

## 1. Purpose

A ChatGPT conversation is a work surface, not the source of truth for VNExT state.

The continuity system exists so that a conversation may end, be truncated, or be replaced without losing:

- current objective;
- completed steps;
- active work;
- blockers;
- open issues;
- frozen or preserved decisions;
- current authorizations and prohibitions;
- authoritative artifact references;
- exact next action;
- resume instructions.

## 2. Continuity files

```text
docs/continuity/
├── OROTITAN_CONTINUITY_PROTOCOL_V1.md
├── OROTITAN_CONTINUITY_STANDING_AUTH_V1.md
├── CURRENT_STATE.json
├── HANDOFF_LATEST.md
├── DECISION_LOG.md
├── OPEN_ISSUES.md
└── sessions/
    └── <dated checkpoint>.md
```

## 3. Authority boundary

This continuity layer does not replace or supersede:

- frozen OroTitan methodology;
- Gate 18 contracts or calibration artifacts;
- persisted run / registry state;
- Git history;
- private source artifacts;
- production state.

If continuity text conflicts with an authoritative persisted artifact, contract, or repository fact, the higher-authority persisted source wins and continuity must be reconciled before work continues.

Chat memory is never authority.

## 4. Checkpoint triggers

Create or update a continuity checkpoint after any material event:

- PR merged;
- Gate or phase status changed;
- model run completed;
- forensic completed;
- candidate admitted or rejected;
- blocker discovered or cleared;
- architecture decision made;
- authorization changed;
- NEXT_ACTION changed;
- material test result changed execution direction.

Do not wait for a conversation to approach its context limit.

## 5. Required CURRENT_STATE fields

`CURRENT_STATE.json` must contain at minimum:

- protocol_version;
- resume_id;
- project;
- repository;
- branch;
- last_verified_head_sha;
- last_verified_at;
- current_gate;
- current_phase;
- status;
- objective;
- completed;
- active_work;
- blockers;
- open_issues;
- preserved_decisions;
- authorized_actions;
- forbidden_actions;
- authoritative_artifacts;
- next_action;
- resume_sequence.

The file is operational continuity metadata only.

## 6. HEAD reconciliation rule

A checkpoint cannot contain its own final Git commit SHA without creating a self-reference problem.

Therefore:

```text
last_verified_head_sha
=
repository HEAD verified immediately before the checkpoint update
```

On resume:

1. fetch the current branch HEAD;
2. compare it with `last_verified_head_sha`;
3. if equal, continue normal bootstrap;
4. if different, inspect commits from `last_verified_head_sha..HEAD`;
5. reconcile continuity files against those commits and authoritative artifacts;
6. only then execute `next_action`.

A HEAD mismatch is not automatically an error. It is a reconciliation trigger.

## 7. Resume bootstrap

When the user says, for example:

```text
Reprends OroTitan VNExT.
```

the assistant should:

1. resolve repository and branch;
2. read `CURRENT_STATE.json`;
3. read `HANDOFF_LATEST.md`;
4. read relevant recent entries in `DECISION_LOG.md`;
5. read `OPEN_ISSUES.md`;
6. verify current Git HEAD;
7. reconcile commits since `last_verified_head_sha`;
8. verify referenced authoritative artifacts where material;
9. preserve all explicit prohibitions;
10. resume from `next_action`.

Do not reconstruct material state from conversational memory when persisted state is available.

## 8. Fail-closed conditions

Stop and reconcile before further execution if:

- referenced artifact is missing;
- branch cannot be resolved;
- a material authorization is ambiguous;
- continuity contradicts a frozen contract;
- continuity contradicts persisted Gate state;
- NEXT_ACTION depends on a result that was never persisted;
- a historical failure would need to be silently rewritten.

## 9. Privacy boundary

Do not copy private evidence bodies, licensed research, credentials, secrets, or sensitive user material into this public continuity layer.

Only record identifiers, statuses, non-sensitive execution metadata, and references needed to resume.

## 10. Session checkpoints

A dated session checkpoint should record:

- starting state;
- work executed;
- material observations;
- decisions;
- files / PRs / commits created;
- tests;
- resulting state;
- next action.

Session checkpoints are historical aids. `CURRENT_STATE.json` remains the current machine-readable continuity view.

## 11. No analytical authority

This protocol may record an analytical decision already made by an authorized process.

It may not create, alter, score, reinterpret, or override an analytical decision merely for continuity convenience.
