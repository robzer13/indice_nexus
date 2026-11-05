# P0 — OroTitan V3.1: recovery of blocked runs (phase 1, REVIEW ONLY)

**Status:** draft GitHub review; **no production change**, no Supabase write, no registry/manifest mutation, no new ChatGPT tool, no publication. Authoritative workflow: frozen Research Execution Process (gap-closing loop, critical input assistance), Pilotage V1.0.1, Research Stage V1.0, V3.1 pinned runtime.

## Confirmed baseline (2026-10-22, READ ONLY)

Supabase Registry `cugpgtzygqqlxetyeven` has five V3.1 runs:
- Besi `48b9426b-4d0d-4858-9d87-efcd13ac415d`: Registry `ACTIVE/RESEARCH`, stage `IN_PROGRESS`, active CHECKPOINT, contract code `BLOCKED_INSUFFICIENT_INPUT`, `READY_FOR_DEEP_DIVE=NO`; six observed evidence deficiencies. No verified handoff available in the read-only MCP tool.
- Two `BLOCKED/RESEARCH` CHECKPOINT runs; both gate `NO`.
- One `ACTIVE/RESEARCH` run without active stage manifest.
- One `BLOCKED/DEEP_DIVE` run with economic share-count unresolved.

The MCP `get_orotitan_next_prompt` returns `BLOCKED_OBSERVED`, `verificationPerformed=false`, no verified prompt. The MCP `list_orotitan_submissions` fails with `DELEGATED_QUARANTINE_READ_DISABLED`. This last failure is a **separate technical access gap**; do not attempt to bypass it with an unscoped GitHub token or a public endpoint.

## This PR — phase 1 (diagnosis -> specific recovery task)

1. New pure `buildRecoveryWorklist(OperatorRun)`, based solely on a **Registry observation**, returns one still-open recovery task per observed blocker. Recipes cover contradiction, independent corroboration, technology adoption, audited primary filings, owner cash / ROIC / tax normalizations and share-denominator bridges.
2. Missing stage manifest becomes a technical remediation task; missing Registry blocker detail yields a separate diagnostic item. Nothing is silently considered resolved.
3. Admin-only `/admin/orotitan?run=<uuid>` shows fixed research instructions, explicit close-out evidence and when to escalate to the user if reasonable sources are exhausted.
4. No run status change, no artifact fabrication, no handoff readiness, no direct publication, no mass permissions, no external data reads. The original blocker description is **untrusted data**, never instructions; only fixed recipe text can instruct research.
5. Tests include exactly six Besi blockers, rejected stage leap and missing manifest, denied injection, and a share-count case.

### Crucial limitation

This phase does **NOT** yet close any blocker or resume Besi Research: providing a task plan is not executing it. ChatGPT's existing Besi pilot exposes only three read-only actions. **Do not market phase 1 as an autonomous recovery system.**

## Mandatory phase 2: controlled completion loop

- Provide a scoped, authenticated MCP `prepare_blocker_recovery` and `get_blocker_worklist` (read-only) after OAuth and server tenant identity gates.
- Make Research fetch sources, compare evidence and contradictory claims, and emit an evidence/gap provenance bundle. Retrieval cannot borrow inaccessible sealed GitHub artifacts or user credentials.
- Implement privileged, narrowly scoped **quarantine intake** with binding `run_id`, `stage_revision`, contract pins, content hashes and immutable backend. Authenticate client and authorize server-side; preserve raw source, timestamps and provenance; reject stale CAS state.
- Verify immutable evidence bytes independently; re-evaluate Research sufficiency; write a **new versioned CHECKPOINT** or FINAL only through approved Registry CAS transaction. Do not edit prior sealed artifacts or self-attest a gate.
- When critical evidence remains genuinely unavailable, produce the frozen contract's 5-field human-assistance request: exact document, why, search attempts, substitute, impact. Do not block without an actionable ask.
- Only on independently verified gate `READY_FOR_DEEP_DIVE=YES` may D2 compute a handoff prompt. Later stages and `GO PUBLISH` remain separate.
- Test Besi's 6 blockers and the 4 other V3.1 Registry states end to end in a staging environment with exact role boundaries. Never run automatic production migrations or releases as a side effect of merging this PR.

### P0 acceptance criteria (entire program; not achieved in this PR)

1. A user can say "reprends Besi", and ChatGPT presents the authoritative worklist plus current Registry revision.
2. Accessible missing sources are researched autonomously, with citation/provenance and contradictory search; each result is explicitly `RESOLVED_WITH_VERIFIED_EVIDENCE` or `OPEN`.
3. Every item resolved via **persisted** authenticated and hash-verified artifact revisions, not model memory or plain-text claims.
4. Stage-only re-evaluation reliably continues when evidence permits; otherwise stops with a precise, minimum user intervention.
5. No cross-run leakage, no silent publish, no implicit privileges; screener main and VNext remain stable.

## Release

CI -> review -> optional isolated preview -> explicit approval to merge/deploy in the principal Screener. No publication rights or Registry mutation granted by this worklist.
