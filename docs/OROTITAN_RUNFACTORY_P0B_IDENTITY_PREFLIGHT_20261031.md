# OroTitan Run Factory P0-B — identity + run reuse, read-only SHADOW
Date: 2026-10-31 | status: SOFTWARE CANDIDATE

## Goal
An issuer-independent, deterministic, fail-closed PRE-FLIGHT executed from:
`/admin/research-factory/run-preflight`

## Already existing authority used
- Identity and lineage tables: `issuers`, `securities`, `research_dossiers`,
  `research_snapshots`, `orotitan_runs`.
- Three verified-structure contract profiles from P0-A.
- Existing controlled RPCs: `create_orotitan_run`,
  `bind_orotitan_run_identity`, and governed successor RPC.
- V3.0.3 production bootstrap is already implemented; **NOT invoked** from SHADOW.

## Dry run algorithm
1. Require explicit Contract Profile, valid cutoff, ISO 6166 ISIN.
2. Exact ISIN lookup in SHADOW security registry; do NOT fuzzy-match an issuer.
3. If multiple listings share the ISIN, operator specifies exchange or resolve ambiguity.
4. Resolve exact issuer and exactly one active dossier.
5. Resolve current canonical snapshot if present, check issuer/security/dossier,
   obtain baseline contract version. No snapshot -> INITIAL, snapshot -> REFRESH.
6. Check existing noncancelled runs for issuer/cutoff:
   - different Contract Set -> conflict; governed successor review
   - identical Contract Set, one consistent run -> REUSE candidate
   - >1 identical run -> ambiguity block
   - no run -> NEW RUN PLAN CANDIDATE
7. Record next action, export evidence of the **observation**, not a run.

## Hard security rules
- Admin authorization + origin + exact hosted SHADOW environment required.
- The API uses SELECTs only with hard limits. No RPC, INSERT, UPDATE, UPSERT or DELETE.
- No arbitrary environment, URL, contract-pin content or run UUID is accepted.
- Output cannot claim GO, contract byte-verification, production environment,
  run creation, Registry admission or stage transition.
- No immutable identifiers are generated in the dry-run.
- Invalid/ambiguous/missing records fail closed.
- Query failure is UNKNOWN/ERROR, never misrepresented as "issuer absent".
- The cutoff is compared as a date, and snapshot identity is validated.
- The V3.0.3 builder requires production environment and must never be made
  to pass by feeding fake production observations in SHADOW.
- The user has approved the workflow architecture, not issued a controlled GO
  authorizing an actual run creation in a verified production environment.

## Known pilot L’Oreal result
The SHADOW identity Registry has no L’Oreal security/issuer record at
this time; the honest result is `BLOCKED_SECURITY_NOT_FOUND`.
Official ISIN FR0000120321. This absence does not imply the company
is absent from the OroTitan production registry. No cross-environment
identity assumption is permitted.

## Technical acceptance
- New fixtures test exact ISIN checksum, initial/refresh, duplicate reuse,
  missing issuer/dossier, ambiguous listing/dossier, snapshot mismatch,
  immutable Contract Set conflict, and no-GO/no-RPC guarantees.
- Build, TypeScript and lint required before release.
- P0-A remains unmodified and uses Git-pinned contract bytes.
- L’Oreal and ASML financial evidence never becomes run identity authority.

## Next P0-B extension (not included)
- Governed issuer/security/dossier onboarding, with a verified independent
  reference for legal entity, ISIN and listing.
- Authorized environment selection, per-run permission gates, and
  original contract bytes reverified for the selected profile.
- The active official, version-specific bootstrap produces exact
  idempotent RPC args. The controlled RPC is invoked ONLY after explicit GO
  and complete eligibility. Read back persisted run + state_version,
  then initialize Research only through official controlled stage rules.
- Make authorization an audited command, not an inferred user utterance.
- Integrate checkpoint resume and machine-measured latency / cost.

No Registry writes, no production changes, no new stage.
