# METHOD-V2-RUNTIME-FIREWALL-001 — inactive implementation

Original baseline: `vnext@e5c3b107116a998496eb15a3c0a81546c3c827b2`.
Resumed base after independently merged PR #372: `vnext@cf83831d13b87bc892aef59e2951c7c4518119e8`.
Branch: `method-v2-runtime-firewall-001`. One mission / one branch / one PR; no merge authorization.

## Mission lock and scope

Objective: complete existing PR #371's inactive Registry V1.14 firewall with exact persisted Certification Challenge lineage verification for NEW Method-V2 artifacts.
Authorized actions: controlled merge of origin/vnext into the existing branch, Certification profile pin and binding rehash, applicable V1.14 SQL/TypeScript/documentation/tests, required local validations, commit/push and fresh Codex review on PR #371.
Forbidden actions: production access/mutation, activation, selector/API wiring, runtime-role creation EXECUTE, frozen authority/profile/pin changes, scoring changes, canonical snapshots or Method-V2 atomic publishing.
Acceptance: B01–B44, exact Certification lineage and complete concern propagation, historical grandfathering/reopening/concurrency/replay regressions and all required checks pass; CI/reviews have no unresolved blocker.
Done: a merged-ready inactive implementation PR. Maximum conclusion: SAFE TO REQUEST METHOD-V2 RUNTIME FIREWALL MERGE.

## Frozen identities

| Identity | SHA-256 |
| --- | --- |
| Analytical authority | `1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2` |
| Exact 13-pin Contract Set | `23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea` |
| Evidence Ledger persistence profile | `8679e2aeb8f7be4569670629866a9ee2a63d933f5b04d6ede209aa8310e29a83` |
| Certification persistence profile V1.0 | `a9b3eff1930a9a7e6164cfb1f0248c98c8041975ef9adc77f9d152062df0a53c` |
| Runtime binding, exact raw bytes | `b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0` |

The binding creates execution identity only. production_active, publication_active and canary_active are false. The current production selector, frozen authority/profile/parser/pin bytes and V1.13 migration are byte-checked against the mission baseline.

## Database boundary

`20261007212324_orotitan_registry_v1_14_method_v2_runtime_firewall.sql` follows V1.13. Its operational singleton cannot be deleted/truncated. Defaults: INSTALLED_INACTIVE / INITIAL_ONLY / publication DISABLED, with NULL deployment commit. A fully bound 40-character deployment commit is required before CANARY_ONLY or ACTIVE_FOR_NEW_RUNS. There is no activation RPC.

The dedicated relational canary table is empty initially and accepts only exact `(dossier_id, issuer_id, security_id)` tuples, with issuer/dossier/security FKs. No partial match, wildcard, fallback or self-enrollment exists. CANARY_ONLY never overrides normal identity, snapshot CAS, runtime binding, cutoff or admission_scope checks. ACTIVE_FOR_NEW_RUNS does not require canary membership. These control surfaces and the proof table allow service_role reads only; PUBLIC/anon/authenticated have no access.

The SECURITY DEFINER fresh-run RPC has `search_path = pg_catalog, public` and no PUBLIC/anon/authenticated/service_role EXECUTE grant. It derives all analytical, pin, version and routing fields. It serializes creation keys, locks the exact active dossier row, reconciles the exact security/issuer, and compares the snapshot pointer. NULL pointer derives INITIAL/ANALYZE/NULL baseline. REFRESH requires local INITIAL_AND_REFRESH, the exact baseline dossier/issuer/security and a strictly later cutoff. Fresh runs are fully bound at birth with NULL parent. The partial unique index covers all five non-terminal fresh Method-V2 statuses only for the exact frozen V1.14 runtime binding and excludes controlled successors and grandfathered unbound runs. The admission precheck uses the same discriminator.

Exact request/fingerprint replay returns the same run, even after admission is disabled; changed identity or fingerprint rejects. The legacy insert firewall rejects new METHOD_V1 company-analysis runs with NULL parent when future Method-V2 admission is enabled. Existing replay performs no insert. V1.13 successor functions and historical stage/RPC definitions remain unchanged.

## Persisted-byte and lineage boundary

The server adapter reuses `lib/supabase/server.ts` for read-only Registry and Storage access. It obtains exact full run pins, stage revision, Registry artifacts and CONSUMES edges. Callers can identify an exact FINAL candidate manifest ArtifactRef, but cannot supply paths, bytes, analytical identities or PASS claims.

The existing owner Registry registration surface prepares the FINAL candidate before verification. The candidate resolver rereads exact persisted bytes, checking run/revision/Contract Set and exact output refs. This permits verification while the stage still points at a previous CHECKPOINT, before finalization executes its database guard. Current Locks reconcile to the same FINAL bundle. No production execution role is provisioned by this mission.

The byte resolver verifies artifact UUID/version, run/stage/type, expected authority class/state, SEALED and AVAILABLE states, backend, approved bucket, exact Registry object path/URI and size/SHA. It downloads only Registry-derived coordinates and recomputes size/SHA. Challenge, Evidence Ledger and Certification JSON must use `orotitan-text-artifacts-v1`. The other approved bucket is `orotitan-source-files-v1`. There is no cached PASS result.

The Challenge verifier rereads Ledger, Report and both Locks and then calls the unchanged pure `admitMethodV2Certification`. DELTA requires an exact prior owner-recorded proof and rereads prior artifacts through that same validator. Proven superseded historical versions are allowed only on this prior-proof path; current superseded artifacts reject. Cyclic prior lineage rejects.

The server adapter selects the unique authoritative Certification from the exact current or FINAL candidate bundle independently of its claimed binding and edges. The verifier rereads its exact persisted bytes through the same resolver, then uses the unchanged PR #372 `parseMethodV2CertificationArtifact` and `verifyMethodV2CertificationChallengeBinding`. Trusted expected values come from Registry run/revision/cutoff, the four exact Challenge refs and independently validated Challenge status/limitations. PASS requires an empty array; PASS_WITH_CONCERNS requires exact set equality for every question ID, decision impact and mitigation/resolution, without trimming or paraphrase. Omissions, additions, duplicates, altered values and stale refs fail closed. Opaque inherited Certification content still participates in the raw-byte hash but no judgment, scoring or legacy issueRef conversion is performed.

Owner-only proof identity `verifyPersistedMethodV2Challenge:1.1` includes the Certification ID/version/raw SHA-256 and exact profile SHA-256. SQL requires that proof to match the current Certification as well as the Challenge and current Locks. A Certification→Challenge CONSUMES edge alone cannot admit Deep Dive completion or Integration. Prior DELTA Certification is selected from its exact immutable proof and reread/validated against that prior Challenge, never replaced with a newer active artifact.

B41 follows the exact coverage evidence reference → Evidence Ledger ArtifactRef → Registry CONSUMES lineage → exact Storage bytes → size/SHA → ONLY `parseMethodV2EvidenceLedger` → direct case-sensitive `entries[i].EVIDENCE_ID` chain. Aliases, recursive evidence search, duplicate identity syntax/values and inferred evidence dates are not accepted. Cutoff checks remain in the frozen Challenge validator.

`verifyAndRecordMethodV2Challenge` sends only successful verifier output to an injected trusted owner proof sink. Failed persisted-byte or analytical validation never reaches it. The sink provides no production owner connection, public self-attestation endpoint, or runtime-role write grant.

The database guard applies only to METHOD_V2 runs bearing the exact frozen runtime binding b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0. Pre-V1.14 METHOD_V2 runs retain NULL runtime bindings and their existing execution behavior; no backfill or silent upgrade occurs. For bound runs, it requires all inherited FINAL DEEP_DIVE outputs plus Ledger, Report and Certification, an immutable exact verifier proof for the current revision, authoritative current FINAL Locks, and exact current-run CONSUMES edges Ledger/Valuation → Challenge → Certification. Integration insert/update rechecks the current COMPLETE/FINAL/ready DEEP_DIVE and rejects stale or invalidated lineage. Challenge remains internal to DEEP_DIVE. No formula, Investment Policy or publication path is changed.

## B01–B44 evidence map

| Tests | Executable evidence |
| --- | --- |
| B01–B06 | PostgreSQL singleton, enum, defaults, actual role ACL denials, owner-only empty canary table |
| B07–B10 | Exact control identities, raw binding and full immutable pack locator; baseline byte checks including selector |
| B11–B16 | PostgreSQL function safety/arguments/ACLs, denied service_role execution, full pins and same-digest name tampering rejection |
| B17–B24 | Exact dossier/security routing, CAS, INITIAL/REFRESH and baseline/cutoff rejection; real dossier-lock timeout |
| B25–B27 | DB unique-index rejection; actual distinct-key race/exact-key replay; fingerprint and request conflicts |
| B28–B32 | Empty/wrong/exact canary tuples, scope cannot be overridden, inactive/disabled rejection, ACTIVE admission without membership |
| B33–B36 | Inactive legacy create, post-enable replay, new legacy rejection; full Registry V1.6–V1.13 and successor concurrency regressions; actual V1.13 fresh creation/binding fixtures, historical FINAL/Integration mutations after V1.14, and every historical row checked for no rewrite/backfill |
| B37–B39 | Registry authority/status/identity and backend/bucket/path tampering; exact downloaded size/SHA checks |
| B40 | Challenge/Locks/DELTA rereads, unchanged validator, proof-sink failure isolation, FINAL candidate bytes/revision |
| B41 | Full persisted-byte identity chain, wrong ref/case, alias/nested ID, duplicate member/value, storage tamper, cutoff ownership |
| B42–B43 | Individual missing Ledger/Report/Certification, missing proof/edges, stale proof revision/Lock/Certification version, valid current lineage and Integration bypass rejection; complete persisted Certification binding/concern adversarial matrix and offline real server-adapter current/candidate tests |
| B44 | Historical regressions, frozen selector/authority byte checks, no activation/grant/publisher/snapshot implementation |

Local clean-room environment: disposable `postgres:17`, actual **17.11 (Debian 17.11-1.pgdg13+2)**, separate from the existing local Supabase stack. The Registry runner requires major 17, checks unchanged canonical/legacy identity rows and drops its disposable database. Synthetic scope/mode changes roll back or reset to installed inactive.

Original implementation validation (historical): full Registry V1.6–V1.14 and real concurrency PASS; selected authority/pack/profile/Pre-Certification/resolver suite 82 PASS; complete unit suite 1,528 PASS; subsequent focused resolver/candidate/proof-sink suite 12 PASS; typecheck PASS; production build PASS with inert local Supabase environment values. Lint PASS. External CI/reviews are reported with the PR/end-of-mission status.

Resumed mission: the old runtime binding SHA `0832d3c90afab1e4e044d0b84af3992edcb2961298b379890cd7d978d391ca2c` is superseded by the newly computed raw-byte SHA above. The binding pins the exact Certification profile at commit `cf83831d13b87bc892aef59e2951c7c4518119e8`, blob `1e44968777667d4aced85a288e6c673e53cf5c4d`. All eight applicable V1.14 binding literals and the concurrency fixtures use the new hash; frozen analytical authority, 13-pin Contract Set, both profile bytes and pure parsers/verifiers remain unchanged. The P1 thread is not marked resolved before tested implementation and independent review. Latest validation and external review results are reported on the existing PR #371. No production access, activation, publication or PR merge is authorized.

Resumed validation: full Registry V1.6–V1.14 PostgreSQL 17.11 and real reopening/concurrency/replay regressions PASS; full unit/contract suite 1,609 PASS, including the new persisted Certification and offline server-adapter matrix; final typecheck and lint PASS; production build PASS with inert `http://127.0.0.1:9` Supabase values. Windows sandbox worker-spawn EPERM required the authorized test/build retries. The new adapter-test harness was corrected for Windows command-line length (stdin) and the repository's CommonJS module representation; no product change was made for those test-execution failures. Generated-only Next.js configuration edits were inspected and restored.

Controlled corrections: throwing-helper TypeScript narrowing (first correction, passed); current-manifest sequencing corrected by exact candidate-manifest verification (first correction, passed). Windows worker-spawn EPERM was resolved by one authorized build retry; product code was not changed for the infrastructure error. Generated-only Next.js tsconfig changes were inspected and restored.

P1 grandfathering correction: stage enforcement, the fresh-run partial index and admission precheck now require the exact frozen runtime binding. Revalidation on 2026-10-08 passed in an isolated disposable PostgreSQL 17.11 container: full Registry V1.6–V1.14 regression, grandfathering/no-backfill/adjacent runtime-discriminator checks, B42/B43 adversarial guards, and actual concurrency. The existing local Supabase stack was neither used nor mutated. A PL/pgSQL variable/table-alias ambiguity in the new grandfathering test was corrected on its first targeted attempt; the firewall SQL fix was unchanged. The selected 84 authority/pack/profile/Pre-Certification/resolver tests, typecheck, lint and inert-environment production build passed. The build required one Windows worker-spawn EPERM retry outside the sandbox; generated-only Next.js configuration changes were inspected and restored. Runtime binding bytes and SHA-256 remain unchanged. CI/reviews are tracked on the same PR #371; no merge or activation is authorized.
