# Method-V2 execution Contract Pin Pack freeze V1.0

MISSION_ID = METHOD-V2-EXECUTION-PACK-FREEZE-001
BASELINE = vnext@ba3f31704ff82979bf89ed5d90c0303576b12bb1
CURRENT_BRANCH = feat/method-v2-execution-pack-freeze-001
TARGET_PR = one PR against vnext

MISSION_OBJECTIVE = freeze execution identity from V1 plus five frozen successors.
AUTHORIZED_ACTIONS = inspect authorities and Registry canonicalization; add pack,
deterministic tests, their required CI history checkout and this freeze record;
validate; commit and open one PR.
FORBIDDEN_ACTIONS = production access/mutation, migrations, activation, run creation,
publication, methodology/scoring/Certification changes, historical byte changes,
Runtime V2/V3/V3.1 imports and unrelated changes.
ACCEPTANCE_CRITERIA = exact 13 pins; A01–A14 pass; authority unchanged; relevant
tests, typecheck, lint and build pass; only authorized files in the PR.
DONE_CONDITION = merged-ready PR, without merging or activation.
CURRENT_STATE = V1 pack and five successors frozen; no Method-V2 execution pack.
TARGET_STATE = reproducible inactive execution pack with immutable Git locators.
MINIMUM_CHANGE = one JSON pack, one test file, this freeze record and full-history
checkout in the two existing workflows running npm test.
VALIDATION_PLAN = A01–A14; Method-V2 authority and contract-pin tests; typecheck;
lint; build; scoped diff and PR checks.

The frozen artifact is
`contracts/orotitan-equity/method-v2/OROTITAN_METHOD_V2_EXECUTION_CONTRACT_PIN_PACK_V1.0.json`.
It is execution identity only. The execution support manifest remains provenance
support; it is not the final pack or analytical identity.

METHOD_V2_CONTRACT_SET_SHA256 =
`23b75bf5c2d7448e8270e7e8f3a0223e639d0be3a1406b3c089e6e885dd063ea`

METHOD_V2_AUTHORITY_SET_SHA256 =
`1e97ad30595d24d10345cfcb58c8b6c0feeb7272144af1fc12d0affd2d2e33b2`

Exactly eight full V1 pin objects are retained: research_stage, analysis_standard,
investment_policy, execution_patch, integration_spec, screener_schema, i2 and i3b.
Their names, versions, content hashes and locators are unchanged, including archive
formats. The V1 set digest remains
`34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63`.

The five explicit successor bindings replace process, pilotage, deep_dive_stage,
integration_stage and master_prompt. Their canonical names are the frozen document
identifiers (filename stems), matching the subject of each first-line title.
Each title explicitly says successor V1.0, giving pin version `1.0`.
All five source locators use commit `3ba9cb7e83368155abcca9d44429df490bb08b9e`
in `robzer13/indice_nexus`. They remain scoped overlays over V1, with only the
approved analytical successors; freezing a locator grants no additional authority.

## Deterministic reproduction

The Registry function `public.orotitan_contract_set_sha256(jsonb)` in
`migrations/20260914_orotitan_registry_v1_3_rpcs.sql` sorts logical keys, serializes
each as `logical_key|version|content_sha256`, joins with LF, appends one final LF
and computes SHA-256 over UTF-8 bytes. All 13 keys are ASCII. The JSON pin keys
are stored in that sorted order. Names, locators and metadata are outside this
historical digest algorithm; exact full-pin comparisons and immutable byte
resolution validate those fields independently. The analytical digest uses its
own unchanged manifest canonicalization.

Run `npx tsx --test tests/method-v2-execution-contract-pin-pack.test.ts` from the
repository root with Git history available. A01–A14 verify exact inheritance,
successor names/versions/hashes, current and commit-addressed bytes (including
archive parts), independent canonical digest reproduction, tamper detection and
the unchanged analytical authority. Git object resolution is local and read-only;
no database or production client is used.
The existing VNext and Screener CI checkouts use `fetch-depth: 0` solely to make
the pinned historical Git objects available to these tests.

## Freeze boundary

`production_active = false`. No runtime selector imports this artifact. This pack
creates no run, grant, migration, snapshot, publication or production Contract Set.
Historical bytes, scoring, valuation, Challenge and Certification remain unchanged.
Runtime V2/V3/V3.1 and their packs do not supply any member. Any activation or
change to this frozen identity requires separately authorized work.

## Local freeze verification

A01–A14 PASS. The targeted execution-pack, Method-V2 authority, V1/V2 contract-pack
and pin-source suites passed: 61 tests, zero failures. Historical runtime V2 tests
are regression checks only and contribute no authority to this pack.
Typecheck, lint and build PASS. The first sandbox build compiled successfully but
failed to spawn a worker (EPERM); the single retry outside the sandbox passed.
Only the verified Next.js-generated diffs in tsconfig.json and next-env.d.ts were
restored. No historical authority, production state or activation was changed.
Required remote CI and review results are reported on the PR independently.
