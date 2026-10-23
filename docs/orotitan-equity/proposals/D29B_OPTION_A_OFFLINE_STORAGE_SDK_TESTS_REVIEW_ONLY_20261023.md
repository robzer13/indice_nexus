# D29b — Option A / zero-hosted-cost SDK Storage protocol tests

**Status: DRAFT / REVIEW ONLY. Stacked on PR #407 (and #406, #405, #404, #403).**

## User decision and cost guard

The user selected **Option A**, continuing D29b on the existing PostgreSQL 17
and GitHub Actions test infrastructure without enabling billable Supabase
resources. No Supabase Preview Branch or Pro plan has been created or approved.

The baseline currently passes a disposable PostgreSQL 17 integration using the
**actual Registry V3.1 `pause_orotitan_stage` RPC** concurrently with D29b
quarantine receipt creation. That validates the database half, but the blob
Storage adapter previously had only manually mocked `supabase.storage`
methods. This PR adds the next **offline** layer without any hosted database.

## Implementation

`tests/equity-v3-research-supabase-sdk-loopback.test.ts`:

- Uses the project-pinned actual `@supabase/supabase-js` client
  (`2.110.7`) and actual `createSupabasePrivateResearchBlobPort`.
- A Node.js HTTP fixture bound to `127.0.0.1` with a random ephemeral
  port implements only the minimal Storage `POST /storage/v1/object/...`
  and private `GET /storage/v1/object/authenticated/...` response semantics
  needed for the contract tests. All credentials are synthetically generated
  strings, never Supabase keys.
- Verifies the SDK's actual multipart transfer includes the candidate bytes,
  the upsert header is not true, the object key is constrained to the
  designated private bucket/prefix, and readback reproduces exact bytes.
- Verifies duplicate upload yields an error and does not overwrite; missing
  bytes, invalid storage path and HTTP failure do not silently return success.
- Runs one `submitResearchDraftCandidate` through the real SDK transport
  with the private D29b orchestrator and a **fake** DB transaction port:
  only `RECEIVED_NOT_VERIFIED / QUARANTINE_ONLY`, never admission.
- Simulates corrupted Storage readback and HTTP 503, asserting that the
  transaction commit is not called and unadmitted candidate bytes are not
  promoted.
- The project `npm test` glob (`tests/*.test.ts`) automatically includes
  these tests. No added dependency, Docker installation, paid CI service,
  cloud route or secret is required.

## Exact limits of what this proves

The local HTTP fixture is a **protocol emulator, not self-hosted Supabase
Storage**. It does NOT prove Storage metadata RLS, JWT signature checking,
PostgREST, service-key isolation, concurrent object storage semantics,
server-side rate limits, hosted Supabase pricing or production authorization.
Those still require an independently reviewed isolated environment and the
user's separate cost authorization if such an environment is billable.

These tests also do **not** validate Research evidence content or discharge
the six Besi `OPEN_UNVERIFIED` blockers. They make no changes to the active
run or its immutable Registry state. No `main` merge, OAuth upload tool,
migration, Supabase workspace change, Vercel deployment, stage transition
or Deep Dive admission is part of this PR.

## Next no-cost engineering gates

1. Add a local **real Supabase Storage service** stack if available within
   existing hardware/CI resource limits, with completely synthetic data.
2. Validate Storage RLS with separate private writer and D2 reviewer
   credentials, source byte hashes and no bucket listing.
3. Provide a separately auditable draft object reconciler design before any
   real orphan deletion; request-path deletion on ambiguous commits is banned.
4. Commission independent security review of the full stacked PR chain
   prior to any production privilege/contract change.
