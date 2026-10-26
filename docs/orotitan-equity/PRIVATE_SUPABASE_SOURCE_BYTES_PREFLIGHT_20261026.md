# OroTitan V3 — Private Supabase source-byte preadmission (2026-10-26)

Status: REVIEW ONLY / NO PRODUCTION ACTIVATION. Public implementation module only: no private case content, company names, run identifiers or credentials.

## Delivered

- uploadPrivateCandidate verifies public issuer URL, declared source publication date within the cutoff, PDF/HTML/JSON structure, source bytes, SHA256, UUIDs, approved private bucket; then performs one create-only Supabase Storage upload (upsert false) under preadmission/run_uuid/artifact_uuid/vNNNN/content_sha.ext.
- rereadPrivateCandidate uses a separately provided reader client to re-check the non-public bucket, immutable-looking object path, exact byte size and SHA-256.
- Returns BYTES_REREAD_MATCHED_UNATTESTED with registryAttested=false, registryAdmitted=false, independentPrincipalProven=false, storageImmutableProven=false, and cutoffHistoricalBytesProven=false.

## Security and rollout gates

1. A server-only uploader and a separate server-only reader must have genuinely distinct, least-privilege credentials. Injection of two clients does not itself prove separation. No service-role key in browser or public CI.
2. Storage must disallow UPDATE and DELETE for ingestion roles; content-addressed path and upsert=false alone do not prove immutable retention. Require a verified data retention and restoration policy.
3. Verify with synthetic data on a NONPRODUCTION project. No canonical equity run must be altered.
4. Independently validate publication timeline and cutoff evidence. A declared date or retrieval hash from a later day is not sufficient to attest historical bytes.
5. Prepare a separate forward-only registry migration for all six persistence and receipt validation functions, immutable event attestor, guarded authorization, and predecessor compatibility; new source bytes must not be marked AUTHORITATIVE by this module.
6. Negative tests must include forged receipt, wrong principal/bucket, replacement, post-cutoff source, SHA mismatch, wrong source URL, replay and stage/version mismatch. Any Registry transition requires an authorized independent writer and new immutable stage manifest.
7. Existing private-GitHub artifact records, frozen run pins, valuation restrictions and publication gates remain unchanged.

The production Registry table already supports the SUPABASE_STORAGE backend, but active writer/attestation functions require PRIVATE_GITHUB. This PR intentionally does NOT change those functions or claim to unblock Registry source admission. It solves only create-only storage and separate client byte re-read preflight, pending security and governance gates.

SDK integration with actual Supabase Storage must be separately tested using read/write principals in non-production before any real intake.
