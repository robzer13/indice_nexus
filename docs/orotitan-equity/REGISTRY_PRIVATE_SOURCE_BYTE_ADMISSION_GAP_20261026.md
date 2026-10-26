# OroTitan Registry — source-byte admission infrastructure gap (review-only, 2026-10-26)

**Status: DESIGN / NO PRODUCTION MIGRATION / NO SOURCE ADMISSION.** This document concerns a generic process defect, not an instruction to rewrite any frozen run or to move private research to a public repository.

## Verified runtime limitations

Production read-only audit showed that six Registry persistence functions contain `PRIVATE_GITHUB`-only logic. The candidate-staging, locator-attestation and bundle-validation functions additionally pin the same legacy private GitHub repository. The Registry has private Supabase Storage buckets, but merely creating these buckets does **not** make that backend eligible for the existing writer's attestation.

Affected routines (not necessarily exhaustive downstream enforcement):
`stage_orotitan_persistence_candidate`,
`attest_orotitan_persistence_locator`,
`orotitan_insert_artifact_registration`,
`orotitan_validate_attested_bundle_object`,
`orotitan_validate_manifest_persistence_receipt`,
`orotitan_verify_existing_manifest_persistence_receipt`.

A legitimate issuer-hosted PDF captured over TLS with content SHA-256 and issuer URL is currently **CAPTURE_ONLY_UNATTESTED**. Hashing the retrieved document proves present-day byte integrity, not the original published-before-cutoff state or Registry admission.

## Required architecture for a new private storage transport

1. Keep application source and schema migrations in the **approved implementation repository**; keep private research case artifacts in access-controlled object storage, not the public implementation repo.
2. Define a distinct `PRIVATE_SUPABASE_STORAGE` backend with a non-public, approved `bucket_id` and canonical object path. No service-role keys in the browser, GitHub artifacts, client workflows, or committed migrations.
3. Provision a privileged **ingestion** service that streams captured source bytes to a private object with immutable/versioned identity and a deterministic SHA-256, recording media type, original issuer URL, publisher, publication-date evidence and actual capture time.
4. An **independent attestor** with server-side trust boundary must re-read the persisted private object, hash those exact bytes, check locator+version+SHA+size+media type, and emit a signed/independently verifiable attestation under a *new governed verification method*. An uploader-provided SHA is not an attestation.
5. Only then modify Registry storage-backend allowlists in *all* six routines and downstream schema/persistence-receipt checks. Explicitly version the attestation schema and require `SOURCE_CAPTURE != HISTORICAL_CUTOFF_ADMISSION` unless independently supported by cutoff-eligible primary filing/publication evidence.
6. Enforce no overwrite of immutable source versions, short-lived signed reads, minimal writer role, RLS or strictly private schemas, idempotent staging, CAS on run/stage versions, immutable audit events and dual-reader verification. Review existing `SECURITY DEFINER` RPC EXECUTE grants and check caller authorization before exposing new paths.
7. Write negative PostgreSQL regression tests: forged receipt, SHA mismatch, wrong bucket, outside-run path, overwritten object, missing independent re-read, post-cutoff evidence, wrong stage/manifest, duplicate replay, role bypass, mixed backends, and malicious URL. Keep existing Private GitHub historical registrations immutable and readable.
8. Deploy the migration separately after review + a verified restore/rollback strategy. No production rewrite or implicit Stage finalization. Refuse checkpoint if any receipt is absent, incomplete or inconsistent.

## Live proof discipline

Original-source issuer reports and financial statements can be collected in public GitHub Actions solely when they are **public**. Their SHA and GitHub Actions archive receipts remain non-authoritative until independent private-storage re-read. The original capture run is not itself the immutable Registry Artifact; do not register a public GitHub CI artifact as a private Registry locator.

## Safe next step

Implement and test a backend-neutral writer/attestor in a separate reviewed change, then perform a read-only dry run and a signed test object in a **nonproduction** project. Only after the governed controls pass should an existing in-progress Deep Dive candidate be attached to a new immutable checkpoint using its own frozen contract pins and cutoff. Do not mark the report `READY_FOR_INTEGRATION` or authorize valuation/publishing on a transport-success signal.
