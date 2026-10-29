# OroTitan — signed independent source reread and nonproduction smoke (review only)

This change is stacked on the unmerged private-source preflight PR. Status: NOT PRODUCTION-READY, NO REGISTRY ADMISSION.

## Technical boundary implemented

The uploader produces an immutable-looking candidate path and SHA-256 and **cannot sign** the proof with the attestor key. A separate reader service invokes `signRereadFromPrivateStorage()` with an Ed25519 private key accessible only to that process; it signs the *result of a storage reread*, not a user-supplied hash. The verifying process invokes `verifySignedReread()` with its own pre-pinned public key, attestor ID and key ID, a fresh challenge nonce, a bounded timestamp, and the exact expected source locator.

Cryptographic verification proves a signature under the configured public key over a statement claiming the exact bytes and path. It does NOT on its own prove operational principal separation or immutability. The receipt explicitly returns `registryAttestation=false`, `registryAdmission=false`, `historicalCutoffBytesProven=false`, `independentPrincipalProven=false`.

No production writer or PostgreSQL PERSISTENCE_ATTESTED function accepts this new receipt. It is intentionally non-authoritative until a separate governed Registry migration and a management-plane independent identity audit are complete. Source publication dates remain declared, not attested.

## Shadow project prerequisite

The nonproduction OroTitan shadow has Registry tables but **no Storage buckets provisioned** at this review point. Provision a `public=false` bucket with ID `orotitan-source-files-v1` on the nonproduction shadow only, via the Supabase Storage API or dashboard. Configure the two separate Auth users with dedicated **per-user RLS**; the uploader must have INSERT without SELECT, UPDATE or DELETE on `storage.objects`. The attestor must have authenticated object GET only, without INSERT, UPDATE or DELETE. The bucket metadata GET also requires applicable bucket access; configure `storage.buckets` SELECT for these two users only if necessary.

Policy sketch for **shadow only** (replace placeholders with real, independently verified Auth UUIDs, preferably use the project Dashboard policy editor; do not apply to production or copy into a frozen Registry migration):

    -- storage.objects: create-only uploader
    CREATE POLICY "shadow source writer" ON storage.objects
    FOR INSERT TO authenticated WITH CHECK (
      bucket_id = 'orotitan-source-files-v1'
      AND auth.uid() = '<SHADOW_WRITER_UUID>'::uuid
      AND (storage.foldername(name))[1] = 'preadmission'
    );

    -- storage.objects: reader can GET only, not LIST (operation-aware)
    CREATE POLICY "shadow independent rereader" ON storage.objects
    FOR SELECT TO authenticated USING (
      bucket_id = 'orotitan-source-files-v1'
      AND auth.uid() = '<SHADOW_READER_UUID>'::uuid
      AND storage.allow_only_operation('storage.object.get_authenticated')
      AND (storage.foldername(name))[1] = 'preadmission'
    );

    -- storage.buckets metadata for both nonprod users, if required by SDK getBucket:
    CREATE POLICY "shadow private source bucket metadata" ON storage.buckets
    FOR SELECT TO authenticated USING (
      id = 'orotitan-source-files-v1'
      AND auth.uid() IN ('<SHADOW_WRITER_UUID>'::uuid, '<SHADOW_READER_UUID>'::uuid)
    );

Before enabling these, inspect **all existing policies** on storage.objects and storage.buckets. Broad policies can combine to bypass an apparently restrictive policy. No service_role or elevated key for either testing principal; those bypass RLS entirely. Source: https://supabase.com/docs/guides/storage/security/access-control .

## Opt-in smoke invocation

Run only in a secured, isolated nonproduction operator environment after provisioning the two users and bucket. Environment variables must be provisioned in a secrets manager, never committed or logged:

- PREFLIGHT_NONPROD_APPROVED=YES
- SUPABASE_TEST_URL (exact https://<test-ref>.supabase.co)
- SUPABASE_TEST_REF (20-char nonproduction project ref)
- SUPABASE_PRODUCTION_REF (distinct production ref, must not equal test ref)
- SUPABASE_TEST_PUBLISHABLE_KEY
- SUPABASE_TEST_WRITER_JWT and SUPABASE_TEST_READER_JWT (two distinct server-verified authenticated identities)
- ATTESTOR_TEST_PRIVATE_KEY_PEM_BASE64 (test-only Ed25519 private key)
- ATTESTOR_TEST_PINNED_PUBLIC_KEY_PEM_BASE64 (the independent verifier's approved matching public key)

Command: `npx tsx scripts/orotitan-equity/shadow-private-source-smoke.ts`

The smoke script verifies Auth identity via Supabase Auth, uploads **synthetic PDF content only** under a new random synthetic run/artifact UUID, verifies that uploader download is denied and reader upload is denied, rereads/signs/verifies the allowed object, and outputs only generic metadata (SHA, object URI, test results). The private test object is intentionally retained in shadow for later byte re-check; clean up only under an explicit governed retention process. **Never use ASML documents in shadow tests.**

## Fail-closed deployment gates

- No real smoke can pass until bucket/policies and two scoped user tokens exist. Local simulated clients and full CI do not substitute for a live RLS test.
- A signing key provisioned together with the uploader credentials does not establish separation. The smoke holds both under a test runner, so `independentPrincipalProven` remains false.
- Add real issuer timeline evidence and immutable object retention. Current signature says nothing about bytes on a historical cutoff.
- Next governed development: dedicated server-side attestation endpoint/process with verifiable auth binding, private Storage object version immutability, and a forward-only update to all active persistence-registration and receipt-checking Registry functions. Test exclusively in disposable PostgreSQL/isolated project first. Historical Private GitHub records and frozen V3 run are not migrated implicitly.
- Any live migration remains blocked pending schema/role audit, negative-tests, migration review and dry-run, and explicit production rollout authorization.

There is no automated secret-using CI step. The smoke is an explicit manual nonproduction process, separate from automated synthetic tests.
