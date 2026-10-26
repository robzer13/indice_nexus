# OroTitan shadow — real Storage ACL setup (2026-10-26)

**Status: SHADOW CONFIG DEPLOYED; AUTH USERS NOT CREATED; NO REAL HTTP SMOKE.**

Supabase's documented SQL bucket creation path was used in the *nonproduction* shadow project to configure one private bucket with 35 MiB maximum object size and allowed MIME PDF/HTML/JSON. Three authenticated RLS policies apply: role-tagged writer (INSERT only under preadmission), role-tagged attestor (operation-scoped authenticated GET only), and bucket metadata read for either approved role. They match the preflight path/roles, and use **admin-controlled app_metadata**, not user_metadata. No UPDATE or DELETE policy added.

## Exact SQL executed on shadow

```sql
-- SHADOW ONLY. Executed as migration orotitan_shadow_private_source_bucket_acl_20261026.
-- Not included in automatic production migration chain.
BEGIN;
INSERT INTO storage.buckets (id,name,public,file_size_limit,allowed_mime_types)
VALUES ('orotitan-source-files-v1','orotitan-source-files-v1',false,36700160,ARRAY['application/pdf','text/html','application/json']::text[]);
CREATE POLICY "orotitan_shadow_source_bucket_metadata_v1" ON storage.buckets
FOR SELECT TO authenticated USING (
 id='orotitan-source-files-v1'
 AND (auth.jwt()->'app_metadata'->>'orotitan_shadow_source_role') IN ('writer_v1','attestor_v1')
);
CREATE POLICY "orotitan_shadow_source_create_only_writer_v1" ON storage.objects
FOR INSERT TO authenticated WITH CHECK (
 bucket_id='orotitan-source-files-v1'
 AND (storage.foldername(name))[1]='preadmission'
 AND (auth.jwt()->'app_metadata'->>'orotitan_shadow_source_role')='writer_v1'
);
CREATE POLICY "orotitan_shadow_source_get_only_attestor_v1" ON storage.objects
FOR SELECT TO authenticated USING (
 bucket_id='orotitan-source-files-v1'
 AND (storage.foldername(name))[1]='preadmission'
 AND storage.allow_only_operation('object.get_authenticated')
 AND (auth.jwt()->'app_metadata'->>'orotitan_shadow_source_role')='attestor_v1'
);
COMMIT;
```

**Do not re-apply to an already-configured project**: policy/bucket names are unique; this is an immutable review snapshot of an executed migration, not an idempotent or production migration.

## Actual verification

The deployed bucket is private, has the expected size/MIME restrictions and zero persisted objects; storage.objects and storage.buckets have RLS enabled. SQL tests with `SET LOCAL ROLE authenticated` and simulated signed-JWT claim values returned:

- writer bucket metadata SELECT: 1 (visible)
- writer authenticated GET over an in-transaction synthetic object: 0 (denied)
- attestor bucket metadata SELECT: 1
- attestor authenticated GET: 1 (visible)
- attestor list: 0 (denied)
- unrelated authenticated user bucket SELECT: 0
- anonymous bucket SELECT: 0
- writer INSERT under preadmission: accepted; fixture independently observed before ROLLBACK
- attestor INSERT and unaffiliated user INSERT: RLS reject SQLSTATE 42501
- writer INSERT outside preadmission: RLS reject SQLSTATE 42501
- writer UPDATE: did not change fixture metadata under RLS
- SQL direct DELETE is prevented by built-in storage.protect_delete() and **does not alone establish an HTTP API delete test**.

All synthetic storage.objects fixtures were rolled back (no orphaned metadata). No payload bytes were uploaded to the Storage HTTP service. These are **SQL RLS simulations**, not signed Auth user sessions.

## Next required steps (no user credentials in this repository)

1. In the shadow **only**, provision two separate Auth users via supported Auth Admin API or Dashboard, with test-only email addresses and credentials stored in a secrets manager. The connector currently has no Auth Admin createUser action; do not insert raw auth.users rows directly.
2. A trusted admin must assign user A `app_metadata.orotitan_shadow_source_role = writer_v1` and user B `app_metadata.orotitan_shadow_source_role = attestor_v1`, via Auth Admin API. Make sure the metadata lives in `app_metadata` (not user-editable `user_metadata`). Refresh both sessions/JWTs.
3. Execute the opt-in `scripts/orotitan-equity/shadow-private-source-smoke.ts` with nonproduction URL/ref, distinct signed Auth JWTs, a non-service publishable key and distinct test Ed25519 keys, in a private operator environment. The script must verify each Auth user via getUser and both RLS negative HTTP operations.
4. Record immutable source-blob/read receipts and independently demonstrate different privileged execution identities; the current signer cannot prove separation merely by receiving two clients.
5. Only then review forward-only Registry writer migration with receipt schema and hash revalidation. Keep production frozen methodology/ASML run untouched until certified.

No service_role key or JWT is committed; no authority/Registry records were registered. The public source transport PR and the private preflight PR are separate reviewed dependencies.
