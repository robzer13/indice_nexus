# OroTitan V3 — authenticated shadow Storage smoke / operator runbook (2026-10-27)

**Status: READY TO RUN WITH LOCAL CREDENTIALS. NOT EXECUTED OVER HTTP. NOT A REGISTRY CERTIFICATION.**

The project shadow has an existing private source-files bucket, three restrictive Storage RLS policies, and two real email-confirmed Supabase Auth users. Admin-controlled `app_metadata.orotitan_shadow_source_role` was checked via privileged read-only SQL as exactly one `writer_v1` and one `attestor_v1`. Role-indexed simulated SQL-RLS checks succeeded; they are **not** equivalent to a true logged-in HTTP Storage test.

## Security boundary

- **Never** send account passwords, authenticated JWTs, attestor signing keys, or an API secret to chat.
- Do not put credentials in GitHub issues, PRs, logs, Actions `workflow_dispatch` fields or source-controlled .env files.
- Only run in a private local operator terminal with Git checkout of this review branch. Do not deploy the launcher as a public endpoint.
- The client must use a **publishable** project key, never a service-role/secret key, or the Storage RLS checks are invalidated.
- Password sign-in is performed with Supabase Auth over HTTPS. The script validates `getUser()` server-side, checks the correct role in `app_metadata`, distinct user IDs and fresh sessions. Signing material is ephemeral test-only Ed25519.
- The script **refuses to run** if there is no explicit nonproduction opt-in, the URL does not match the test ref, test and production refs are equal, the emails match, or the key is not a publishable key.
- This run creates one **synthetic PDF**, stored under a random preadmission path in the shadow bucket for audit, and does not delete it automatically.

## Operator instructions

From a local, trusted terminal in `indice_nexus` with its Node dependencies installed and this PR branch checked out, set the **nonsecret** values appropriate to the two Supabase projects:

```bash
export PREFLIGHT_NONPROD_APPROVED=YES
export SUPABASE_TEST_REF="<your-nonproduction-project-ref>"
export SUPABASE_PRODUCTION_REF="<your-separate-production-project-ref>"
export SUPABASE_TEST_URL="https://${SUPABASE_TEST_REF}.supabase.co"
export SUPABASE_TEST_PUBLISHABLE_KEY="<nonproduction-sb_publishable_key>"
```

Input existing *test-account credentials only in the trusted terminal*. Avoid entering passwords in shell commands, terminal history, or copy/pasting them into ChatGPT:

```bash
read -r -p "Writer email: " SHADOW_WRITER_EMAIL; export SHADOW_WRITER_EMAIL
read -r -s -p "Writer password: " SHADOW_WRITER_PASSWORD; echo; export SHADOW_WRITER_PASSWORD
read -r -p "Attestor email: " SHADOW_ATTESTOR_EMAIL; export SHADOW_ATTESTOR_EMAIL
read -r -s -p "Attestor password: " SHADOW_ATTESTOR_PASSWORD; echo; export SHADOW_ATTESTOR_PASSWORD
npx tsx scripts/orotitan-equity/shadow-authenticated-smoke-login.ts
unset SHADOW_WRITER_EMAIL SHADOW_WRITER_PASSWORD SHADOW_ATTESTOR_EMAIL SHADOW_ATTESTOR_PASSWORD
```

The launcher signs in both users, validates the roles, then calls the existing smoke. The smoke must verify:
- writer can create a private object with no overwrite;
- writer cannot GET the object;
- attestor cannot INSERT an object;
- attestor can GET the original bytes;
- SHA-256 and size match;
- Ed25519 signature verifies with an ephemeral test-only public key;
- Registry authority is explicitly **false**.

A successful smoke only proves effective Storage HTTP controls for the test principals at that moment; it does **not** prove the signing key lives in a separate security domain or that the object is historically identical to the pre-cutoff published source. The trusted Registry writer and six production persistence-function changes are not part of this PR.

If passwords were never established in the Auth Dashboard, set each test user's password through the **official Supabase Auth Admin interface** first. Do not manipulate password hash rows with raw SQL.

## Current environment results

Shadow DB checks (2026-10-27) succeeded for a real assigned role's **simulated** `auth.jwt()` claims: Writer bucket SELECT 1; Writer object GET 0; Attestor bucket SELECT 1; Attestor object GET 1; Attestor object LIST 0. Metadata probes used synthetically inserted storage.objects rows in transactions followed by ROLLBACK. No byte-content original was sent over the Storage HTTP API from the assistant environment.

## Nonproduction test verdict

`SHADOW_RLS_DATABASE_SIMULATION = PASS`

`TWO_ACCOUNT_AUTH_HTTP_STORAGE_SMOKE = NOT_RUN_NO_SESSIONS_IN_CONNECTOR`

`REGISTRY_ATTESTED_SOURCE_ADMISSION = NO`

`PRODUCTION_ASML_RUN_MODIFIED = NO`
