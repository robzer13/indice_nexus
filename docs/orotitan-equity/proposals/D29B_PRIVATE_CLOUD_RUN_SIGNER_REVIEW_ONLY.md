# D29b — Private Cloud Run signing service (REVIEW ONLY, NOT DEPLOYED)

**Status:** DRAFT PROPOSAL. This branch implements a bounded standalone Node.js HTTP signer, tests and container blueprint, but **does not activate Cloud Run, create a service, alter IAM, provision credentials, spend Google Cloud resources or enable D29b**.

## Verified user-provisioned resources (2026-10-22)

- Google Cloud project: `orotitan-d29b-signing`, billed to the existing account. The $300 free-trial credit is **expired**, not usable.
- Monthly **€1 alert budget** scoped to this project. This is NOT a hard spending ceiling.
- API Cloud KMS enabled; key ring `d29b-signing` in `europe-west1`.
- One SOFTWARE Ed25519 asymmetric signing key `receipt-sign`, cryptoKeyVersion **1**, active.
- Google Cloud service account created: `orotitan-d29b-signer@orotitan-d29b-signing.iam.gserviceaccount.com`; user reports successful grant of `roles/cloudkms.signer` **on that key only**. Live IAM binding has not been re-audited programmatically.
- Public Ed25519 SPKI PEM downloaded by the operator; the private key is held by KMS and must not be exported.

## Separate deployment and security identities

**Key principle:** Cloud Run's service identity (`orotitan-d29b-signer`) and its **invoker** are NOT the same authority:

| Identity | Permissions | Location |
|---|---|---|
| Cloud Run runtime `orotitan-d29b-signer@...` | `cloudkms.cryptoKeyVersions.useToSign` via `roles/cloudkms.signer` on only `receipt-sign` | Signing service only; never OAuth pilot |
| Privileged D2 issuer identity (separately provisioned) | `roles/run.invoker` only on the one signer Cloud Run service | Protected issuer runtime; **no Cloud KMS signer** permission |
| ChatGPT/OAuth/MCP pilot | **NONE** on signer service, KMS key, or D2 private issuer | Public OAuth gateway — must never receive signer invocation ability |
| Google Cloud human/deployer | Minimal deploy privileges only for approved rollout | Not the signing runtime credential |

A Vercel-hosted issuer must obtain a short-lived Google Cloud identity via **Vercel OIDC → GCP Workload Identity Federation** with constraints matching **the exact protected project, deployment environment, and trusted issuer**. A public OAuth-pilot Vercel project must never become a WIF principal with `roles/run.invoker`. Do not copy user credentials or service-account JSON keys.

Cloud Run IAM authentication is **mandatory** (no public `allUsers` invoker). The application intentionally does NOT treat any HTTP request header as proof of authorization: Google Cloud Run verifies IAM tokens at the edge. A publicly deployed service with disabled IAM checks would be an immediate security failure and must be rolled back.

## Source

- `policy.mjs`: exact run, subject, OAuth client, resource, contract pack digest, phase, lifecycle/gate, prompt hash and age admission. Bounds incoming body and requires canonical attestation JSON. Cloud Run does NOT re-fetch immutable evidence: that task stays exclusively in D2.
- `kms.mjs`: Google metadata server ADC token from the **attached service account**, request to a fixed `cryptoKeyVersions/1:asymmetricSign`, raw Ed25519 domain-separated bytes, CRC32C in/out, response exact key-version/name and SOFTWARE protection-level checks, and independent signature verification against the pinned Ed25519 public key. No redirect; timeouts; bounded JSON; no retry storm.
- `server.mjs`: only `POST /v1/sign-d2-handoff` and `GET /healthz`. Health does not grant signature. Closed generic error responses; no request/receipt/tokens/secrets logged. Cloud Run IAM remains the primary authorization barrier.
- `Dockerfile`: Node 22, nonroot, no dependencies, keys or credentials.
- `server.test.mjs`: synthetic Ed25519 fixtures, fake metadata/KMS services and local HTTP negative matrix; no live cloud access or billing.

## Preconditions for a **separate** Cloud Run deployment authorization

1. **Independent security review:** confirm this signer will only be invoked by a trusted D2 issuer after real immutable-proof verification and cannot be reached by the ChatGPT OAuth gateway. Approve identity separation and Vercel OIDC workload federation claims.
2. **Billable rollout approval:** specify the exact Cloud project, region, service name, image digest, service account and maximum expected spend; review running KMS costs, Cloud Run usage, Cloud Build/Artifact Registry/logging costs, and the €1 alert budget. The budget does not block spending.
3. **Immutable configuration:** exact `D29B_RUN_ID` (Besi only), `D29B_SUBJECT_ID`, `D29B_OAUTH_CLIENT_ID`, `D29B_RESOURCE`, trusted `D29B_CONTRACT_SET_SHA256`, `D29B_KEY_ID`, `D29B_KMS_KEY_VERSION` and `D29B_PUBLIC_KEY_SPKI_PEM`. Use the actual verified PEM from Google KMS, never copy/construct key data from screenshots. `D29B_KMS_KEY_VERSION` must exactly match:
   `projects/orotitan-d29b-signing/locations/europe-west1/keyRings/d29b-signing/cryptoKeys/receipt-sign/cryptoKeyVersions/1`.
4. **Runtime policy:** use Cloud Run request-based billing, `min-instances=0`, `max-instances=1`, `concurrency=1`, CPU 1, memory 256 MiB, timeout 15s, region `europe-west1`, no anonymous invocation. `max-instances` is a risk control, **not** a guaranteed spending cap. Disable public ingress if the trusted D2 issuer has properly configured internal Google Cloud networking; if D2 is external, IAM-protected public endpoint ingress is required, with additional Vercel OIDC controls.
5. **IAM binding:** set `roles/run.invoker` on this Cloud Run **service only** for the independently approved D2 issuer identity. Never grant it to `allUsers`, `allAuthenticatedUsers` or the ChatGPT pilot. Attach `orotitan-d29b-signer` as runtime identity; no `GOOGLE_APPLICATION_CREDENTIALS`, static tokens or JSON keys.
6. **Canary:** test both unauthenticated invocation (must return 401/403 *at Google edge*) and appropriately authenticated restricted issuer invocation. Verify only synthetically admitted proofs are signable and wrong run/client/resource/contract/expired manifests are denied. Confirm no logs leak prompt or token contents and Cloud KMS's signature matches the pinned key.
7. **Registry/RLS:** independently authorize and validate the already-drafted D29b receipt and writer SQL. The Cloud Run signer never writes to Supabase. No stage changes or publication.

**STOP** if any prerequisite cannot be verified. There is no deployment command executed by this proposal. Existing D29a continues operating with `prompt:null` for blocked Besi RESEARCH.

## Price and provider sources

- Cloud Run request-based pricing / free tier: https://cloud.google.com/run/pricing
- Cloud Run service identity / ADC: https://docs.cloud.google.com/run/docs/securing/service-identity
- IAM-authenticated interservice invocation: https://docs.cloud.google.com/run/docs/authenticating/service-to-service
- Cloud Run ingress distinctions: https://docs.cloud.google.com/run/docs/securing/ingress
- Vercel OIDC → Google Cloud WIF: https://vercel.com/docs/oidc/gcp
- Cloud KMS asymmetricSign / CRC: https://docs.cloud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cryptoKeys.cryptoKeyVersions/asymmetricSign
