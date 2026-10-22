# D29b — Google Cloud KMS signer integration: review only

**No cloud key was created. No IAM binding, token, Supabase permission, Vercel setting or deployment was changed.**

## Candidate provider and algorithm

Use a **dedicated Google Cloud KMS** key version with purpose `ASYMMETRIC_SIGN`, algorithm `EC_SIGN_ED25519`, and region/protection tier reviewed by the operator. Google Cloud KMS documents PureEdDSA Ed25519 signing of raw data using the `asymmetricSign` endpoint.

- Documentation: https://docs.cloud.google.com/kms/docs/algorithms
- REST: https://docs.cloud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cryptoKeys.cryptoKeyVersions/asymmetricSign

This is a candidate implementation, not an authorized cloud provisioning decision. The existing D29b protocol requires Ed25519. Other providers require explicit review for compatibility.

## Security boundary

`mcp-d29b-google-kms-ed25519-v1.ts` constructs a fixed HTTPS Google API URL from a strictly parsed **key-version** resource. It uses `data` (raw bytes), not `digest`, because `EC_SIGN_ED25519` signs raw messages. Access tokens come from an **injected trusted workload-identity provider**; no personal token, service-account JSON key, OAuth client secret or private key is embedded.

Google Cloud IAM must grant only the separate privileged signer workload `cloudkms.cryptoKeyVersions.useToSign` on the exact key, not the entire project. A distinct security operator must retrieve the corresponding Ed25519 public key and pin its SPKI + `keyId` in the trusted D2 issuer and the read-only MCP verifier. The D2 issuer also independently verifies the KMS signature before storing a receipt.

The pilot MCP runtime must **never possess** the workload identity used for KMS signing. The signed receipt and pinned public keys are safe to read there only with the reviewed RLS restriction.

## Required before activation

1. Decide on the GCP project, region, key ring, nonexportable key version and cost/protection tier. Do not provision automatically.
2. Provision isolated privileged runtime identity and narrowly scoped IAM role, ideally via workload identity federation with short-lived credentials and no JSON key.
3. Audit KMS requests and restrict network egress. Add request/response integrity checks (Google documents CRC32C), bounded timeouts, and retry policy; the review-only adapter does **not yet implement CRC32C** and must be hardened before production.
4. Implement the separately authenticated server-side signer invocation and the admin database transaction adapter, with immutable audit/logging and receipt revocation, and test against a staging key.
5. Review the SQL/role/RLS proposals independently. No existing public ChatGPT endpoint gains writing or signing permissions.
6. Run adversarial end-to-end tests using a **synthetic admitted D2 fixture**. Confirm the present Besi Research blockers still prevent issuing a handoff.
7. Obtain explicit, separate user approval for any cloud provisioning, production database migration or Vercel deployment.

The draft PR contains only the **transport adapter**, mocks, and this documentation. It is not a connected production signer. Tests cover the fixed endpoint, raw-data signing and fail-closed behavior.

## Engineering limitations

The adapter accepts an injected HTTP transport for unit testing and future hardened network implementation. The runtime integration must enforce HTTPS, redirect rejection, endpoint allowlist, bounded response sizes, timeouts, non-leaking logs and cloud-issued workload credentials. It must verify CRC32C response integrity and the pinned key identity in addition to the Ed25519 verification already enforced by the privileged D2 issuer. No real Besi attestation has been signed.
