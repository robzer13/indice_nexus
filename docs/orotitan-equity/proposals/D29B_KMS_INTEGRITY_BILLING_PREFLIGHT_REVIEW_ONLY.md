# D29b KMS integrity and cost preflight (review only)

**Status:** source change and tests only. No cloud resource, paid billing, key,
principal or IAM binding created.

## Official Google references (verified 2026-10-22)

- Software KMS pricing: https://cloud.google.com/kms/pricing
- EC_SIGN_ED25519 / PureEdDSA signs **raw data**:
  https://docs.cloud.google.com/kms/docs/algorithms
- REST asymmetricSign shape (data/dataCrc32c, signature/signatureCrc32c,
  verifiedDataCrc32c, exact name, protectionLevel):
  https://docs.cloud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cryptoKeys.cryptoKeyVersions/asymmetricSign
- Trial eligibility: https://cloud.google.com/signup-faqs
- Billing budget alerts **are not a spending ceiling**:
  https://docs.cloud.google.com/billing/docs/how-to/budgets
- Spend cap budgets, where available:
  https://docs.cloud.google.com/billing/docs/how-to/budgets-spend-caps

## Cost and account gate

User has **Google AI Plus**, not Google AI Pro. Do not assume recurring Google
Cloud credits from this subscription. Eligibility for a new-user Google Cloud
$300 / 90-day trial must be verified on the actual Google Cloud billing
account; do not claim a trial is currently active.

A single manually created SOFTWARE key version costs approximately
**USD 0.06/month**, plus **USD 0.03 per 10,000 cryptographic operations** at
published list prices. Autokey-specific always-free benefits do not
automatically apply to this manually provisioned signing key. The separate
signing workload/runtime, requests, logs, data egress and other products may
have distinct charges.

**No billing changes without separate authorization**. In the chosen Cloud
project, verify the active billing mode, location and project, inspect current
trial and credits, set a project-scoped budget or supported spend cap, IAM
limits and quota protections *before* enabling API, creating a key or binding
a principal. Alerts alone do not reliably enforce a zero-dollar ceiling.

## D29b KMS adapter hardening

The adapter now:
- submits a `dataCrc32c` Castagnoli checksum of **raw Ed25519 message
  bytes**, without prehashing;
- requires `verifiedDataCrc32c===true` in the response;
- requires the *exact expected key-version name* and SOFTWARE protection;
- validates signature as canonical 64-byte Ed25519 base64;
- requires a matching `signatureCrc32c` over returned signature bytes;
- passes an 8-second AbortSignal to the transport and fails closed;
- never retries inside the adapter, discloses a private key, selects a remote
  endpoint dynamically, or accepts unpinned response public keys.

Tests use synthetic local Ed25519 private keys only for fixtures, never
cloud credentials. They also verify the standard CRC32C test vector
(`123456789` -> `0xe3069283`).

## Missing release gates

This remains a **candidate** Cloud KMS transport only. Before real signing:
1. Verify the live Google Cloud project, billing, trial status, restrictions
   and explicit user approval for any billable provisioning.
2. Implement the authenticated workload identity and HTTP transport with
   strict no-redirect, bounded response size/timeout and restricted egress.
3. Bind `cryptoKeyVersions.useToSign` narrowly to the isolated signer
   identity and one key version, pin SPKI externally, audit every signature.
4. Perform actual integration against a designated *staging* Ed25519 key
   with cost and operation limit; test resource, CRC and signature response.
5. Complete independent security review for the privileged writer/receipt
   migrations and negative-RLS end-to-end tests, then obtain a distinct
   deployment approval. The live Besi RESEARCH block remains intact.
