# D29b — Google Cloud KMS Ed25519 + privileged PostgreSQL adapter

**Review-only. No production deployment, external key, IAM principal, database connection or Supabase migration was created.**

## Concrete vendor candidate

The existing D29b proof format requires **Ed25519** (not ECDSA). Google's Cloud KMS `EC_SIGN_ED25519` supports raw PureEdDSA data under the `SOFTWARE` protection level, **not HSM**. This is *managed Cloud KMS with nonexported key custody*; it is NOT a Cloud HSM guarantee. If an HSM-backed private key is a hard requirement, an independently approved change to signing algorithm/proof schema/verifier is necessary; do not silently substitute ECDSA.

Source:
- https://docs.cloud.google.com/kms/docs/algorithms
- https://docs.cloud.google.com/kms/docs/reference/rest/v1/projects.locations.keyRings.cryptoKeys.cryptoKeyVersions/asymmetricSign

`mcp-d29b-gcp-kms-signer.ts` implements an **injectable** Cloud KMS REST adapter:

1. A pinned full `projects/.../cryptoKeyVersions/N` name and `keyId`, independent pinned Ed25519 SPKI public key and a short-lived token supplier, all supplied **only by a privileged admin runtime**.
2. `POST /v1/{keyVersion}:asymmetricSign` with exact **raw bytes** under `data`, and `dataCrc32c`.
3. Demand `verifiedDataCrc32c = true`, exact version `name`, `protectionLevel = SOFTWARE`, signature length = 64, matching `signatureCrc32c`, and local Ed25519 verification against pinned SPKI key. Any mismatch fails closed.
4. The existing independent D2 issuer **again** verifies returned signature using its own pinned public key and then rechecks Registry CAS after signing. Neither the public MCP runtime nor Postgres stores a private key.

**IAM is not provisioned.** An authorized future runtime should use Workload Identity Federation / service account impersonation with only `cloudkms.cryptoKeyVersions.useToSign` on one key version. Avoid downloadable long-lived JSON service-account keys. Configure key rotation explicitly by version and published verifier `keyId`. Never put access tokens or signer credentials in GitHub or the public MCP project.

## Privileged Postgres SQL adapter

`mcp-d29b-pg-writer-adapter.ts` wraps an externally supplied **private PostgreSQL connection pool**. Each call begins a PostgreSQL transaction and executes `SET LOCAL ROLE orotitan_d29b_writer` (requires separately authorized role membership). It:

- Reads Besi Registry/stage state inside one read-only repeatable-read snapshot.
- Calls only the approved SQL private writer procedure in `SERIALIZABLE` isolation with parameterized arguments and expected run/stage versions. Never retries a stale short-lived signature.
- Reads back exactly one receipt; revokes through its dedicated private procedure in a new SERIALIZABLE transaction.
- Always releases connections, rolls back failed transactions and never grants OAuth, `anon`, `authenticated` or Vercel MCP permission to assume the private role.
- Injected DB pool must come from a privileged runtime, not Supabase `service_role`. No driver/DSN or credentials have been configured; no SDK installation or lockfile change was needed.

The private writer SQL proposal was strengthened to **audit every insert and update using an AFTER ROW SECURITY INVOKER trigger** in the same transaction, even if a private-role session bypasses the helper stored procedures. Only hashes and state versions enter the append-only audit table; no prompt bytes are written into that audit.

The code uses safe mock transports in unit tests and the existing disposable PG17 tests for the SQL contract. **This is not evidence of actual Google Cloud IAM/KMS or Supabase live availability.**

## Current hard deployment gates

- A human chooses Cloud KMS SOFTWARE Ed25519 vs a separately designed HSM signature protocol. No provider account or key has been created.
- Independent privileged runtime; IAM and auth; long-lived credentials forbidden in the public MCP.
- Actual admin-only pool connection with DB membership and audit guarantees; schema/RLS migration approval, rollback and incident plan.
- Staging end-to-end with short-lived synthetic D2 attestation and full negative authorization/expiry/revocation matrix.
- Re-evaluate multi-read CAS/TOCTOU before claiming prompt authority. A short-lived proof with live CAS check does not freeze the Registry against a later change.
- Besi remains blocked in RESEARCH until the six evidence deficiencies are resolved. An OAuth tool must never bypass it.

No feature flag is enabled and no release is authorized by this review PR.
