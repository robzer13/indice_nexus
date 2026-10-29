# OroTitan V3 — AWS independent attestor / immutable archive architecture decision (PROPOSED, NOT DEPLOYED)

Date: 2026-10-29. Scope: SHADOW synthetic-only. Status: **architecture proposal / operator must create separate AWS account and complete acceptance tests**. No Vercel/production/Registry modifications authorized.

## Decision

Choose a **separate AWS account** (initial account name suggested `orotitan-attestation-shadow`) rather than another Vercel project in the existing team. Region preference `eu-west-3` (Paris), subject to console-verified availability of every needed key spec/service and a price check.

Keep current Next.js/Vercel research app, **Writer** and Supabase SHADOW where they already are. Build separate **Attestor** capability in AWS using:
1. Lambda Node.js 22, IAM role with least privilege, private runtime (no public anonymous invoke).
2. AWS KMS asymmetric `ECC_NIST_EDWARDS25519` key, `SIGN_VERIFY`, `ED25519_SHA_512`, `MessageType:RAW`. Private key stays inside KMS; a **new public key** is exported once and pinned by Gate out-of-band. Do not reuse the operator-generated local testing key.
3. Fresh short-lived Supabase Attestor Auth token, obtained exclusively by Lambda from a secret stored in AWS Secrets Manager. **Never use `service_role`**, never give this token to Writer, never log access tokens or passwords.
4. Attestor performs server Auth role validation, downloads original **exact bytes from private SHADOW Storage**, verifies 715 byte count, SHA-256 and content-addressed URI, signs the existing canonical length-prefixed `SignedRereadStatement` bytes with KMS. Gate verifies 19-field wire format and pinned public key using the existing exact Ed25519 verifier after KMS-vs-Node interoperability testing.
5. Separate S3 bucket with Object Lock and versioning for signed receipt, source byte snapshot, Gate audit head and decision evidence. **First test Governance mode**; only adopt **Compliance mode** after explicitly testing and approving retention duration, backup/restoration, billing and operator recovery. Record bucket, object key, immutable **version ID**, retain-until timestamp, policy proof and reread hash.
6. Enable AWS CloudTrail and cost monitoring/alerts; treat audit logs, infrastructure and billing as independent review controls, not merely code tests.

## Security and authority boundaries

- Same human owning **both** cloud account roots does **NOT prove independent administrative custody**, even though service-level identities and cryptographic key isolation can be demonstrated. External governance certification needs an independently authorized custodian, root recovery / MFA audit, role and secret-access review.
- Existing local Attestor Ed25519 key fingerprint `5288aa9bf268dc32feff913810d21ffb3ff4af7d3efc3ca65c7d64713604d096` was for the **previous synthetic smoke**; do not import that private key into AWS KMS or claim it has been transferred. AWS KMS must issue a new key ID and out-of-band public-key pin, with formal key rotation to a separate trust version.
- Current detached signer invokes local `node:crypto.sign`; it must NOT be presented as AWS-ready without a new pluggable signer adapter and integration tests against KMS `ED25519_SHA_512` RAW semantics.
- Avoid public Lambda URLs and open API Gateway authorization. Gate request authentication via IAM/OIDC and replay-protected challenge/correlation ID. Narrow external network egress where feasible.
- Idempotency: same challenge may never be recorded twice. Gate ledger already rejects replay locally; AWS evidence archive must add conditional create and independently inspectable immutable object version IDs.
- For Object Lock: protected object **versions** remain locked; a delete marker may still hide latest object without destroying locked version. Therefore archived evidence always resolves by **version ID**, never simply by latest key path.
- Compliance mode retention cannot be shortened and cannot be removed by the account root. Configure deliberately; test Governance first.
- Historical issuer publication date and cutoff are NOT certified by storage bytes/hash alone. A reproducible 2026-09-21 cutoff-source evidentiary chain remains an independent gate.
- The testing account must never receive production Supabase credentials or perform Registry writes; do not merge existing PRs #433/#434/#435 automatically.

## Minimal acceptance stages (fail closed)

- G1. Account created and controlled separately from current Vercel/Supabase environment; root MFA/passkey, billing alerts, IAM role inventory documented. **PENDING**.
- G2. KMS Ed25519 key created, `GetPublicKey` out-of-band verified/pinned; prove Sign is allowed only to Lambda role and private key is nonexportable; deny wrong principal. **PENDING**.
- G3. Lambda only uses dedicated `attestor_v1` Auth identity in SHADOW, cannot upload/delete/list, re-reads exact synthetic 715 bytes, canonical signed receipt verifies offline under fresh nonce; must reject Writer and compromised path. **PENDING**.
- G4. Store source bytes + signed receipt + Gate audit head in Object Lock Governance test, reread via immutable S3 **version ID**, SHA-256 check and rejected unauthorized deletion. **PENDING**.
- G5. Explicit independent storage/retention decision: Compliance-mode test object with approved bounded policy; verify retention and recovery, capture audit and price evidence. **PENDING**.
- G6. Independent administrative review with custody claims attested by separate authorized person/service account. **PENDING**.
- G7. Canonical Registry design/migration and issuance of `PERSISTENCE_ATTESTED`, only after the governing contracts, source provenance, publication-date evidence and actor authorization requirements are satisfied. **BLOCKED**.

## Operator first action

Create a **standalone AWS account** through the official AWS signup UI. Do not create any billable cloud infrastructure until IAM root MFA and billing budget alerts are enabled and account region/plan capabilities checked. Set budget alerts at proposed $5 and $10 thresholds; alerts are **not a hard spend cap**.

Share ONLY a non-secret status such as `AWS_ACCOUNT_CREATED / ROOT_MFA_ENABLED / BUDGET_ENABLED / REGION=eu-west-3`. Never share passwords, recovery codes, root credentials, payment details, AWS access keys, client secrets, or Supabase JWTs.

## Official vendor sources (checked 2026-10-29)

- https://docs.aws.amazon.com/kms/latest/developerguide/symm-asymm-choose-key-spec.html
- https://docs.aws.amazon.com/kms/latest/APIReference/API_Sign.html
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock.html
- https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-lock-configure.html
- https://aws.amazon.com/kms/pricing/
- https://aws.amazon.com/lambda/pricing/
- https://aws.amazon.com/s3/pricing/
- https://aws.amazon.com/aws-cost-management/aws-budgets/faqs/
