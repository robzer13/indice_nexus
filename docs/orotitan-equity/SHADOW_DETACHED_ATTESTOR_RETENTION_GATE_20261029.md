# OroTitan V3 — Detached Attestor & retention control (SHADOW only)
Status: **IMPLEMENTED FOR REVIEW / NOT OPERATED / NOT REGISTRY AUTHORITATIVE**. Date: 2026-10-29.

## 1. Scope and authority

This is a STACKED nonproduction design on top of #433 and #434. It does not modify Registry writers, production secrets, company research, or any frozen contract. The earlier fully authenticated single-process synthetic HTTP smoke is **PASS**, but the separation-of-key-custody evidence is **NOT YET PROVEN**. The new tools deliberately refuse any real issuer document: the only permitted source is the known 715-byte synthetic PDF with SHA-256 `f9cda65b308c24e62fd9a5070df43202c2c0d4560a8aba8f6e697a8a9f467b9b`.

The frozen Run / Artifact Registry Manifest contract requires **exact bytes SHA-256**, stable private storage coordinates, historical availability states, artifact-first/registry-second writes, immutable registered bytes, and fail-closed downstream resolution on missing/withdrawn bytes. Its target storage supports private Supabase Storage, but it does not permit an ordinary Storage RLS policy to be described as WORM or as independent signer-custody proof.

## 2. Explicit security principals

- **Writer host:** holds only Writer Auth credentials and creates the synthetic object. It MUST NOT possess the Attestor login, private Ed25519 key, or verifier's trusted-pubkey configuration.
- **Gate/verifier host:** knows the existing content-addressed object URI; generates a fresh challenge/nonce and trusted handoff; owns a public key fingerprint and public-key file independently approved *before* receiving the signature. It MUST NOT hold the private signing key or a Writer/Attestor password.
- **Attestor host:** possesses **only** Attestor Auth credentials, a locally provisioned persistent Ed25519 PKCS#8 private key, and the received handoff. It must be a different operational principal and host/secret store from Writer. No Writer JWT/password, no `service_role` or `sb_secret_`, and no process-generated test keypair.
- **Registry:** NEVER called by these utilities. All outputs remain `NONAUTHORITATIVE`.

The implementation checks only what software can observe: credentials in the current process, a server-verified Attestor user role, exact bytes via Storage, Ed25519 signatures and pinned Gate trust inputs. Distinct hosts, administrators, key-manager/IAM policies, audit trails, signer private-key exclusivity and independent human ownership require **separately reviewed deployment evidence**.

## 3. Nonprod-only files

- `scripts/orotitan-equity/shadow-detached-challenge.ts`: Gate-side offline handoff and nonce, exact synthetic source locator.
- `scripts/orotitan-equity/shadow-detached-attestor-login.ts`: Attestor-only password login, server Auth verification, no Writer context.
- `scripts/orotitan-equity/shadow-detached-attestor.ts`: Attestor-only authenticated reread (SHA-256) and Ed25519 signed receipt written create-only to an operator-selected local file.
- `scripts/orotitan-equity/shadow-detached-verify.ts`: Gate-side offline verification with public key and nonce PINNED OUTSIDE the receipt.
- `lib/orotitan-equity/v3/shadow-detached-attestor-boundary.ts`: fail-closed project/ref, synthetic bytes, source locator, role and secret-separation guards.

### Provisioning prerequisites

Pin and review the Git commit of this PR before local execution. Use Node.js 22.x, `npm install --ignore-scripts --no-audit --no-fund` (repository intentionally has no package-lock), `npm run typecheck`, and the local `npx --no-install tsx` runtime. Do not run the scripts on production or place credentials, PEMs, receipts, or source bytes in public GitHub.

Use **two distinct controlled hosts/accounts** for Gate and Attestor, and keep the existing Writer account on a third isolated operator context for a meaningful independence demonstration. On the Attestor host create an Ed25519 PKCS#8 keypair under an operator-controlled secret directory, using an approved key-management mechanism or a protected local implementation. Example Node 22 offline generation (run ON THE ATTESTOR HOST only):

```powershell
node -e "const c=require('node:crypto'),f=require('node:fs');const p=c.generateKeyPairSync('ed25519');f.writeFileSync(process.env.SHADOW_DETACHED_PRIVATE_KEY_FILE,p.privateKey.export({type:'pkcs8',format:'pem'}),{flag:'wx',mode:0o600});f.writeFileSync(process.env.SHADOW_DETACHED_PUBLIC_EXPORT_FILE,p.publicKey.export({type:'spki',format:'pem'}),{flag:'wx',mode:0o600})"
```

**Windows caveat:** `mode:0o600` is NOT an adequate Windows ACL audit. Restrict and independently inspect the filesystem ACL before use. Better use an approved secret manager/HSM when a real independent service is deployed. Transfer ONLY the public key to the Gate via a separately authenticated channel. Gate pins the exact public-PEM SHA-256 fingerprint out of band and records Attestor/key identity. The offline verifier REQUIRES `SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256` and recomputes the SHA-256 of the received public PEM to prevent a key-swap. Compute it locally on Gate as `(Get-FileHash -Algorithm SHA256 $env:SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE).Hash.ToLowerInvariant()`, then compare it to the separately authenticated expected digest before pinning. Never transfer the private key.

### Gate: create a fresh challenge

Set the following variables *on the Gate host*. `SHADOW_GATE_EXISTING_SYNTHETIC_URI` must be an already-uploaded synthetic URI from the passing smoke. Use an absolute file path outside Git checkout; output must not already exist.

```powershell
$env:PREFLIGHT_NONPROD_APPROVED = "YES"
$env:SUPABASE_TEST_REF = "awgsurdyvsyolcgpnygh"
$env:SUPABASE_PRODUCTION_REF = "cugpgtzygqqlxetyeven"
$env:SHADOW_GATE_EXISTING_SYNTHETIC_URI = "<paste-existing-synthetic-supabase-uri>"
$env:SHADOW_GATE_HANDOFF_FILE = "C:\\secure-gate\\handoff.json"
npx --no-install tsx scripts/orotitan-equity/shadow-detached-challenge.ts
```

This creates a distinct nonce for every inspection and validates the whole synthetic-only locator. Save the exact generated `challengeNonce` on the Gate **before** transfer. Transfer `handoff.json` to Attestor. A handoff is an untrusted request, not a trust anchor.

### Attestor: authenticate, reread, sign

On the **separately controlled Attestor host**, set only these fields. Keep PEM outside the repository and access restricted. Enter the password by a local secure prompt, not on the command line:

```powershell
$env:PREFLIGHT_NONPROD_APPROVED = "YES"
$env:SUPABASE_TEST_REF = "awgsurdyvsyolcgpnygh"
$env:SUPABASE_PRODUCTION_REF = "cugpgtzygqqlxetyeven"
$env:SUPABASE_TEST_URL = "https://awgsurdyvsyolcgpnygh.supabase.co"
$env:SUPABASE_TEST_PUBLISHABLE_KEY = "<shadow-sb_publishable-key>"
$env:SHADOW_DETACHED_HANDOFF_FILE = "C:\\attestor-private\\handoff.json"
$env:SHADOW_DETACHED_RECEIPT_FILE = "C:\\attestor-private\\receipt.json"
$env:SHADOW_DETACHED_PRIVATE_KEY_FILE = "C:\\attestor-private\\signer-private.pem"
$env:SHADOW_DETACHED_ATTESTOR_ID = "independent-shadow-attestor"
$env:SHADOW_DETACHED_KEY_ID = "independent-shadow-key-1"
$env:SHADOW_DETACHED_ATTESTOR_EMAIL = "<attestor-account-email>"
$pw = Read-Host -AsSecureString "Attestor password"
$env:SHADOW_DETACHED_ATTESTOR_PASSWORD = [System.Net.NetworkCredential]::new("", $pw).Password
npx --no-install tsx scripts/orotitan-equity/shadow-detached-attestor-login.ts
Remove-Item Env:SHADOW_DETACHED_ATTESTOR_PASSWORD -ErrorAction SilentlyContinue
```

The password remains a process-environment secret only until deleted, but OS-level process inspection / memory capture remain risks; prefer service-level secret injection for production deployment. No JWT/private PEM should be pasted into chat or stored in GitHub. The signed `receipt.json` is created with exclusive create semantics and may be transferred to Gate. The signed statement remains `TRANSPORT_ONLY_NOT_REGISTRY_ADMITTED`.

### Gate: offline verify

Gate must independently pin `SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE` and the nonce it generated. Do NOT read any key or nonce out of the received receipt to establish trust.

```powershell
$env:PREFLIGHT_NONPROD_APPROVED = "YES"
$env:SUPABASE_TEST_REF = "awgsurdyvsyolcgpnygh"
$env:SUPABASE_PRODUCTION_REF = "cugpgtzygqqlxetyeven"
$env:SHADOW_GATE_HANDOFF_FILE = "C:\\secure-gate\\handoff.json"
$env:SHADOW_GATE_RECEIPT_FILE = "C:\\secure-gate\\receipt.json"
$env:SHADOW_GATE_TRUSTED_PUBLIC_PEM_FILE = "C:\\secure-gate\\pinned-signer-public.pem"
$env:SHADOW_GATE_TRUSTED_PUBLIC_PEM_SHA256 = "<sha256-of-approved-public-pem-as-lowercase-hex>"
$env:SHADOW_GATE_PINNED_CHALLENGE_NONCE = "<the-gate-generated-nonce>"
$env:SHADOW_GATE_TRUSTED_ATTESTOR_ID = "independent-shadow-attestor"
$env:SHADOW_GATE_TRUSTED_KEY_ID = "independent-shadow-key-1"
npx --no-install tsx scripts/orotitan-equity/shadow-detached-verify.ts
```

The verifier rejects altered metadata/hash, an unexpected key/attestor ID, stale timestamp, wrong nonce, public-key PEM bytes that do not match the independently pinned SHA-256, altered signature and any attempt to flip certification flags. A passing signature **does not** cause automatic release or Registry admission. The 10-minute signature freshness check applies at time of Gate verification. There is no persistent nonce-consumption database yet; replay tracking beyond that window is deployment work.

## 4. Retention & immutability: hard boundaries

Storage RLS with create-only Writer INSERT and Attestor GET/GET_INFO proves **normal authenticated operation segregation**. It does **NOT** guarantee:
- immutability against project owners, admin/service-role, bucket deletion or provider operators;
- WORM / Object Lock / legal hold;
- a recovery replica independent of the primary Storage account;
- that a pre-cutoff publisher truly served the same bytes at the historic cutoff;
- availability beyond a future cleanup, withdrawal or provider outage.

**Proposed retention evidence chain (not yet deployed):**

1. Gate creates T0 challenge and archives resulting exact-byte proof + signed receipt and public-key fingerprint in an independently controlled immutable audit archive.
2. Gate repeats with a *new* nonce at T+1/T+n via independent Attestor, checks same URI, size, content SHA-256; stores new signed read audit. This proves sampled historical availability/integrity, not continuous immutability.
3. Provide separate noneditable backup replica in independently administered storage with a reviewed retention policy / actual WORM or Object Lock guarantees, versioning and timed deletion tests. Record provider, bucket/object version, retention-until, policy, operator evidence, restore/re-read hash.
4. Independent reviewer validates ACLs, managed key identity, secret access logs, control-plane deletion / restore tests, legal withdrawal, and incident recovery.
5. Only after original historical provenance and governed Registry migration are separately implemented can a future authority gate consider the evidence. Do not update existing V1 contract language to fake compatibility.

**Current gate matrix:**

| Property | State |
|---|---|
| Two Auth principals, writer create, reader GET | PASS, actual HTTP on 2026-10-29 |
| Exact bytes & pinned signature in single operator smoke | PASS, synthetic only |
| Detached script and immutable-key input controls | IMPLEMENTED, CI validation required |
| Separate real Attestor host/key custody | NOT TESTED |
| Independent immutable retained copy (WORM) | NOT IMPLEMENTED |
| Re-read at later sampling time | NOT TESTED |
| Pre-cutoff ASML issuer bytes provenance | NOT CERTIFIED |
| Canonical Registry persistence/transition | BLOCKED |

Production is out of scope. No test here justifies a WORM, independent-service, Registry admission, valuation or publication claim.
