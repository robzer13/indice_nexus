# OroTitan V3 — ZERO-EUR proof preservation architecture (SHADOW decision, 2026-10-29)

**Status: PROPOSED, FREE-TIER-BOUND, NO AUTHORITY ESCALATION.** This proposal supersedes the *AWS-first deployment recommendation* in `SHADOW_AWS_KMS_LAMBDA_OBJECT_LOCK_ARCHITECTURE_PROPOSAL_20261029.md` for the present **strict 0 EUR recurring cost** constraint. It does **NOT** amend or weaken any FREEZE contract and does **NOT** certify independent custody, external WORM, historical provenance, or permit Registry admission.

## Why AWS is not a prerequisite

The frozen `OROTITAN_RUN_ARTIFACT_REGISTRY_MANIFEST_SPEC_V1_FREEZE_V1.0`, sections 3.2, 12 and 16.2, already specifies:
- **PRIVATE** GitHub repository (preferred) for sealed Markdown/JSON stage artifacts + manifests; **PRIVATE** Supabase Storage as the binary source storage target;
- SHA-256 of exact bytes and, for GitHub artifacts, blob/commit SHA;
- artifact-first, Registry-second, verifying all stored bytes before registry finalization;
- no private artifacts in public `indice_nexus`.

It does not require AWS KMS, S3 Object Lock, an extra Vercel account, or paid services as conditions for *development*. This zero-cost design preserves existing gates, which must remain `BLOCKED` where required proof is unavailable.

## Proposed 0 EUR recurring-cost components

1. **Existing Supabase SHADOW Free private bucket**: keep the original content-addressed PDF, receipt evidence, private file coordinates. Existing Writer/Attestor RLS tests PASS. Free tier subject to usage caps and inactivity pausing; never treat it as guaranteed long-term availability.
2. **Separate PRIVATE GitHub Free repository**, not a branch of the PUBLIC `indice_nexus` repo: sealed text/JSON stage artifacts, Gate audit snapshots, non-secret cryptographic receipts, hashes, optional permitted private source-copy backups. Use the dedicated private-repository boundary required by FREEZE. Private repos are included in GitHub Free.
3. **GitHub immutable releases**, explicitly enabled under the new private repository's Settings > Releases before the **first** evidence release. After publication, release asset bytes and Git tag are locked, as long as that release exists. Every release must contain a deterministic manifest with the exact SHA-256 of every archive member plus Gate nonce, receipt digest, log-chain head, and storage coordinates *only if permitted to be disclosed within the private repo*. Preserve exact tag, associated commit SHA, release asset digest, and publication UTC. Immutable releases are *not equivalent* to S3 Object Lock Compliance: a repository or a release can be deleted by authorized administrative action and platform access/availability is not guaranteed. No claim of governance-independent WORM.
4. **OpenTimestamps** public calendars: produce a small, **non-sensitive** anchor manifest and submit its hash to free calendar servers (no password, account, Bitcoin balance, or fee required). Retain the original manifest and resulting `.ots` file, perform `ots upgrade` once Bitcoin confirmation is available, then run independent `ots verify`. Bitcoin inclusion later gives objective proof the hash existed **before the confirmation block time**, not evidence that the underlying financial report was authentic, or existed at a historical source publication date. A pending calendar receipt is **NOT** final timestamp proof. Never publish source PDFs, URLs with sensitive query parameters, secrets, or private individual identifiers.
5. **Offline recovery copy**: retain an encrypted external USB/drive copy of source bytes and exported evidence `manifest + .ots + signed receipt + Gate log + public key`. This is a recommended additional availability control but is not WORM. A real two-principal/independent-admin test still needs two operators/hosts or independently administered custodians.

## Proof-unit / release layout (illustrative, no actual archive deployed)

```text
PRIVATE GitHub repo: orotitan-evidence-v1  [must be created PRIVATE]
  releases/ (GitHub UI/API: immutable releases enabled BEFORE publishing)
  release tag: shadow-evidence-v1-YYYYMMDD-<short-digest>
    SHA256SUMS.txt
    source-manifest.json                 # redact secrets and non-public identifiers
    signed-attestor-receipt.json        # no JWT, no passwords, no private PEM
    gate-verification-report.json
    gate-ledger-snapshot.jsonl          # sensitive -> private-only
    public-key.pem
    ots-anchor-manifest.json            # only non-sensitive hashes
    ots-anchor-manifest.json.ots        # upgraded proof after Bitcoin confirmation where feasible
```

Source binary files remain in PRIVATE Supabase Storage and may be copied to encrypted backup, with legally/contractually appropriate retention. Do not assume the source is redistributable. Never commit private evidence to public `indice_nexus`.

### Required hash linkage

```text
source original exact bytes
  -> source SHA-256  -> signed receipt statement -> Ed25519 signature
  -> gate verification report (signed receipt hash)
  -> Gate append-only-local audit head digest (LOCAL_ONLY_NOT_WORM)
  -> SHA256SUMS archive manifest (hash of each evidence file)
  -> GitHub immutable release tag + asset digest
  -> OpenTimestamps proof of non-secret archive-manifest digest
```

The external OTS anchored digest detects *subsequent changes* to the original bytes as long as the verifier retains or can recover the originals and the confirmed `.ots` proof. It does not keep deleted files retrievable.

## Operator workflow — no cloud purchase

0. Confirm GitHub Free account and existing Supabase Free quotas. Do NOT enter payment data or upgrade existing accounts as part of this plan; set spending limit 0 where offered.
1. Create the dedicated GitHub **PRIVATE** repository `orotitan-evidence-v1` via GitHub UI or approved GitHub connector. Verify PRIVATE status **before** first artifact upload.
2. In that repository, enable **Settings -> Releases -> Enable release immutability**. Confirm API/UI reflects enabled before publishing (only affects future releases).
3. Prepare the private evidence bundle from *previously verified* exact local file bytes, excluding credentials, JWTs, API keys, private PEM. Verify the 715-byte synthetic exact hash `f9cda65b308c24e62fd9a5070df43202c2c0d4560a8aba8f6e697a8a9f467b9b`; receipt hash `a9af122689524c9b7efd335a0de07c1604fb3c897713c12ab0a01adaba7374eb`; verified public PEM SHA-256 `5288aa9bf268dc32feff913810d21ffb3ff4af7d3efc3ca65c7d64713604d096`; Gate log head digest `789a5c088063715cd57a0927762733e75fde215420ae73a1dce16dadb0bad241`. These are **prior-test values**; recompute from originals for new evidence, do not copy prior hashes as proof of new bytes.
4. Publish a **synthetic-only** immutable release with exact manifests, log and public material, respecting private repository permissions. Test read-after-publish and check attempted asset replacement is rejected (no destructive security tests on actual archive).
5. Create an `ots-anchor-manifest.json` of only non-sensitive data (e.g. exact SHA-256 hashes), stamp at https://opentimestamps.org/ or `ots stamp ots-anchor-manifest.json`. Keep `.ots`, then `ots upgrade` / `ots verify` once confirmed. If proof remains pending, record `OTS_CONFIRMATION_PENDING` and fail-closed.
6. Make a second encrypted offline backup of evidence and essential source bytes, verify restore and exact hashes.
7. Save GitHub repo/tag, release checksums, .ots proof, Bitcoin block proof, backup restore result and source availability status in the non-authoritative SHADOW verification report. Only propose Registry admission through its **pre-existing** FREEZE contract after all contract-required gates independently pass.

## Negative acceptance / must remain blocked

- If `SOURCE_BYTES_AVAILABLE=false` or SHA mismatch, **BLOCK**.
- If no confirmed OTS, status `EXTERNAL_TIMESTAMP_PENDING`; never display `ANCHORED_CONFIRMED`.
- If GitHub repo is public or immutable releases off, do **NOT** upload private evidence or claim immutable artifact.
- If key custody/host/admin independence is not separately demonstrated, `INDEPENDENT_CUSTODY_CERTIFIED=false`.
- Always `WORM_COMPLIANCE_CERTIFIED=false`, `EXTERNAL_IMMUTABLE_REGULATED_ARCHIVE=false` under this free configuration. GitHub release immutability is repository-level tamper resistance, **not regulatory object retention**.
- Historical issuer provenance, financial figures, ASML G001/G006, Gate-to-Registry policy and production publishing are **not** proven by this evidence packaging.
- No PR merge, production writes, Registry modifications or stage-state changes.
- Free-tier availability and commercial-use restrictions of each provider must be checked prior to any operational promotion; no SLA or guaranteed indefinite 0 EUR.

## References checked 2026-10-29

- FREEZE source: `OROTITAN_RUN_ARTIFACT_REGISTRY_MANIFEST_SPEC_V1_FREEZE_V1.0`, sections 3.2, 12, 16.2.
- GitHub Free and private repositories: https://github.com/pricing
- GitHub immutable releases: https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases
- Enable: https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/establish-provenance-and-integrity/prevent-release-changes
- OpenTimestamps free calendar servers: https://opentimestamps.org/
- OTS upgrade/verify: https://github.com/opentimestamps/opentimestamps-client
- Supabase Free quotas, project pauses: https://supabase.com/docs/guides/platform/billing-on-supabase ; https://supabase.com/docs/guides/platform/free-project-pausing
