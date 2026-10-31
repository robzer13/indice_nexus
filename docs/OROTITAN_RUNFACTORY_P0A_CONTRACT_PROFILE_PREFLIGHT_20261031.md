# OroTitan Run Factory P0-A — Contract Profile Resolver / Preflight

**Status:** SOFTWARE CANDIDATE / SHADOW ONLY. 2026-10-31.
**Blueprint:** `OROTITAN_BLUEPRINT_INDUSTRIALISATION_RESEARCH_V1_2026-10-31.md`.
**Methodology change:** NO. **Registry write:** NO. **Production:** UNTOUCHED.

## Goal

Stop reconfiguring contract pins for each issuer. Pull existing immutable
authority packs directly from the repository:

| User-selected profile | Existing pack | Expected logical pins |
|---|---|---:|
| `OROTITAN_V1` | `contracts/orotitan-equity/v1/contract-pin-pack-v1/OROTITAN_CONTRACT_PIN_PACK_V1.json` | 13 |
| `OROTITAN_V2` | `contracts/orotitan-equity/v2/contract-pin-pack-v2/OROTITAN_CONTRACT_PIN_PACK_V2.json` | 13 |
| `OROTITAN_V3_1` | `contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json` | 15 |

V3.1 is the *methodology label*; its pin-pack file version is `3.0.2`.
A new version must get a new explicit profile, not silently change an
existing frozen set. The `contract_set_sha256` is recomputed by the
existing Registry-compatible canonical algorithm and compared with its
profile-specific frozen constant.

The catalog validates exact key set, expected count, repository, immutable
Git commit, Git blob SHA-1, canonical SHA-256, locator format and safe path.

## Byte verification

The admin-only, SHADOW-only page:
`/admin/research-factory/contracts`

1. Requires the operator to select an **explicit profile**; no default
   or inference from ticker/cutoff/newest.
2. Displays pack structural verification separately from byte verification.
3. POSTs each logical pin to
   `/admin/research-factory/contracts/verify`.
4. Endpoint re-fetches **only** pinned content from the fixed
   `raw.githubusercontent.com/robzer13/indice_nexus` host, at the pinned
   commit. Response capped by bytes and timeout.
5. Reuses `lib/orotitan-equity/v1/contract-pin-resolver.ts`
   to verify Git blob and canonical uncompressed SHA-256. Supports raw,
   gzip and multipart gzip, including cross-version V1 reused pins.
6. Stops and returns fail-closed on any missing/tampered authority.
7. Can export a PRE-ADMISSION JSON report in client memory.

HTTP operations never accept arbitrary URL, arbitrary repository or
user-provided SHA-256. Session and origin checks are required, and the
existing hosted SHADOW feature gate remains in force.

## Coverage and safety

CI test checks all 41 `(profile,pin)` references: 13 V1, 13 V2, 15 V3.1.
The same logical authority can be reused immutably across profiles.
Corruption and missing-profile tests fail closed.

A complete 13/13 or 15/15 verification proves **that the bytes
match the existing pinned authority**, not that a given user selected the
correct methodology, nor that the issuer has been resolved, nor that
the Registry is accessible, nor that a Research run is admitted.

- `issuerResearchAuthorized=false`
- `registryAdmitted=false`
- `runCreated=false`
- `stageAdvanced=false`

No Supabase queries or mutations in this module. No issuer-specific
configuration is needed to verify the contract catalog.

## Next P0-B (NOT DELIVERED HERE)

Implement the immutable identity resolver and run bootstrap as a single
idempotent, documented transaction using the **existing controlled RPC**,
only after the selected contract profile and environment permissions are
affirmed. Re-query the created run before entering Research. Maintain
V1/V2/V3 generation boundaries and immutable cutoff / pins.
