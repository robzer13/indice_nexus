# OroTitan V3 — private Storage shadow: real Auth HTTP smoke PASS (2026-10-29)

## Decision / scope

**TRANSPORT / ACL SMOKE = PASS.**
**REGISTRY ADMISSION = NOT PERMITTED.**
**INDEPENDENT ATTESTATION SERVICE = NOT PROVEN.**
**PRODUCTION = NOT TOUCHED.**

The human operator executed the opt-in synthetic-source HTTP smoke in the nonproduction Supabase project `orotitan-vnext-shadow`, against the reviewed GitHub PR #434 commit `df5e158f8b16e1e978732efcb0c5d652af3a48b8`, after the shadow-only RLS operation fix `shadow_attestor_authenticated_get_and_info_acl_fix_20261029`.

### Operator-returned result (non-sensitive excerpt)

- `schema`: `SHADOW_PRIVATE_STORAGE_OPT_IN_SMOKE_V1`
- `testProjectConfirmed`: `true`
- `distinctAuthenticatedUsersConfirmed`: `true`
- `uploaderReadDenied`: `true`
- `readerWriteDenied`: `true`
- `signerTransportProof`: `VALID_PINNED_ED25519`
- `sha256`: `f9cda65b308c24e62fd9a5070df43202c2c0d4560a8aba8f6e697a8a9f467b9b`
- `sizeBytes`: `715`
- `residualNonprodObjectPresent`: `true`
- `admittedToRegistry`: `false`
- `fullyIndependentServiceProven`: `false`

**Do not record account emails, passwords, publishable/secret API keys, JWTs, private PEMs or privileged user UUIDs in this document.** The claim that the signed reread bytes/hash matched is grounded in the operator's smoke output and the inspected source code; the connected management tool cannot independently reconstruct or inspect the body via read-only SQL.

### Independently checked live project evidence

A subsequent read-only query to Supabase `storage.objects` found **two** private-bucket synthetic PDF objects (one from the previous failed smoke, one from the passing smoke), both 715 bytes, content-addressed by the test SHA-256 above, with MIME `application/pdf` and under `preadmission/`. The passing object has creation timestamp `2026-10-29 14:28:33.822593+00`.

The corresponding Storage HTTP logs for the passing object show, at UTC times:
- `14:28:33.858`: `POST /object/orotitan-source-files-v1/...` → HTTP **200** (Writer create).
- `14:28:34.116`: `GET /object/info/orotitan-source-files-v1/...` → HTTP **400** (negative Writer read test).
- `14:28:34.334`: `POST /object/orotitan-source-files-v1/preadmission/denied/...` → HTTP **400** (negative Attestor insert test).
- `14:28:34.532`: `GET /object/info/orotitan-source-files-v1/...` → HTTP **200** (Attestor info read, fixed).
- `14:28:34.623`: `GET /object/orotitan-source-files-v1/...` → HTTP **200** (Attestor byte retrieval).

Shadow was confirmed to have **exactly one Writer**, **one Attestor**, and **one private source-files bucket**. Thus both positive and negative transport observations are corroborated by the Storage service, not only SQL RLS simulations.

### What this does NOT certify

1. The smoke verifier and test Ed25519 private key were created in one operator-managed process; cryptography is valid, but independent **signing credential custody / separate service execution principal** is not proven.
2. An RLS create-only authorization policy is not proof of immutable retained bytes at the storage-admin/service-role level; no WORM/object-lock attestation or long-term byte-retention guarantee has been demonstrated.
3. The source file was **synthetic**. Neither genuine ASML source bytes nor verifiable historical cutoff bytes were admitted. Source-publication date was only declared, not independently certified.
4. No canonical Registry persistence receipt/attestation, source admission, handoff, frozen method, valuation, DCF, price or GO PUBLISH follows from this smoke. The legacy Registry writer functions remain unchanged, and production must remain protected.
5. PR #433 and PR #434 remain **open, unmerged and review-only**.

### Next governed gate

Prepare and review the independent attestor key-custody/identity and persistence-design controls. Prove retention/immutability properties, historic source provenance and forward-only Registry backend support separately, with regression testing and explicit governance; **do not** promote a synthetic `NON_AUTHORITATIVE` receipt into a production Registry claim.
