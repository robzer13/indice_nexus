# OroTitan Contract Pin Pack V1

Status: PREPARED / NOT YET MERGED. This pack does not create a live Registry run and does not authorize publication.

The pack closes the byte-level provenance boundary for the 13 required `contract_pins`.

Authoritative source commit:

```text
8aba7cee6a9b38204785c16976e65e9010f5d959
```

Final contract-set hash:

```text
34b009f05715bab482dbc00b02194b3714e9b2f8151872144677bb6db19f3c63
```

Rules:

1. `content_sha256` is always the SHA-256 of canonical uncompressed authority bytes.
2. Every `GITHUB_IMMUTABLE` locator pins repository, commit SHA, path and Git blob SHA.
3. Frozen `.gz` contracts use `OROTITAN_GZIP_V1`: Git blob is verified, gzip is decompressed, canonical bytes are SHA-256 verified.
4. Large/project-origin contracts use `OROTITAN_MULTIPART_GZIP_V1`: Base64 parts are individually Git/SHA-256 verified, concatenated, decoded, gzip-verified, decompressed, then canonical-SHA-256 verified.
5. Raw repository authorities use `RAW_CANONICAL_V1`.
6. Resolver failures are fail-closed.
7. `OROTITAN_CONTRACT_PIN_PACK_V1.json` contains exactly the 13 Registry-required logical pins and reconciles to `contract_set_sha256` using the Registry's sorted `logical_name|version|content_sha256` algorithm.
8. A live run must use the final pack unchanged. Any material contract change requires a successor pack/run.

The historic `.xz` Deep Dive archive remains preserved. The pin pack adds a deterministic gzip repackaging of the same canonical Markdown solely so the runtime resolver uses a supported compression codec. Its canonical SHA-256 remains unchanged.

Scope boundary:

```text
LIVE RUN CREATION = NO
PRODUCTION REGISTRY MUTATION = NO
CANONICAL PUBLICATION = NO
```
