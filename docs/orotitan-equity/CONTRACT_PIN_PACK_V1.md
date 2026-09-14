# OroTitan Contract Pin Pack V1

Status: PREPARATION ONLY. This pack does not create a live Registry run and does not authorize publication.

The pack closes the byte-level provenance boundary for the 13 required `contract_pins`.

Rules:

1. `content_sha256` is always the SHA-256 of canonical uncompressed authority bytes.
2. Every `GITHUB_IMMUTABLE` locator pins repository, commit SHA, path and Git blob SHA.
3. Large frozen contracts may use `OROTITAN_MULTIPART_GZIP_V1`. Base64 parts are individually Git/SHA-256 verified, concatenated, decoded, gzip-verified, decompressed, then canonical-SHA-256 verified.
4. Resolver failures are fail-closed.
5. The final `OROTITAN_CONTRACT_PIN_PACK_V1.json` is generated only after source commit A is immutable and its CI has reported the exact direct-source SHA-256 values.
6. A live run must use the final pack unchanged. Any material contract change requires a successor pack/run.

The historic `.xz` Deep Dive archive remains preserved. The pin pack adds a deterministic gzip repackaging of the same canonical Markdown solely so the runtime resolver uses one compression codec. Its canonical SHA-256 remains unchanged.
