# OroTitan Frozen Execution Contract Archive V1

**Status:** archive candidate for immutable Git commit

This directory archives the five frozen execution contracts required before a first live Registry run.

These files are contract-only public artifacts. They MUST NOT be used as a store for private run artifacts, licensed reports, user documents, or source evidence.

## Verification rule

The authoritative `content_sha256` for a contract is computed over the exact uncompressed Markdown bytes. Compression is packaging only.

Verify before pinning:

```bash
# .gz archives
gzip -dc <archive>.md.gz | sha256sum

# .xz archive
xz -dc <archive>.md.xz | sha256sum
```

The result MUST equal `canonical_sha256` below. The Git locator for a live `GITHUB_IMMUTABLE` pin must use the final immutable commit SHA plus the listed Git blob SHA.

| logical pin | version | archive path | packaging | canonical_sha256 | archive_sha256 | git_blob_sha |
|---|---:|---|---|---|---|---|
| `process` | `1.0` | `contracts/orotitan-equity/v1/execution-freeze/OROTITAN_RESEARCH_EXECUTION_PROCESS_V1_FREEZE_V1.0.md.gz` | `gzip -n -9` | `943745d3c413c39ddbabb2512605d364754c213628b15fdf9b9b58e78be0eb6a` | `76efbde88e405f2c5bd8041b46025e1bd4a5034b67da26ed9b4a7e327dc77aaf` | `7fccc918b8bc5a1d4b03df6c957ebcbc1dfac4f3` |
| `pilotage` | `1.0.1` | `contracts/orotitan-equity/v1/execution-freeze/OROTITAN_PILOTAGE_ORCHESTRATION_CONTRACT_V1_FREEZE_V1.0.1.md.gz` | `gzip -n -9` | `304e19d572be39516914b069d33e10ee53e472b15c5601f2e0a2d2e0a8ccc446` | `ae392dcc6bab2f07a279283936007b24240b37f14f35fb7fa146dcfb9ed8da90` | `5b7ed7ad92906c2fa1cbdaa5775bb0a36ac69c83` |
| `research_stage` | `1.0` | `contracts/orotitan-equity/v1/execution-freeze/OROTITAN_RESEARCH_STAGE_CONTRACT_V1_FREEZE_V1.0.md.gz` | `gzip -n -9` | `b9a5930c91d90e498e41c1640f912fb5e38fd0ad607899a5e2ca0a616f3d47a2` | `f872e5baa14a644856a072bbde6c81e2b9edfd97cef8151e57ba08b7f4e8e8b6` | `3bf94124835fe17b31e7b11410274ed585f92aa5` |
| `deep_dive_stage` | `1.0` | `contracts/orotitan-equity/v1/execution-freeze/OROTITAN_DEEP_DIVE_STAGE_CONTRACT_V1_FREEZE_V1.0.md.xz` | `xz -9e` | `21586e572739e2564781ce2814758d3d557c151fee5866d8c75cb0394f9b82c5` | `eebcc29a4f30a9211a65d1224b61540adf16fcb32765b1ea25cb3790e60d4e6f` | `8ab0cdc8509af5711c111c9323d1caa3e1f8277a` |
| `integration_stage` | `1.0` | `contracts/orotitan-equity/v1/execution-freeze/OROTITAN_INTEGRATION_STAGE_CONTRACT_V1_FREEZE_V1.0.md.gz` | `gzip -n -9` | `66bbe2f0ee8955feaba33a2c52b174f94695148a7884e35ef66704e2a6304ccb` | `3b8ca5ed4df5eabd595d4bd09c0ac32a4aab2a0ea90be5b319e24ccf815b253e` | `612bcb58f629cd6e7338964424a94f1a71f83931` |

## Scope boundary

This archive does not authorize or perform:

- a Registry migration on production Supabase;
- creation of a live research run;
- canonical snapshot publication;
- any production data rewrite.

The private run-artifact storage boundary remains separate from this contract archive.
