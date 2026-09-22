# OroTitan Equity Research VNext

Status: DEVELOPMENT / SHADOW ONLY

## Frozen production baseline

- Tag: `orotitan-production-pre-vnext-2026-09-22`
- Git commit: `5442172d98b19078e2c46d187b697efbbedfae73`
- Production Supabase project: `cugpgtzygqqlxetyeven` (`orotitan-screener`)
- Shadow Supabase project: `awgsurdyvsyolcgpnygh` (`orotitan-vnext-shadow`)
- Production canonical corpus at freeze: 33 published dossiers / 33 V2 canonical snapshots
- Active production runtime at freeze: V3.0.2
- Active runtime contract-set SHA-256: `257c287357c19a5d47a42f140a1eb0377d48701b04b07e1e9e740646797c172c`
- Frozen V2 contract-set SHA-256: `1116ca12dce2d30ddbb4b699945d92ae235c940cbf69d104a01019fc21efbf5e`

## Hard boundaries

1. VNext MUST NOT write to production canonical snapshots.
2. VNext MUST NOT mutate production Registry state during shadow development.
3. VNext publication authority is disabled until an explicit later release gate.
4. Every VNext change must be committed on the `vnext` branch and validated before merge.
5. Production history remains immutable and reconstructable from the frozen tag.

This directory is the conceptual and contractual authority root for VNext. No new analytical rule becomes authoritative merely by being implemented in code.
