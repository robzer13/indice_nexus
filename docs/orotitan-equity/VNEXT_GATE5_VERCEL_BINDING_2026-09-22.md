# OroTitan VNext - Gate 5 Vercel Binding Attestation

Date: 2026-09-22

## Project binding

- Vercel project: `orotitan-vnext-pilotage`
- Git repository: `robzer13/indice_nexus`
- Tracked production branch for this isolated VNext project: `vnext`
- Shadow Supabase project ref: `awgsurdyvsyolcgpnygh`
- Production Supabase project ref explicitly forbidden to VNext runtime: `cugpgtzygqqlxetyeven`

## Environment contract

The Vercel project is configured with the following VNext shadow contract:

- `OROTITAN_ENVIRONMENT=VNEXT_SHADOW`
- `OROTITAN_EXPECTED_SUPABASE_PROJECT_REF=awgsurdyvsyolcgpnygh`
- `NEXT_PUBLIC_SUPABASE_URL=https://awgsurdyvsyolcgpnygh.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY` is supplied from the shadow project only and is never committed.
- `OROTITAN_PUBLICATION_ENABLED=false`

## Gate condition

This commit intentionally triggers the first post-binding deployment from the `vnext` branch. Gate 5 passes only after Vercel reports a READY deployment whose Git metadata identifies `vnext` and this commit (or a descendant) as the deployment source.

Production remains outside the VNext execution path.
