# OroTitan — Product MVP slice 01: Research intake inbox (local review pilot)

**Product outcome:** a signed-in admin selects an **existing active Screener company**,
submits a Research **request**, receives a durable request UUID, and can view
that exact receipt/status again after the Next.js development server restarts.

**Status:** review-only DRAFT, isolated branch from **main**; NOT stacked
on the D30–D38 technical PR chain (#410–#418). This is an intentionally
small vertical slice of the USER JOURNEY, not a new OAuth / KMS layer.

## Where to see it

In a local checkout of this review branch:

1. Install existing dependencies (`npm install`) and configure the **existing**
   app's Supabase read settings and admin password in local `.env.local`.
   Point any test to a non-production data source when possible.
2. Set `OROTITAN_LOCAL_REQUESTS_ENABLED=1` in `.env.local`.
3. Run `npm run dev`, log into `/admin`, then open
   `/admin/research-requests` (also linked in the admin nav).
4. Select an **active Screener company** and click **Enregistrer une demande locale**.
   The server verifies that ID against the current company list, then stores
   a single `REQUESTED_NOT_EXECUTED` receipt in
   `.orotitan-local/research-requests-v1.json`.
5. Reload the page or restart the local dev server: the same request UUID
   and non-executed status remain visible. Repeated clicks return the
   **same** receipt for that company; no second request is created.

The local directory is gitignored; files have private permissions.
Writes are serialized with a lockfile and atomic rename to reduce
local concurrency/race issues. Corrupt files fail closed instead of
resetting data silently. The pilot is disabled by default and
**unconditionally disabled when NODE_ENV=production**, even if the
feature flag is accidentally set in a hosting environment.

## Distinct systems — do NOT confuse them

- Screener `companies` already exists and is shown in the admin interface.
  Creating a Screener company is separate from resolving its canonical
  Research `issuer` / `security` (including an independently verified ISIN).
- The local `research-requests-v1.json` is NOT a Supabase table, run
  Registry, research engine job or OAuth/MCP command.
- A receipt with `REQUESTED_NOT_EXECUTED` means exactly that: **the request
  was saved locally**. The app does not claim a real Research has begun.
- The existing `/admin/orotitan` Registry cockpit remains read-only and
  separate; the pilot provides a link, never an invented run ID.
- This local file does NOT synchronize across independent computers,
  phone and PC. A shared durable production queue plus reviewed auth
  remains required for that acceptance criterion.
- This local singleton admin demo is not a general multi-tenant
  authorization design and is intentionally not connected to MCP.

## Acceptance matrix

| Criterion | Expected test |
| --- | --- |
| Choose an existing company | Validate active company ID against server-fetched Screener list |
| Persist a request | Generated UUID and status survive process restart |
| Avoid duplicates | Concurrent submissions for same company return one stable receipt |
| Observe real state | Only `REQUESTED_NOT_EXECUTED`; no fabricated progress |
| Contain blast radius | Development-only, explicit flag, local disk; no Supabase writes |
| Respect Research contract | No automatic Registry run, evidence admission or Deep Dive |
| Avoid technical stack lock-in | Independent main-based PR, no D30–D38 imports |

Automated unit tests cover durable reads, 12 concurrent requests, two
different companies, inactive/unknown IDs, and file corruption. These
unit tests do **not** verify a real browser or live deployment.

## Next product gate: persistence that works across devices

After reviewing the operator workflow and verifying a real browser test:
- Define a single authoritative shared queue backed by the **existing
  Supabase Free project**, and obtain explicit authorization before any
  production schema/role changes or cloud writes.
- Reconcile the currently unmerged D30–D38 experimental branches and
  choose the **minimum** secure backend components; do NOT merge nine
  branches just because their unit tests are green.
- Add issuer/security provenance resolution, cutoff/contract scope,
  operator authentication and server-side confirmation before executing
  anything. Include lifecycle and auditable statuses.
- Prove the same request UUID is visible from phone and PC using the
  shared backend, then connect ChatGPT commands to **that same backend**.
- Separately prove one requested Research can produce a correctly cited,
  quality-gated Equity Research output, with Besi remaining gated until
  its six outstanding evidence issues are resolved.

**Production authorization:** none. No GitHub merge, Vercel deployment,
Supabase migration, plan upgrade, Registry write, Research execution,
Besi stage change, paid resource or ChatGPT write exposure is authorized.

**Product decision:** next work is a *shared persistent inbox and honest
status tracking*, not another generic OAuth/security prototype.

## Triage décisionnel des PR #410–#418

**Ne pas les fusionner en bloc** pour livrer ce premier parcours :
- **#410 D30** — conserver le langage de demande/confirmation, les validations
  et les exigences d'idempotence comme référence pour l'adaptateur partagé,
  mais ne pas activer la fonction MCP candidate.
- **#411–#414 D31–D34** — conserver les prototypes OAuth/capabilities/
  rôles PostgreSQL/coffre AES-GCM comme matériau de revue de sécurité.
  Aucun de ces composants n'est requis pour visualiser la boîte locale.
- **#415 D35** — prendre en compte explicitement les constats de sécurité
  et les critères de NON-LIVRAISON; rien à débloquer automatiquement.
- **#416 D36** — reprendre ultérieurement la règle produit de déduplication
  par opérateur/émetteur/contrat/cutoff, une fois l'identité canonique
  et la file partagée validées.
- **#417–#418 D37–D38** — conserver comme expériences d'isolation de session.
  Ce ne sont pas des connecteurs ChatGPT opérationnels et ils ne
  prouvent pas le support de l'OAuth mobile en production.

**Prochaine PR produit souhaitée :** une seule boîte de réception Research
partagée et accessible sur plusieurs appareils, accompagnée d'une revue
séparée du schéma et des droits nécessaires. Pas de système parallèle
de demandes dans le Screener et le Registry.
