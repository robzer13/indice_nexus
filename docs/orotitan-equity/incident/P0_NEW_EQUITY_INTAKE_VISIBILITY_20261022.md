# P0 — Entrée d'une nouvelle action dans le Screener principal

**Status: DRAFT / REVIEW ONLY. No production rollout, DB mutation or publication.**
Observed on 2026-10-22 in production `orotitan-screener` (main SHA `86a26f44...`):

1. The admin "Ajouter une société" form writes to legacy `public.companies` only. No `issuers`, `securities`, `research_dossiers` or canonical `research_snapshots` are created.
2. `getCompanyStates()` explicitly reads **only active published canonical dossiers** with a non-null `current_snapshot_id`; hence new legacy rows do not appear in published Screener / homepage rankings.
3. Supabase has 39 active legacy companies (all have audited quote crosswalks as of Oct 22); only 34 active published dossiers. Five preexisting unscored candidates have no published snapshot: **Auto Trader (AUTO), Baltic Classifieds Group (BCG), Medistim (MEDI), Scout24 (G24), SEI Investments (SEIC)**. No newly inserted legacy row since 2026-10-05 22:33 UTC was observed. This does **not** prove form submissions succeed or fail in users' browser.
4. There is a second older legacy `public.snapshots` workflow. Creating such a snapshot does not populate `public.research_snapshots` or a canonical published pointer; therefore it is **not** a shortcut to canonical publication.

## P0 fix: visible pending intake, *not* fake research

- Keep the public dashboard, canonical research ranking, scores OQS/OVS/Investment, entry thresholds and Data Health **strictly bound** to the 34 published canonical dossiers.
- Admin-authenticated users see a separate, clearly labeled section on `/screener`, **"Titres ajoutés au suivi — non publiés"**, and a `Publication` status in `/admin/companies`. A newly created legacy row goes to that tracking queue but never to published scoring.
- The pending classifier matches an existing legacy company against an approved four-part crosswalk (legacy company, issuer, security, dossier) **and** the active canonical snapshot pointer, without ticker or UUID equality shortcuts.
- Non-admin visitors do not receive the pending queue in the server response. It uses the existing cookie-based server admin session.
- After creation, the admin sees an explicit success message stating that the action is tracked, **not canonically published**.
- No schemas, RLS, stored data, publication gate, historical prices or Registry records change.

## Remaining work for a complete end-to-end new-action workflow

This PR is **stage 1** (visible & understandable tracking). It does not automate:
- security identity/ISIN verification or exchange/currency/unit validation by a trusted data source;
- creation of canonical `issuers`/`securities` or a pending research dossier;
- research stage execution and approval of an immutable canonical snapshot;
- manual/governed publication and transfer of a pending action into canonical scoring;
- cotation feed diagnostics or automatic provider lookups at create time.

Implement these later via a separately reviewed atomic onboarding workflow with explicit user controls. Never silently fabricate scores or switch legacy `snapshots` into canonical `research_snapshots`.

## Release requirements

- GitHub CI (lint, typecheck, tests, PostgreSQL integration, production build), and PR code review.
- Admin-authenticated smoke test must show 34 canonical + 5 pending, no scores for pending, and the existing 34 unchanged. No new real company or synthetic prod row should be created simply to test.
- Anonymous/nonadmin request must not include pending names or server-provided candidate data.
- Explicit, **separate** approval before merging into main and deploying `orotitan-screener`.
