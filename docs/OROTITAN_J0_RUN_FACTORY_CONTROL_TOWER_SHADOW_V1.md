# J0 — OroTitan Run Factory Control Tower (SHADOW)

Version 1.0 · 2026-10-31 · Programme approuvé J0→J8.
Développement J0 autorisé, **acceptation J0 non présumée**.

## Boundary

- Production Supabase `cugpgtzygqqlxetyeven` : aucune intervention.
- SHADOW Supabase `awgsurdyvsyolcgpnygh` : cible unique.
- Contrats FREEZE V1/V2/V3.1 : jamais modifiés.
- Runs, stage, artefacts, snapshots et publication : **lecture seule**.
- J1→J8 : aucun GO de création de fonctionnalités de ces jalons dans J0.
- J9 production : expressément exclu.

## Data authority and single source of truth

| Question | Source autoritative | Contournement interdit |
|---|---|---|
| Quels jalons existent ? | `orotitan_j0_milestones_v1` | Statuts proposés par assistant |
| Quelle est la prochaine validation ? | Critères figés `gate_criteria` et décisions append-only | « Tout est prêt » sans tests |
| Quels faits sont déclarés ? | `orotitan_j0_events_v1` / état `REPORTED` | Lien seul => preuve vérifiée |
| Quels runs et stages existent ? | `orotitan_runs` + stages et events Registry | Cache UI ou plan local |
| Quelle preuve est vérifiée ? | `verified_snapshot` généré par une RPC à partir du Registry | Annotation du navigateur |
| Quelles exceptions restent ouvertes ? | Dernier événement par clé `issue:...` | Suppression d’une ligne historique |

Milestone statuses at initial seeding:
- J0 = `EN_COURS`
- J1 = `EN_COURS` (code partiellement construit ; **non accepté**)
- J2–J8 = `NON_DEMARRE`
- Aucune ligne n’est `ACCEPTE`.

La Control Tower n'implique pas que J1 est fonctionnel de bout en bout :
l'HTTP authentifié doit être démontré après J0.

## Event journal

All events are immutable, with a unique `event_key` and an actor
`AUTHORIZED_SHADOW_OPERATOR`.

Allowed operator event kinds:
- `EVIDENCE_RECORDED`
- `BLOCKER_OPENED`
- `BLOCKER_RESOLVED`
- `GATE_REVIEW_REQUESTED`

Seed governance events:
- `ROADMAP_APPROVED`
- `J0_STARTED`

**No** `ACCEPTED` or `GO_RESEARCH` event can be recorded through J0.
`BLOCKER_RESOLVED` requires a real preceding open issue. A repeated event
key with another payload is denied. Events never overwrite history.

A proof reference starting with `registry:run:<UUID>` is verified by the
server against the actual `orotitan_runs` and related receipts, with
re-read facts captured in `verified_snapshot`. Other proof references,
including links, are initially labelled `REPORTED` — they require a separate
verification before passing any gate.

## J0 exit criteria — explicit audit

1. 9 milestone rows and exact gates live in SHADOW.
2. Governance authorization and start decision persisted and read back.
3. At least one actual Registry run re-read as direct verified evidence.
4. Negative tests pass: wrong event type, cross-project gate,
   unauthorized roles, duplicate idempotency key with different payload,
   issue closure without opening.
5. Real admin UI is accessible, shows Registry state and logs evidence events.
6. Real authenticated browser action persists and re-renders a received event;
   unauthenticated 401 and cross-origin 403 are validated.
7. Production and historical stage/artifact/snapshot counters unchanged.
8. Independent receipt with Git SHA, deployment ID, DB counts, tests and
   explicit operator review. **Only then** request `GO ACCEPT J0`.

Until all eight items are evidenced, J0 remains `EN_COURS` or `TESTE`,
never `ACCEPTE`. J1 E2E qualification and J2 source acquisition are not
automatically started by completing this dashboard.

## Operator procedure

1. Login as an administrator in the isolated Preview.
2. Open `/admin/research-factory/control-tower`.
3. Check the 9 milestones and the L’Oréal run ID
   `204390a1-5cba-4b9c-9821-ca0c7df43eca`.
4. Journal: record an evidence reference or issue key. Entries are
   immutable and include a post-write Registry readback.
5. Keep a blocker open for any unproven acceptance criterion.
6. Export the exact readback/commit/Preview evidence to the gate review.
7. Explicit acceptance and J1 activation remain separate user decisions.

The current plan `OROTITAN_RUN_FACTORY_PLAN_DIRECTEUR_JALONS_V1_20261031`
is an operational roadmap, not a substitute for the frozen methodology.
