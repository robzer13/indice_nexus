# OroTitan Screener V1.2 — Codex Execution Plan

## Mission

Implement the approved V1.2 UX/UI redesign on branch:

`feat/orotitan-screener-v1-2-ux`

Repository:

`robzer13/indice_nexus`

Primary specification:

`docs/UX_UI_V1_2_SPEC.md`

The specification is authoritative for this refactor.

## Critical guardrails

- Do not change OroTitan financial formulas.
- Do not invent financial values.
- Do not mutate historical snapshots.
- Do not rewrite market_prices.
- Do not weaken auth, RLS, secret handling or append-only guarantees.
- Avoid schema changes.
- Avoid unnecessary dependencies.
- Use Server Components by default.
- Preserve current Yahoo Finance / Twelve Data provider routing.
- Preserve the current 65-second provider cooldown and price-aberration guard.
- Keep all existing routes functional.

## Working method

Implement in small reviewable commits.

For every ticket:
1. inspect current implementation before editing;
2. reuse existing components where sensible;
3. make the smallest coherent change;
4. run:
   - npm run lint
   - npm run typecheck
   - npm test
   - npm run build
5. do not proceed if a regression is unresolved.

Do not merge to main directly.

## Tickets

### UX12-001 — Design foundations
Implement:
- semantic surface/text/state tokens;
- normalized spacing/radii;
- premium dark cockpit palette;
- shared Panel, SectionHeader and MetricCard primitives;
- consistent focus/hover states.

Acceptance:
- no business logic changes;
- existing pages remain usable;
- build green.

### UX12-002 — Global shell
Implement:
- left sidebar;
- active nav state;
- top command bar;
- responsive navigation;
- preserve only existing routes.

Acceptance:
- desktop and mobile navigation usable;
- no fake user or market status values;
- actual freshness/sync information only where available.

### UX12-003 — Screener visual refactor
Implement:
- premium filter area;
- compact/sticky table header where clean;
- improved columns/spacing/badges;
- Upside FV emphasis;
- visible row hover/focus;
- responsive overflow behavior.

Do not change filtering/sorting semantics except presentation.

Acceptance:
- all V1.1 filters still work;
- default distance sort unchanged;
- NULL behavior unchanged;
- row keyboard navigation works.

### UX12-004 — Screener filter ergonomics
Implement:
- common filters prominent;
- advanced numeric filters grouped cleanly;
- active filter summary/chips if maintainable;
- full reset behavior preserved;
- result count easy to scan.

Acceptance:
- reset clears filters and restores default sorting;
- no query semantics changed.

### UX12-005 — Dashboard cockpit
Implement:
- premium PageHeader;
- KPI strip;
- priorities block;
- entry-zone distribution;
- screener preview.

Use actual data only.

Acceptance:
- counts reconcile to current company states;
- priority order remains O90-first then closest above;
- no uncalibrated distance fabrication.

### UX12-006 — Company detail page
Implement:
- premium identity header;
- current state;
- valuation ladder;
- Upside FV;
- chart visual polish;
- thesis/risk/invalidation cards;
- snapshot history polish.

Acceptance:
- current Hermès example still computes distance exactly;
- chart contains market_prices only;
- analytical thresholds come from selected snapshot only.

### UX12-007 — Admin shell and feedback
Implement:
- consistent admin nav/panels;
- improved forms;
- improved success/warning/error feedback;
- refined price sync journal;
- refined Data Health presentation.

Acceptance:
- admin auth unchanged;
- immutable snapshot behavior unchanged;
- price sync remains functional.

### UX12-008 — Loading / empty / error states
Implement reusable:
- skeletons;
- empty states;
- error panels;
- retry affordances where possible.

Acceptance:
- no replacement numbers are generated on failure.

### UX12-009 — Responsive and accessibility pass
Audit:
- 1440+ desktop;
- 1280 desktop;
- tablet;
- mobile.

Check:
- keyboard;
- focus;
- contrast;
- labels;
- table semantics;
- reduced motion.

### UX12-010 — Final release verification
Run complete CI.
Update version/display copy to V1.2.
Prepare PR against main with:
- screenshots;
- change summary;
- regression statement;
- test evidence.

## Definition of done

The release is done only when:
- UX spec acceptance criteria all pass;
- lint/typecheck/tests/build all green;
- no V1.1 business regression;
- production-ready PR is open for review;
- no direct main merge without explicit approval.
