# OroTitan Screener V1.2 — UX/UI Implementation Spec

Status: APPROVED DESIGN DIRECTION  
Branch: `feat/orotitan-screener-v1-2-ux`  
Product scope: UX/UI refactor only. Preserve V1.1 data, business rules, security and append-only guarantees.

## 1. Product goal

Transform OroTitan Screener from a functional V1.1 cockpit into a premium, highly legible investment workspace.

The approved visual direction is a hybrid of:
- an elegant editorial / premium screener;
- a data-dense institutional control room.

The result must feel:
- premium;
- sober;
- dark;
- data-first;
- fast to scan;
- coherent across Dashboard, Screener, Company and Admin;
- realistic and maintainable in Next.js + Tailwind.

This is not a redesign of the OroTitan financial model. It is a UX/UI refactor.

## 2. Non-negotiable invariants

Do not change these rules while implementing V1.2:

1. Distance O90 remains exactly:
   `(price_o90 / current_price - 1) * 100`.
2. NULL analytical values remain NULL. Never invent financial data.
3. `market_prices` remains append-only.
4. Analytical snapshots remain immutable and versioned.
5. Market prices never rewrite snapshots.
6. Service-role secrets remain server-only.
7. `quality_orotitan` remains a structural boolean.
8. O90 absent means `Non calibré`.
9. Existing V1.1 routes and workflows must keep working.
10. Existing provider routing remains:
    - European symbols through Yahoo Finance;
    - supported US symbols through Twelve Data.
11. Existing price normalization and anti-aberration guards must remain active.
12. No database migration should be introduced solely for presentation.

## 3. Approved visual direction

### 3.1 Overall aesthetic

Dark institutional cockpit with restrained premium styling.

Visual characteristics:
- deep navy / graphite background;
- teal-cyan primary accent;
- emerald positive state;
- amber warning / proximity state;
- rose-red negative / invalid state;
- thin low-contrast borders;
- subtle gradients and glows only where they improve hierarchy;
- no decorative sci-fi effects;
- no gratuitous glassmorphism;
- dense but breathable information architecture.

The product should look more like a serious investment workstation than a generic SaaS dashboard.

### 3.2 Typography

Use a two-family hierarchy if the existing stack allows it without fragile dependencies:
- display / editorial face for major page titles only;
- neutral sans-serif for all operational UI and numerical data.

Fallback is acceptable if adding a new font complicates performance or licensing.

Requirements:
- tabular numerals for prices and percentages when practical;
- strong visual distinction between labels and values;
- avoid tiny text below 12px for important information;
- primary body text generally 14px minimum on desktop.

### 3.3 Spacing system

Use a consistent 4px base grid:
- 4 / 8 / 12 / 16 / 20 / 24 / 32 / 40 / 48.
- standard card padding: 16–20px;
- major panel padding: 20–24px;
- page vertical rhythm: 24–32px.

### 3.4 Shape language

- Cards: 12–16px radius.
- Inputs/buttons: 8–10px radius.
- Borders: 1px subtle slate/teal.
- Avoid excessive shadows.
- Hover state should be visible through border/background lift, not movement-heavy animation.

## 4. Design tokens

Create or normalize reusable semantic tokens in Tailwind/CSS.

### Surfaces
- app background
- sidebar background
- panel background
- panel elevated
- interactive hover
- active row / selected company

### Text
- primary
- secondary
- muted
- disabled

### Semantic states
- accent
- success
- warning
- danger
- info
- neutral

### Financial semantic states

Use semantic meaning consistently:
- O90 reached / actionable: emerald
- within 5%: teal
- 5–10%: amber
- 10–20%: orange
- >20%: muted/slate or restrained orange
- uncalibrated: neutral grey
- stale data: amber
- missing/critical data: rose

Never rely on color alone. Each state also needs text/badge/icon.

## 5. Global application shell

### Desktop structure

Persistent left sidebar + top command bar + content canvas.

#### Sidebar
Sections:
- Dashboard
- Screener
- Sociétés
- Analyses or current equivalent if route exists
- Data Health
- Admin

Requirements:
- clear active state;
- compact icon + label;
- OroTitan logo/wordmark at top;
- restrained brand statement at bottom;
- no fabricated modules/routes. Only expose implemented routes.

#### Top command bar
Must contain:
- global search affordance or existing route-level search entry;
- data freshness indicator;
- sync state;
- admin/user affordance where relevant.

Do not show fake real-time status. Use actual available application state only.

### Responsive
- desktop >= 1280: full sidebar + right detail panel where applicable;
- tablet: collapsed sidebar / reduced side panel;
- mobile: stacked layout, drawer navigation, no unusable horizontal dashboard compression.

## 6. Dashboard `/`

### Goal
Answer in under five seconds:
1. What is actionable now?
2. How many companies are calibrated?
3. How fresh is the data?
4. Which companies are closest to O90?

### Header
Title:
`Cockpit OroTitan`

Subtitle:
short statement about versioned analysis, market prices and entry discipline.

### KPI row
Use actual current V1.1 metrics:
- active companies;
- O90 calibrated;
- O90 reached;
- fresh market data;
- optional Data Health issue count if it improves clarity.

Cards should contain:
- short label;
- large value;
- one small contextual line;
- no decorative chart unless backed by actual data.

### Priorities
Primary visual block:
`Priorités du moment`

Sort:
1. O90 reached first;
2. then closest from above;
3. uncalibrated excluded from distance ordering.

Show 5–6 companies.

Each priority card/row:
- rank;
- company + ticker;
- price;
- O90;
- distance;
- OroTitan score;
- zone badge;
- click-through to company page.

### Entry radar / distribution
Prefer a clean bucket distribution over a decorative radar chart unless the radar materially improves reading.

Buckets:
- O90 atteint
- <5%
- 5–10%
- 10–20%
- >20%
- Non calibré

Every bucket count must come from live state.

### Screener preview
Show a small subset of the screener with:
- company;
- price;
- fair value;
- upside FV;
- score;
- O90;
- distance;
- date analysis.

Provide `Voir le screener`.

## 7. Screener `/screener`

This is the highest-priority UX surface.

### 7.1 Top structure
- page title + one-line explanation;
- sticky filter / command area on desktop where practical;
- compact result count.

### 7.2 Filter system
Preserve all current V1.1 filters:
- search;
- status;
- quality OroTitan;
- entry zone;
- country;
- sector;
- O90 calibration;
- market freshness;
- minimum score;
- minimum distance;
- maximum distance;
- secondary sort.

Improve presentation:
- group common filters in the first row;
- move advanced numeric filters into a `Plus de filtres` area if this reduces noise;
- active filters should be visible as removable chips where practical;
- Reset restores filters AND sorting defaults.

### 7.3 Table
Core columns:
- Société
- Ticker
- Statut
- Cours
- Fair value centrale
- Upside FV
- Score
- O90
- Distance O90
- Zone
- Analyse

Rules:
- no raw cross-currency Fair Value sort;
- `Upside FV` is the normalized sortable valuation metric:
  `(fair_value_base / price - 1) * 100`;
- NULL remains at bottom of numeric sorts;
- default sort remains Distance O90 descending;
- default secondary sort remains Score descending;
- row is clickable and keyboard accessible;
- selected/hovered row gets a subtle premium highlight;
- sticky table header where technically clean.

### 7.4 Density
Offer one default density only for V1.2 unless a second density is trivial.
Default should be compact-professional, not cramped.

### 7.5 Optional desktop detail panel
A selected company may open a right-side contextual panel without leaving the screener, but only if it can be implemented cleanly without duplicating business logic.

Minimum content:
- company;
- price;
- O90;
- distance;
- score;
- fair value range;
- thesis excerpt;
- link to full company page.

If this introduces significant complexity, defer it and keep the click-through interaction.

## 8. Company page `/company/[slug]`

### Goal
A premium one-page investment brief.

### Header
Display:
- company name;
- ticker;
- exchange;
- country;
- sector;
- status;
- OroTitan quality;
- entry-zone badge.

### Current state panel
High-priority numbers:
- current price;
- O90;
- distance O90;
- OroTitan score;
- source;
- freshness;
- price timestamp.

### Valuation
Show:
- FV low;
- FV base;
- FV high;
- Upside FV base;
- O85;
- O90;
- O92;
- O95.

O90 gets the strongest visual emphasis.

### Market history chart
Keep current market_prices-only rule.

Improve:
- legible legend;
- current price marker;
- O90/O92/O95 threshold labels;
- helpful empty/short-history state;
- no fake extrapolation.

### Analysis
Three equal semantic blocks:
- Thesis
- Main risk
- Invalidation

Then:
- source;
- model version;
- analysis date;
- notes.

### Score components
Present JSON-backed values as a clean metric grid.
Do not assume a universal schema.

### Snapshot history
Use a clean version history table.
Highlight current snapshot.
If previous comparable snapshot exists, show deltas.

## 9. Admin

Admin must look consistent with the product but remain operational rather than decorative.

### Admin shell
Tabs/routes:
- Overview
- Companies
- New snapshot
- Prices
- Data Health

### Forms
- clear sections;
- strong label hierarchy;
- inline validation;
- explicit destructive/immutable warnings;
- preserve current admin authorization.

### Price sync
Show:
- provider;
- result count;
- failures;
- last sync;
- status badge;
- cooldown feedback;
- journal entries.

Successful sync = green.
Partial = amber.
Failure = rose.

### Data Health
Prioritize issues by severity and actionability.
Do not add a fake severity model without explicit domain rules.

## 10. Reusable components to establish

Prefer reusable components rather than page-specific duplicated Tailwind strings.

Recommended component set:
- AppShell
- Sidebar
- TopBar
- PageHeader
- Panel
- MetricCard
- StatusBadge
- EntryZoneBadge
- DataFreshnessBadge
- ScoreBadge
- PriceMetric
- DistanceMetric
- EmptyState
- AlertBanner
- FilterBar
- FilterChip
- DataTableShell
- SectionHeader
- CompanySummaryCard
- ValuationLadder
- SyncStatus
- SkeletonCard / SkeletonTable

Reuse existing V1.1 components when practical.

## 11. Motion and micro-interactions

Allowed:
- 120–200ms background/border transitions;
- subtle button hover;
- smooth tab underline/active state;
- skeleton loading;
- button pending state;
- row selection state.

Avoid:
- bouncing cards;
- constant animated charts;
- large parallax;
- distracting number animations.

Respect `prefers-reduced-motion`.

## 12. Loading, errors and empty states

Every data-heavy page should have polished states.

### Loading
Use skeletons instead of page jumps when client-side transitions need them.

### Error
Must explain:
- what failed;
- what remains safe;
- retry action where relevant.

Never substitute invented financial values.

### Empty
Examples:
- no company matches filters;
- no previous snapshot;
- short/no price history;
- no sync history.

## 13. Accessibility

Minimum:
- keyboard navigation;
- visible focus state;
- contrast compliant for operational text;
- badges not color-only;
- buttons have accessible names;
- row navigation supports Enter;
- table headers semantically correct;
- form labels associated with controls.

## 14. Performance

Do not make the UI refactor materially degrade the current app.

Targets:
- avoid unnecessary client components;
- keep Server Components by default;
- isolate interactivity;
- no heavyweight chart dependency unless required;
- preserve force-dynamic behavior where live data requires it;
- avoid hydration for purely presentational blocks.

## 15. Implementation sequence

### P1 — Foundations
- global tokens;
- AppShell;
- sidebar;
- top bar;
- Panel / MetricCard / section primitives;
- semantic states.

### P2 — Screener
- filter bar;
- table styling;
- sticky behavior;
- selected/hover states;
- improved mobile/tablet handling;
- preserve all current logic.

### P3 — Dashboard
- premium header;
- KPI row;
- priorities;
- entry distribution;
- screener preview.

### P4 — Company page
- header/current state;
- valuation ladder;
- chart polish;
- analysis cards;
- snapshot history.

### P5 — Admin
- visual consistency;
- forms;
- price sync;
- Data Health.

### P6 — Polish
- responsive audit;
- keyboard/focus audit;
- loading/empty/error states;
- copy consistency;
- final visual pass.

## 16. Acceptance criteria

V1.2 is considered UX-complete only if:

1. Existing V1.1 business and data tests still pass.
2. No analytical value is invented.
3. Market-price sync still works 8/8 for the current bootstrap universe.
4. Dashboard metrics match database state.
5. Screener distance/O90 calculations remain exact.
6. Screener reset restores filters and default sorts.
7. Upside FV is displayed/sorted cross-currency instead of raw FV sorting.
8. Company pages preserve market/snapshot separation.
9. Admin writes remain authenticated.
10. All routes are usable on desktop and mobile.
11. Focus states and keyboard navigation are visible.
12. Lint, typecheck, tests and production build pass.
13. No new runtime dependency is added without a clear UX need.
14. No regression of V1.1 append-only guarantees.
15. Visual implementation clearly matches the approved dark premium cockpit direction.

## 17. Explicit non-goals

Do not add in V1.2 unless separately approved:
- portfolio management;
- brokerage execution;
- news ingestion;
- analyst consensus;
- macro feeds;
- AI-generated recommendations;
- watchlist backend;
- new financial scoring formulas;
- authentication redesign;
- database redesign.

V1.2 is a disciplined UX/UI release.
