# P0 Screener — Wise WSE / ISRG Data Health, 2026-10-22

**Status: AUDITED PROPOSAL — NOT APPLIED — Requires separate production-data GO.**
Screener production is `orotitan-screener`, GitHub main
`86a26f44cd576b40d37a385eb3623bbeb526c423`. D29b and VNext are excluded.

## Confirmed from read-only Supabase checks

Project `cugpgtzygqqlxetyeven`: 34 active canonical published dossiers.
Exactly **two** published dossier/security quadruplets have **zero**
`legacy_company_identity_map` mapping (the 32-row audited 2026-10-11 cohort
intentionally excluded these two):
- Wise Group plc — issuer + historical company
  `185329c2-c577-4461-9954-6ecfd75c7483`, security
  `cc46c84b-a073-448a-8109-fea143605ac0`, dossier
  `b17d1bd3-6a7b-4b89-bbe3-b7022a36b651`, published snapshot
  `06357a6b-e883-4d54-a68a-193d4658619f`, ISIN **JE00BQKY0816**.
  Authoritative security ticker WSE, venue Nasdaq, USD, MAJOR, multiplier 1.
  Its canonical `securities.market_data_symbol` is **NULL**, despite
  `companies.market_data_symbol=WSE:NASDAQ`.
- Intuitive Surgical — issuer + historical company
  `9748c01b-ec84-48dd-b5a5-3dafd0750d89`, security
  `c459062e-baa6-43ec-b75e-86916b805e90`, dossier
  `2796e551-4241-4ae8-b6e1-135f9b653767`, published snapshot
  `c7585971-fdde-4f5b-9dd7-b4778e022063`, ISIN **US46120E6023**.
  Canonical `securities.market_data_symbol=ISRG` is a valid Nasdaq
  reference under existing code; legacy `companies.market_data_symbol=ISRG:NASDAQ`
  is also valid. Neither should be arbitrarily rewritten.

Latest stored verified-symbol observations, `YAHOO_FINANCE`,
`raw.symbol=WSE|ISRG`, `raw.currency=USD`, `raw.exchange=NMS`:
- Wise **11.05 USD** as of **2026-10-21 20:00:00 UTC**.
- ISRG **428.12 USD** as of **2026-10-21 20:00:01 UTC**.
The `market_sync_runs` CRON #30 at 2026-10-21 23:08 UTC had
**39/39 inserted, 0 failures**, including these two.
Current published snapshot reference prices are **11.53 USD Wise** and
**393.33 USD ISRG** with reference date 2026-09-18; these stale reference
prices remain selected because `resolveCanonicalQuoteCompanyId` fails
closed on absent mapping. They must NOT be overwritten.

External authoritative evidence (not a substitute for database identity check):
- Wise's May 2026 scheme explicitly confirmed Nasdaq ticker **WSE**;
  `Wise Group plc` is not a mistaken synonym of the old LSE quote.
  https://www.lse.co.uk/rns/scheme-of-arrangement-becomes-effective-b01xh7vgzhauwuq.html
- Intuitive confirms Nasdaq **ISRG**:
  https://investor.intuitivesurgical.com/

## Manual remediation after explicit GO

The *non-migration* file
`docs/orotitan-equity/proposals/P0_WISE_ISRG_CROSSWALK_STAGED_GO_REQUIRED.sql`
contains a fail-closed, atomic SQL transaction:
1. Lock down the precise TWO audited (legacy, issuer, security, dossier)
   quadruplets, published snapshots, ticker, ISIN, exchange, USD, quote units,
   multipliers, schema state and recent provider evidence.
2. Reject crosswalk ownership conflicts or changes to the 34-dossier cohort.
3. Update only Wise `securities.market_data_symbol` from NULL to
   `WSE:NASDAQ`, asserting exactly one row updated.
4. Insert only the TWO audited rows in `legacy_company_identity_map`.
5. Enforce both crosswalk postconditions; rollback entire transaction on any error.

No legacy historical price row, published snapshot, issuer, security identity,
register, immutable artifact, published pointer, Supabase role or RLS policy is
updated. The `securities` modification is **new controlled metadata** and
requires explicit authorization. **Do not change the frozen 2026-10-11 migration
or its exclusions.** Do not insert by ticker matching or legacy UUID coincidence.

## Verification after separately authorized data apply

- Re-query exactly two identity mappings; assert other crosswalks unaffected.
- Read each published dossier via the production path, not an admin backdoor.
- Assert latest admitted prices 11.05 and 428.12 (or later verified USD/Nasdaq
  quotes), `price_as_of` current and both symbols canonical.
- `Data Health` should lose exactly 3 issues, provided no new staleness or
  other issues arise. Do not fabricate zero warnings without live verification.
- Confirm the paper portfolio still loads the same distinct audited identity,
  as it also uses this crosswalk.
- Rollback via the approved correction procedure if any unexpected impact.
- No code deployment should be needed for these 3 precise data issues.

If the operator does not separately approve this data mutation, retain the
current fail-closed warnings and do not bypass them in the UI.
