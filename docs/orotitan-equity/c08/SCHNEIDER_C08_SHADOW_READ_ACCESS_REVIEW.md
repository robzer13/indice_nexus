# OroTitan Schneider C08 — SHADOW restricted reader extension (REVIEW ONLY)

**Status:** PROPOSED / NOT APPLIED. **Target:** Supabase SHADOW `awgsurdyvsyolcgpnygh`.
**PRODUCTION:** `cugpgtzygqqlxetyeven` is strictly excluded.
**No live mutation, no deployment, no PG17 replay, no Fundamentals V2 admission is authorized by this PR.**

## Verified baseline (2026-11-17, metadata-only)
C07 protected preflight run #43972795066 has PASS provenance 19/19, candidates 3/3 and
read-only snapshot 5 Registry tables, with local bytes deleted after completion.

The Schneider run `db00ce77-04eb-492b-a76b-164ecbe9ccfe` currently joins to exactly
one row each in `public.issuers`, `public.securities`,
`public.research_dossiers`, and `public.orotitan_p0e_run_admission_receipts_v1`.
The dedicated `orotitan_c07_reader` LOGIN has no SELECT on these four tables;
all four have RLS enabled and currently no policies. The role has SELECT for
the single Schneider run in `public.orotitan_runs`, no outgoing memberships,
no SUPERUSER/BYPASSRLS and no CREATE privilege. There is exactly **one incoming
administrative role membership**: `postgres` with `admin_option=true`,
`inherit_option=false`, and `set_option=false`; no other member may gain
the new read access. These facts can change; the script must check both
membership directions and all grant options immediately before the transaction.

## Proposal and strict review gate
The file `schneider-c08-shadow-reader-grants.REVIEW_ONLY.sql` is an **operator-only,
non-migration proposal**. It MUST NOT be executed as part of CI or on a PR runner.
It adds only four SELECT grants and four PERMISSIVE SELECT policies, each
scoped to row identities that match the existing RLS-limited Schneider run.
The policies contain no SECURITY DEFINER functions, no bypass-RLS, no broad
grant to PUBLIC or to authenticated/anon and no mutation permissions.
The executable grant SQL is additionally pinned to the verified SHADOW
cluster fingerprint; this prevents accidental use on another cluster even
when the same names or run rows exist, but does not authorize bypassing
the dashboard/project-URL check.

**Before applying:**
1. Get independent code/security review and approval of the exact commit.
2. Verify from the Supabase dashboard that the selected project ref is
   `awgsurdyvsyolcgpnygh` (not the PRODUCTION project). The SQL
   independently verifies the pinned SHA-256 of the SHADOW PostgreSQL
   `pg_control_system().system_identifier` plus project ref, recorded through
   a read-only SHADOW metadata call on 2026-11-17. If a disaster recovery,
   rebootstrap or database migration changes this identifier, fail closed and
   independently re-attest the correct project; never delete the check or
   substitute an unverified fingerprint.
3. Reconfirm all four target tables have no conflicting RLS policies/grants;
   confirm run CAS 7 and its four matching parent/receipt rows still exist.
4. Approve the security change explicitly and run the reviewed SQL **once**,
   manually using an authorized SHADOW administrator, never via GitHub Actions.
5. Run `schneider-c08-shadow-reader-verify.READ_ONLY.sql` with an **actual
   protected pooler login as `orotitan_c07_reader`**, not as `postgres`
   (the existing admin membership has `set_option=false`; `SET ROLE` is not
   permitted). The SQL requires `current_user=session_user=orotitan_c07_reader`,
   executes only inside `BEGIN ... READ ONLY` and prints one success guard,
   not identities/receipts. Execute using a separately reviewed protected
   manual GitHub Actions workflow with the existing environment secret.
6. Confirm that INSERT/UPDATE/DELETE/TRUNCATE/EXECUTE remain unavailable to
   the dedicated reader. Role metadata checks are not a substitute for a
   successful reader-authenticated runtime check.

**Rollback, only after review and with explicit SHADOW-only approval:**
```sql
BEGIN;
REVOKE SELECT ON TABLE public.issuers,
  public.securities, public.research_dossiers,
  public.orotitan_p0e_run_admission_receipts_v1 FROM orotitan_c07_reader;
DROP POLICY c08_schneider_reader_select ON public.issuers;
DROP POLICY c08_schneider_reader_select ON public.securities;
DROP POLICY c08_schneider_reader_select ON public.research_dossiers;
DROP POLICY c08_schneider_reader_select ON public.orotitan_p0e_run_admission_receipts_v1;
COMMIT;
```

## Remaining C08 work (separate approval)
After proving the reader's four new scoped SELECT grants, extend the **protected
manual** C07 snapshot with exactly these four tables **in the same**
`BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY` context.
No new public artifacts or raw SQL output. Then, in the same ephemeral protected
job, build a local PG17 Registry, replay the authentic CAS 7/2 → 8/3 against
real immutable bytes, and verify the trusted V3.1 loader after local COMMIT.
The current PR explicitly does **none** of this.

## Boundaries
No private rows or credentials in GitHub or public logs; no changes to C07
secrets, Supabase PRODUCTION or frozen contracts; no remote C08 write.
`C08_READER_EXTENSION=NOT_APPLIED`, `PG17_REPLAY=NOT_EXECUTED`,
`FUNDAMENTALS_V2=NO_GO`.
