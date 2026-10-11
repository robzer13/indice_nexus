# OroTitan Method-V2 Pre-Certification Coverage Evidence — Successor V1.1

STATUS = FROZEN
AUTHORIZATION = METHOD-V2-COVERAGE-EVIDENCE-SUCCESSOR-001
PRODUCTION_ACTIVE = NO
BASE_AUTHORITY = OROTITAN_PRE_CERTIFICATION_CHALLENGE_FREEZE_V1.0.md
BASE_AUTHORITY_SHA256 = 6190078e532626d80eaf994e806d4461eece3a7daa3f62cbdd0e908b1f2a654a

## 1. Exact supersession scope

This explicit successor replaces only the coverage evidence representation in
base Sections 3, 5(1–2) and 8, and clarifies negative-disposition reconciliation
in Section 6. Every other base clause remains controlling. V1.0 bytes and its
authority-set identity remain historical and must never be rebound silently.
The V1.1 analytical manifest binds both the base and this scoped successor;
V1.1 ledger/report schemas supersede the V1.0 artifact shapes for that identity.
Both schema_version and challenge_version are 1.1 for the current artifacts.

## 2. Mandatory-family coverage evidence

The report saturation_record.family_coverage contains exactly one disposition
for each of the eight mandatory families defined by the base. No family may be
omitted or resolved by a boolean or rationale alone. Each disposition is exactly
one of:

- QUESTION_COVERED: question_ids contains one or more distinct executed question
  IDs from the effective current ledger, including carried questions in DELTA.
  Every referenced question must exist, have the same question_family as the
  disposition, and have company_specific = true. Dropped candidates cannot cover
  a family. Existing question judgments, materiality and evidence requirements
  remain applicable.
- EVIDENCED_NO_MATERIAL_CHALLENGE: a non-whitespace rationale explains why the
  family adds no material challenge to this company's decision. One or more
  structured evidence_references must each identify an evidence_id in the same
  authoritative Evidence Ledger, its exact evidence_ledger_ref (artifact_id,
  version, content_sha256), and evidence_date. Every date must be <= DATA_CUTOFF.
  A free-form string is not a structured evidence reference. The consuming
  trusted artifact resolver owns evidence-reference resolution and provenance,
  as it owns upstream Evidence Ledger lineage under the base Section 8.

The overall company-specific challenge must include at least one actually
executed effective-ledger question with company_specific = true. Exemptions
cannot replace the entire Challenge. Report company_specific_questions must
still reconcile exactly to those executed question IDs. Existing coverage flags
and rationale fields are retained but cannot authorize admission by themselves.

These are coverage conditions, not a universal question-count quota. There is
no new minimum or maximum total question count, score, weight or pass-rate gate.

## 3. Negative dispositions

The effective ledger remains controlling for non-compensatory aggregation:
FAIL before REOPEN before CONCERN before PASS. REOPEN and FAIL never admit
Certification. PASS or PASS_WITH_CONCERNS additionally requires reopen_required
and fail_reasons both to be empty arrays. A retained negative disposition cannot
be disregarded by positive ledger statuses, coverage flags or saturation claims.
Resolved earlier versions remain immutable history, outside current negative
dispositions; no report-string parsing may infer a resolution.

## 4. Preserved boundaries

Material saturation, all concern dispositions, at least two distinct independent
zero-yield final passes over the whole current conclusion, exact current lineage,
cutoff discipline, DELTA revalidation and limitation propagation are unchanged.
Scoring, Certification formulas, terminal conjunction, historical METHOD_V1
classification and all other analytical rules are unchanged. This successor
performs no production activation, Supabase operation, migration or V2-11A work.
