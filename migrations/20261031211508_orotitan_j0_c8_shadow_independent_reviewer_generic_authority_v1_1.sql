-- OroTitan C8 SHADOW hardening v1.1 — Never execute on production.
-- Does not enable any gate, accept any milestone, create a run, or publish.
-- Source: independently reviewed J0 C8 v1 code, 2026-10-31.
BEGIN;
DO $preflight$
BEGIN
  IF EXISTS (SELECT 1 FROM public.orotitan_j0_c8_acceptance_control_v1
             WHERE enabled OR activation_authority_ref IS NOT NULL)
     OR EXISTS (SELECT 1 FROM public.orotitan_j0_c8_milestone_activation_v1
                WHERE enabled OR authority_ref IS NOT NULL)
     OR EXISTS (SELECT 1 FROM public.orotitan_j0_c8_decisions_v1)
  THEN
    RAISE EXCEPTION 'C8_SHADOW_HARDENING_REQUIRES_ALL_GATES_DISABLED';
  END IF;
  IF (SELECT COUNT(*) FROM public.orotitan_j0_c8_milestone_activation_v1) <> 9
  THEN
    RAISE EXCEPTION 'C8_SHADOW_EXPECTED_NINE_MILESTONE_LOCKS';
  END IF;
END $preflight$;

-- Allow future GO_ACCEPT_Jn_ references for J0..J8, without enabling any GO.
-- Each per-milestone activation must still match that exact Jn reference.
ALTER TABLE public.orotitan_j0_c8_acceptance_control_v1
  DROP CONSTRAINT j0_c8_disabled_without_authority;
ALTER TABLE public.orotitan_j0_c8_acceptance_control_v1
  ADD CONSTRAINT j0_c8_disabled_without_authority CHECK (
    (enabled = false AND activation_authority_ref IS NULL)
    OR (enabled = true AND activation_authority_ref ~ '^GO_ACCEPT_J[0-8]_[A-Z0-9_]{8,120}$')
  );

-- Harden the authoritative acceptance RPC, rather than trusting only a UI label.
-- NOTE: this verifies distinct DECLARED labels, not strong named SSO identities.
DO $patch$
DECLARE
  v_source text;
  v_needle text := E' RAISE EXCEPTION ''J0_C8_REVIEW_NOT_FOUND_OR_HASH_MISMATCH'';END IF;';
  v_guard text := E'\n IF lower(btrim(v_review.reviewer_label))=lower(btrim(p_approver_label)) THEN\n  RAISE EXCEPTION ''J0_C8_REVIEWER_APPROVER_NOT_DISTINCT'';END IF;';
BEGIN
  v_source := pg_get_functiondef('public.orotitan_j0_c8_accept_review_v1(text,uuid,text,text,text,text)'::regprocedure);
  IF position('J0_C8_REVIEWER_APPROVER_NOT_DISTINCT' IN v_source) > 0 THEN
    RAISE NOTICE 'C8 reviewer/approver guard already installed';
  ELSE
    IF position(v_needle IN v_source) = 0 THEN
      RAISE EXCEPTION 'C8_EXPECTED_RPC_ANCHOR_MISSING';
    END IF;
    v_source := replace(v_source,v_needle,v_needle||v_guard);
    EXECUTE v_source;
  END IF;
END $patch$;
COMMIT;
