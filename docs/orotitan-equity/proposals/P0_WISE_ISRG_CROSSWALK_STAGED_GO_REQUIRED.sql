-- STAGED PROPOSAL ONLY — NEVER AUTO-APPLY.
-- OroTitan main Screener P0: 2026-10-22 Wise (WSE) / Intuitive Surgical (ISRG).
-- Requires a separate, explicit production data-change GO and backup/review.
-- Two previously EXCLUDED canonical-to-legacy identities are now independently
-- audited. This does not amend or relax the frozen 2026-10-11 cohort.
-- No changes to immutable research snapshots, registry, market_prices or RLS.
BEGIN;

CREATE TEMPORARY TABLE _p0_quote_links_20261022 ON COMMIT DROP AS
SELECT * FROM (VALUES
  ('185329c2-c577-4461-9954-6ecfd75c7483'::uuid, -- Wise legacy+issuer
   '185329c2-c577-4461-9954-6ecfd75c7483'::uuid,
   'cc46c84b-a073-448a-8109-fea143605ac0'::uuid,
   'b17d1bd3-6a7b-4b89-bbe3-b7022a36b651'::uuid,
   '06357a6b-e883-4d54-a68a-193d4658619f'::uuid,
   'WSE'::text, 'WSE:NASDAQ'::text, 'JE00BQKY0816'::text, NULL::text),
  ('9748c01b-ec84-48dd-b5a5-3dafd0750d89'::uuid, -- ISRG legacy+issuer
   '9748c01b-ec84-48dd-b5a5-3dafd0750d89'::uuid,
   'c459062e-baa6-43ec-b75e-86916b805e90'::uuid,
   '2796e551-4241-4ae8-b6e1-135f9b653767'::uuid,
   'c7585971-fdde-4f5b-9dd7-b4778e022063'::uuid,
   'ISRG'::text, 'ISRG:NASDAQ'::text, 'US46120E6023'::text, 'ISRG'::text)
) AS p(legacy_company_id,issuer_id,security_id,dossier_id,snapshot_id,
       ticker,provider_reference,isin,existing_canonical_symbol);

DO $$
BEGIN
  IF (SELECT count(*) FROM _p0_quote_links_20261022) <> 2
     OR (SELECT count(*) FROM public.research_dossiers
          WHERE active IS TRUE AND current_snapshot_id IS NOT NULL) <> 34
  THEN RAISE EXCEPTION 'P0_QUOTE_PINNED_COHORT_DRIFT'; END IF;

  -- Assert all four canonical owners, exact currently published snapshot,
  -- authoritative venue/currency/unit/ISIN, and a single matching legacy row.
  IF EXISTS (
    SELECT 1 FROM _p0_quote_links_20261022 p
    LEFT JOIN public.research_dossiers d ON d.dossier_id=p.dossier_id
    LEFT JOIN public.research_snapshots snap ON snap.snapshot_id=p.snapshot_id
    LEFT JOIN public.securities s ON s.security_id=p.security_id
    LEFT JOIN public.companies c ON c.id=p.legacy_company_id
    WHERE d.dossier_id IS NULL OR snap.snapshot_id IS NULL
       OR s.security_id IS NULL OR c.id IS NULL
       OR d.issuer_id IS DISTINCT FROM p.issuer_id OR d.active IS DISTINCT FROM TRUE
       OR d.current_snapshot_id IS DISTINCT FROM p.snapshot_id
       OR snap.dossier_id IS DISTINCT FROM p.dossier_id
       OR snap.issuer_id IS DISTINCT FROM p.issuer_id
       OR snap.security_id IS DISTINCT FROM p.security_id
       OR s.issuer_id IS DISTINCT FROM p.issuer_id
       OR s.isin IS DISTINCT FROM p.isin
       OR s.ticker IS DISTINCT FROM p.ticker
       OR lower(s.exchange) IS DISTINCT FROM 'nasdaq'
       OR s.trading_currency IS DISTINCT FROM 'USD'
       OR s.quote_unit IS DISTINCT FROM 'MAJOR'
       OR s.market_data_multiplier IS DISTINCT FROM 1
       OR s.price_decimals IS DISTINCT FROM 2
       OR s.market_data_symbol IS DISTINCT FROM p.existing_canonical_symbol
       OR c.id IS DISTINCT FROM p.issuer_id
       OR c.active IS DISTINCT FROM TRUE
       OR c.ticker IS DISTINCT FROM p.ticker
       OR lower(c.exchange) IS DISTINCT FROM 'nasdaq'
       OR c.currency IS DISTINCT FROM 'USD'
       OR c.quote_unit IS DISTINCT FROM 'MAJOR'
       OR c.market_data_symbol IS DISTINCT FROM p.provider_reference
       OR c.market_data_multiplier IS DISTINCT FROM 1
       OR c.price_decimals IS DISTINCT FROM 2
  ) THEN RAISE EXCEPTION 'P0_QUOTE_FOUR_PART_IDENTITY_NOT_PROVEN'; END IF;

  -- Never attach a different security or overwrite any approved crosswalk.
  IF EXISTS (
    SELECT 1 FROM _p0_quote_links_20261022 p
    JOIN public.legacy_company_identity_map m
      ON m.legacy_company_id=p.legacy_company_id
       OR m.security_id=p.security_id
       OR m.dossier_id=p.dossier_id
       OR m.issuer_id=p.issuer_id
  ) THEN RAISE EXCEPTION 'P0_QUOTE_CROSSWALK_ALREADY_ASSIGNED_REVIEW'; END IF;

  -- Verify the recent provider feed identifies the exact US security;
  -- no same-ticker assumptions, cross-venue fallback or synthetic quote.
  IF EXISTS (
    SELECT 1 FROM _p0_quote_links_20261022 p
    WHERE NOT EXISTS (
      SELECT 1 FROM public.market_prices mp
      WHERE mp.company_id=p.legacy_company_id
        AND mp.source='YAHOO_FINANCE'
        AND mp.price > 0
        AND mp.as_of <= now() + interval '5 minutes'
        AND mp.as_of > now() - interval '96 hours'
        AND mp.raw->>'symbol'=p.ticker
        AND mp.raw->>'currency'='USD'
        AND mp.raw->>'exchange' IN ('NMS','NGM','NCM','NAS','NMQ')
    )
  ) THEN RAISE EXCEPTION 'P0_QUOTE_RECENT_FEED_IDENTITY_NOT_PROVEN'; END IF;
END $$;

-- One approved security-metadata repair: Wise canonical symbol was NULL.
-- Do not change ISRG's valid 'ISRG' Nasdaq alias or historical market rows.
DO $
DECLARE rows_updated integer;
BEGIN
  UPDATE public.securities
    SET market_data_symbol='WSE:NASDAQ'
    WHERE security_id='cc46c84b-a073-448a-8109-fea143605ac0'::uuid
      AND issuer_id='185329c2-c577-4461-9954-6ecfd75c7483'::uuid
      AND ticker='WSE' AND isin='JE00BQKY0816'
      AND market_data_symbol IS NULL;
  GET DIAGNOSTICS rows_updated = ROW_COUNT;
  IF rows_updated <> 1 THEN
    RAISE EXCEPTION 'P0_WISE_CANONICAL_SYMBOL_UPDATE_FAILED';
  END IF;
END $;

-- Exactly two audited, new crosswalk rows. No inferred UUID/ticker fallback.
INSERT INTO public.legacy_company_identity_map
  (legacy_company_id,issuer_id,security_id,dossier_id)
SELECT legacy_company_id,issuer_id,security_id,dossier_id
FROM _p0_quote_links_20261022;

DO $$
BEGIN
  IF (SELECT count(*) FROM _p0_quote_links_20261022 p
       JOIN public.legacy_company_identity_map m
         ON m.legacy_company_id=p.legacy_company_id
        AND m.issuer_id=p.issuer_id
        AND m.security_id=p.security_id
        AND m.dossier_id=p.dossier_id) <> 2
  THEN RAISE EXCEPTION 'P0_QUOTE_CROSSWALK_POSTCONDITION_FAILED'; END IF;
END $$;

COMMIT;
