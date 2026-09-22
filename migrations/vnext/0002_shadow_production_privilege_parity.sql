-- OroTitan VNext shadow bootstrap.
-- Fresh Supabase projects grant broader service_role table/view privileges
-- than the live OroTitan production baseline. This migration narrows the
-- shadow privilege surface to exact production parity.
--
-- No production project is referenced or mutated by this migration.

revoke delete on table public.companies from service_role;
revoke delete on table public.issuers from service_role;

revoke delete, insert, update on table public.latest_company_state from service_role;
revoke delete, insert, update on table public.orotitan_run_status_view from service_role;

revoke delete, update on table public.legacy_company_identity_map from service_role;
revoke delete, update on table public.market_prices from service_role;
revoke delete, update on table public.market_sync_runs from service_role;
revoke delete, update on table public.snapshots from service_role;

revoke delete on table public.research_dossiers from service_role;
revoke delete on table public.securities from service_role;
