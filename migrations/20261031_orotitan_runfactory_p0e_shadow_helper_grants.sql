-- P0-E SHADOW forward patch: permit service_role to invoke ONLY
-- the two immutable/read-only pinned-contract predicates.
-- No anon/authenticated privilege change; no run or historical rewrite.
begin;
grant execute on function public.orotitan_contract_pins_complete(jsonb) to service_role;
grant execute on function public.orotitan_contract_set_sha256(jsonb) to service_role;
commit;
