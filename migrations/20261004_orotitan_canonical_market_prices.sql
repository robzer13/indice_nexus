-- OroTitan canonical market-price overlay.
-- Additive only: certified research_snapshots remain immutable.
-- Current market observations are keyed by canonical security_id.

create table if not exists public.security_market_prices (
  id bigint generated always as identity primary key,
  security_id uuid not null references public.securities(security_id) on delete restrict,
  price numeric not null check (price > 0),
  as_of timestamptz not null,
  source text not null,
  raw jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_security_market_prices_security_asof
  on public.security_market_prices(security_id, as_of desc);

alter table public.security_market_prices enable row level security;

revoke all on table public.security_market_prices from anon, authenticated;
grant select, insert on table public.security_market_prices to service_role;
grant usage, select on sequence public.security_market_prices_id_seq to service_role;

create or replace function public.prevent_security_market_price_mutation()
returns trigger
language plpgsql
as $$
begin
  raise exception 'security_market_prices is append-only';
end;
$$;

alter function public.prevent_security_market_price_mutation() set search_path = pg_catalog;
revoke all on function public.prevent_security_market_price_mutation() from public, anon, authenticated;

drop trigger if exists security_market_prices_append_only on public.security_market_prices;
create trigger security_market_prices_append_only
before update or delete on public.security_market_prices
for each row execute function public.prevent_security_market_price_mutation();

-- Market symbols for securities currently referenced by published canonical snapshots.
update public.securities set market_data_symbol='ADYEN:AMS' where security_id='b77eedc2-13f6-4faf-a4fc-17e09e10ff05'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='AI:EPA' where security_id='925f910f-08fb-4431-bf54-81b07295fe8c'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='GOOGL' where security_id='247e7f79-7a34-4e3d-9542-c217274526e9'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='ASM:AMS' where security_id='b522fe6b-9901-49a5-b3da-42b93a4ec8de'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='BEAN:SWX' where security_id='b127e1fc-ecff-4f4e-ac32-35f74663da3f'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='BKNG' where security_id='05e580c2-96e5-47d3-812a-2d251eac29bd'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='BN:TSX' where security_id='7006178d-1704-4c62-b0dd-971f3407016a'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='CSU:TSX' where security_id='c2bf1235-4ed6-4f10-ac47-eb6853f261ef'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='CPRT' where security_id='894f452d-703f-4d79-a3a5-1bc25aafb304'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='D05:SGX' where security_id='6d44adba-a1a7-4ac2-8283-ae73d3b0f377'::uuid;
update public.securities set market_data_symbol='EQIX' where security_id='2f474fa7-0cc0-4f72-8ab9-7432f28f251d'::uuid;
update public.securities set market_data_symbol='EL:EPA' where security_id='5ff7e189-30f1-478d-8985-00b4c2dd79b5'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='GRAB' where security_id='951b0718-d3fb-4ce1-be06-b9bd3810ee50'::uuid;
update public.securities set market_data_symbol='HLMA:LSE' where security_id='bd6c9473-bfa4-41fa-9022-b8787bfedb00'::uuid;
update public.securities set market_data_symbol='RMS:EPA' where security_id='7d31c334-aa2b-66d2-11ed-9e0c4fb6e1dc'::uuid;
update public.securities set market_data_symbol='ISRG' where security_id='c459062e-baa6-43ec-b75e-86916b805e90'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='KKR' where security_id='27d40e90-21d4-4a99-8cd7-a13bb40e90a7'::uuid;
update public.securities set market_data_symbol='META' where security_id='766fac58-96a3-407a-9302-1484abff41f5'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='MSFT' where security_id='6d82977c-f4bd-4ac2-a2ba-227abb640471'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='MUV2:XETR' where security_id='84b71262-fb37-4198-90f0-9d85f4442e44'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='NOVO-B:CPH' where security_id='9fa42e6f-384d-4d78-b30f-f66590913d2d'::uuid;
update public.securities set market_data_symbol='NVDA' where security_id='1dec8fb7-9acd-4dac-9b4a-9b43290c79b5'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='PEP' where security_id='f14244b2-581d-4040-8a7d-c11bf55114ae'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='RI:EPA' where security_id='5497554e-6f38-4ce0-9800-9a9fc74c7973'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='QLYS' where security_id='5ca6f4a2-a796-55fd-58f8-0798a4b04fbe'::uuid;
update public.securities set market_data_symbol='RAA:XETR' where security_id='2202c604-24d9-2737-b8dd-7b192de1fc11'::uuid;
update public.securities set market_data_symbol='STMPA:EPA' where security_id='6b91c58a-8f54-4a4b-9727-446e7800363d'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='HO:EPA' where security_id='69641ba2-5138-4c3e-994e-4ef1fc2ec8b5'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='TOI:TSXV' where security_id='aee0ff49-412a-4af3-b402-b1b0ccd023d6'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='TTE:EPA' where security_id='28174abd-d2e7-415d-8491-a8ee07325e4d'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='TDG' where security_id='e15698ca-050b-4c70-ab7a-2bc3ef5f58a4'::uuid;
update public.securities set market_data_symbol='2330:TWSE' where security_id='076617ff-e951-4cdc-8782-9d5e0a4392bf'::uuid and market_data_symbol is null;
update public.securities set market_data_symbol='V' where security_id='9587e2df-41e7-430d-b48f-51a45270ea91'::uuid and market_data_symbol is null;

-- Wise's currently published security identity is intentionally left untouched pending listing reconciliation.
