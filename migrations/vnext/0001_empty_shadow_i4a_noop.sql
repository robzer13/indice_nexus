-- OroTitan VNext shadow bootstrap.
-- Production I4-A is a data-bound identity-completion migration for eight
-- historical legacy-mapped issuers. The VNext shadow intentionally contains
-- no production company or research data, so replaying that migration would
-- fabricate production-specific rows and violate the empty-shadow boundary.
--
-- This marker preserves the logical migration position without changing schema
-- or inserting data. Structural parity is established by the surrounding
-- canonical and Registry migrations and verified independently.

select 1;
