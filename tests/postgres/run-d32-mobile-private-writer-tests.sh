#!/usr/bin/env bash
# D32: dedicated disposable PG17 DB and non-login roles. No remote DB.
set -euo pipefail
: "${PGHOST:=127.0.0.1}" "${PGPORT:=5432}" "${PGUSER:=postgres}" "${PGPASSWORD:=postgres}"
export PGHOST PGPORT PGUSER PGPASSWORD
case "$PGHOST" in 127.0.0.1|localhost|::1) ;; *)
 echo 'D32 refuses a remote PostgreSQL server' >&2; exit 2;; esac
[[ "$PGPORT" == 5432 && "$PGUSER" == postgres ]] || {
 echo 'D32 requires local fixture PGPORT=5432 and PGUSER=postgres' >&2; exit 2;
}
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
db="orotitan_d32_mobile_$$"
roles=(orotitan_d32_approval_issuer orotitan_d32_command_submitter
       orotitan_d32_private_executor orotitan_d34_preview_issuer
       orotitan_d34_preview_reader orotitan_d34_vault_issuer
       orotitan_d34_vault_reader orotitan_d36_scope_issuer
       orotitan_d36_research_submitter orotitan_d36_lifecycle_controller
       orotitan_d37_session_gateway orotitan_d38_mutation_gateway)
created=0
cleanup(){
 if [[ "$created" == 1 ]]; then
   dropdb --if-exists --force "$db" >/dev/null 2>&1 || true
   for role in "${roles[@]}"; do
     psql -XAt -d postgres -c "DROP ROLE IF EXISTS $role;" >/dev/null 2>&1 || true
   done
 fi
}
trap cleanup EXIT
# Abort on shared local cluster role collision, never remove someone else's role.
for role in "${roles[@]}"; do
 if [[ "$(psql -XAt -d postgres -c "SELECT count(*) FROM pg_roles WHERE rolname='$role'")" != 0 ]]; then
   echo "D32 local role name already exists; refusing: $role" >&2; exit 3;
 fi
done
createdb "$db"; created=1
version="$(psql -XAt -d "$db" -c 'SHOW server_version_num')"
((version>=170000 && version<180000)) || {
 echo 'D32 requires PostgreSQL 17' >&2; exit 3; }
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/tests/postgres/i1-fixture.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/migrations/20260909_orotitan_equity_i1_identity.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/docs/orotitan-equity/proposals/D30_MOBILE_REQUEST_QUEUE_SQL_REVIEW_ONLY.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/docs/orotitan-equity/proposals/D32_MOBILE_PRIVATE_DATABASE_WRITER_REVIEW_ONLY.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f  "$root/tests/postgres/d32-mobile-private-writer-verify.sql" >/dev/null
bash "$root/tests/postgres/d32-mobile-private-writer-race.sh" "$db"
npx --no-install tsx "$root/tests/postgres/d33-mobile-oauth-pg-e2e.ts" "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/docs/orotitan-equity/proposals/D34_PRIVATE_PREVIEW_ENCRYPTED_CAPABILITY_SQL_REVIEW_ONLY.sql" >/dev/null
npx --no-install tsx "$root/tests/postgres/d34-mobile-private-preview-vault-e2e.ts" "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/tests/postgres/d35-mobile-cross-stack-authz-audit.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/docs/orotitan-equity/proposals/D36_TENANT_SCOPED_RESEARCH_QUEUE_V4_REVIEW_ONLY.sql" >/dev/null
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/tests/postgres/d36-tenant-research-queue-verify.sql" >/dev/null
bash "$root/tests/postgres/d36-cross-device-race.sh" "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/docs/orotitan-equity/proposals/D37_OAUTH_BOUND_POSTGRES_ATTESTATION_REVIEW_ONLY.sql" >/dev/null
npx --no-install tsx "$root/tests/postgres/d37-oauth-session-attestation-e2e.ts" "$db"
psql -X -v ON_ERROR_STOP=1 -d "$db" -f \
 "$root/docs/orotitan-equity/proposals/D38_OAUTH_BOUND_MOBILE_MUTATIONS_V6_REVIEW_ONLY.sql" >/dev/null
npx --no-install tsx "$root/tests/postgres/d38-bound-mobile-mutations-e2e.ts" "$db"
echo 'D32 private PG17 mobile writer security tests PASSED (LOCAL fixture only).'
