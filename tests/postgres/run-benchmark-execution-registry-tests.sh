#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fixture="$root/tests/postgres/i1-fixture.sql"
migration="$root/migrations/20261004_orotitan_engine_benchmark_execution_registry.sql"
verify="$root/tests/postgres/benchmark-execution-registry-verify.sql"
db="orotitan_benchmark_registry_$$"

cleanup() {
  dropdb --if-exists "$db" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "$db"
psql -v ON_ERROR_STOP=1 -d "$db" -f "$fixture"
psql -v ON_ERROR_STOP=1 -d "$db" -f "$migration"
psql -v ON_ERROR_STOP=1 -d "$db" -f "$verify"
