#!/usr/bin/env bash
# Product demo: two independent PG17 sessions (phone and desktop analog).
set -euo pipefail
db="${1:?disposable database required}"
[[ "$db" =~ ^orotitan_product_shared_[0-9]+$ ]] || exit 2
case "${PGHOST:-127.0.0.1}" in 127.0.0.1|localhost|::1) ;; *) exit 2;; esac
[[ "${PGPORT:-5432}" == 5432 && "${PGUSER:-postgres}" == postgres ]] || exit 2
a="$(mktemp)"; b="$(mktemp)"
trap 'rm -f "$a" "$b"' EXIT

# Two distinct SQL connections contend for the same existing company.
q="SET ROLE service_role;
WITH attempt AS (
  INSERT INTO public.orotitan_research_intake_requests(company_id)
  VALUES ('00000000-0000-4000-8000-000000000092')
  ON CONFLICT (company_id) DO NOTHING
  RETURNING request_id
)
SELECT COALESCE((SELECT request_id::text FROM attempt),'ALREADY_EXISTS');"
psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$q" > "$a" &
pa=$!
psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$q" > "$b" &
pb=$!
status=0
wait "$pa" || status=1
wait "$pb" || status=1
[[ "$status" == 0 ]] || { echo 'Concurrent insert failure' >&2; exit 1; }
both="$(cat "$a" "$b")"
[[ "$(printf '%s\n' "$both" | grep -Ec '^[0-9a-f-]{36}$')" == 1 ]] || {
 echo 'Expected exactly one inserted request' >&2; exit 1;
}
[[ "$(printf '%s\n' "$both" | grep -c '^ALREADY_EXISTS$')" == 1 ]] || {
 echo 'Expected one deduplicated request' >&2; exit 1;
}

# Read from two fresh, independent connections (shared backend).
query="SET ROLE service_role; SELECT request_id::text
 FROM public.orotitan_research_intake_requests
 WHERE company_id='00000000-0000-4000-8000-000000000092';"
client_phone="$(psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$query")"
client_desktop="$(psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$query")"
[[ -n "$client_phone" && "$client_phone" == "$client_desktop" ]] || {
 echo 'Two independent clients must observe identical receipt' >&2; exit 1;
}
count="$(psql -XAt -d "$db" -c "
SELECT count(*) FROM public.orotitan_research_intake_requests
WHERE company_id='00000000-0000-4000-8000-000000000092';")"
[[ "$count" == 1 ]] || { echo 'Duplicate shared receipts' >&2; exit 1; }
all="$(psql -XAt -d "$db" -c "
SELECT count(*) FROM public.orotitan_research_intake_requests;")"
[[ "$all" == 2 ]] || { echo 'Expected two distinct companies with requests' >&2; exit 1; }
echo 'Product shared queue two-connection idempotent receipt PASS (DISPOSABLE PG17).'
