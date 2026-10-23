#!/usr/bin/env bash
# Additional D32 two-session PostgreSQL 17 race; only disposable local fixture.
set -euo pipefail
db="${1:?missing disposable d32 DB name}"
[[ "$db" =~ ^orotitan_d32_mobile_[0-9]+$ ]] || {
 echo 'D32 race refuses unexpected DB name' >&2; exit 2;
}
out1="$(mktemp)";out2="$(mktemp)"
trap 'rm -f "$out1" "$out2"' EXIT
psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null <<'SQL'
INSERT INTO orotitan_mobile_private.previews(
 preview_id,kind,actor_subject,actor_client,actor_resource,payload,
 fingerprint_sha256,expires_at
) VALUES ('20000000-0000-4000-8000-000000000005','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"name":"Race Fixture Inc","ticker":"RACE","exchange":"XAMS","isin":"NL3333333333"}',
 repeat('e',64),clock_timestamp()+interval '90 seconds');
SET ROLE orotitan_d32_approval_issuer;
SELECT orotitan_mobile_private.approve_preview_v2(
 '20000000-0000-4000-8000-000000000005','add_company',
 'approved-subject','approved-chatgpt-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 repeat('e',64),
 encode(public.digest(convert_to(repeat('m',100)||'.'||repeat('n',43),
  'UTF8'),'sha256'),'hex'));
RESET ROLE;
SQL
sql="SET ROLE orotitan_d32_command_submitter;
 SELECT * FROM orotitan_mobile_private.confirm_approved_request_v2(
 '20000000-0000-4000-8000-000000000005',
 repeat('m',100)||'.'||repeat('n',43));"
psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$sql" >"$out1" &
a=$!
psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$sql" >"$out2" &
b=$!
wait "$a";wait "$b"
cat "$out1" "$out2" | grep -E '^[0-9a-f-]{36}[|]add_company[|]REQUESTED_NOT_EXECUTED[|][ft]$'   >"${out1}.records"
if [[ "$(wc -l <"${out1}.records")" -ne 2 ]] ||
   [[ "$(cut -d '|' -f 1 "${out1}.records" | sort -u | wc -l)" -ne 1 ]] ||
   [[ "$(grep -c '|f$' "${out1}.records")" -ne 1 ]] ||
   [[ "$(grep -c '|t$' "${out1}.records")" -ne 1 ]]; then
  echo 'D32 race allowed two requests or inconsistent receipts' >&2; exit 1
fi
count="$(psql -XAt -d "$db" -c "SELECT count(*) FROM
 orotitan_mobile_private.command_requests WHERE kind='add_company'
 AND target_key='company:XAMS:RACE:NL3333333333'")"
[[ "$count" == 1 ]] || {
 echo "D32 raced target has $count rows, expected 1" >&2; exit 1;
}
rm -f "${out1}.records"
echo 'D32 two-session approval/confirmation race passed (1 request, 1 replay).'
