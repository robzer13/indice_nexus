#!/usr/bin/env bash
# D36: independent preview confirmations, two real concurrent PG17 sessions.
# Operates only inside the disposable localhost database created by D32 CI.
set -euo pipefail
db="${1:?need fixture db}"
[[ "$db" =~ ^orotitan_d32_mobile_[0-9]+$ ]] || exit 2
case "${PGHOST:-127.0.0.1}" in localhost|127.0.0.1|::1) ;; *) exit 2;; esac
[[ "${PGPORT:-5432}" == 5432 && "${PGUSER:-postgres}" == postgres ]] || exit 2
out1="$(mktemp)";out2="$(mktemp)"
trap 'rm -f "$out1" "$out2"' EXIT
psql -X -v ON_ERROR_STOP=1 -d "$db" >/dev/null <<'SQL'
INSERT INTO orotitan_mobile_private.previews(
 preview_id,kind,actor_subject,actor_client,actor_resource,
 payload,fingerprint_sha256,expires_at)
VALUES
('36000000-0000-4000-8000-000000000007','request_research',
 'person-c-account','approved-mobile-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"issuerId":"00000000-0000-4000-8000-000000000002","name":"Beta Display","ticker":"BET","exchange":"XPAR","isin":"FR1234567890"}',
 repeat('7',64),clock_timestamp()+interval '105 seconds'),
('36000000-0000-4000-8000-000000000008','request_research',
 'person-c-account','approved-desktop-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '{"issuerId":"00000000-0000-4000-8000-000000000002","name":"Beta Display","ticker":"BET","exchange":"XPAR","isin":"FR1234567890"}',
 repeat('8',64),clock_timestamp()+interval '105 seconds');

SET ROLE orotitan_d36_scope_issuer;
SELECT orotitan_mobile_private.issue_research_scope_v4(
 '36000000-0000-4000-8000-000000000007',
 'person-c-account','approved-mobile-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '00000000-0000-4000-8000-000000000002',repeat('c',64),current_date);
SELECT orotitan_mobile_private.issue_research_scope_v4(
 '36000000-0000-4000-8000-000000000008',
 'person-c-account','approved-desktop-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 '00000000-0000-4000-8000-000000000002',repeat('c',64),current_date);
RESET ROLE;

SET ROLE orotitan_d32_approval_issuer;
SELECT orotitan_mobile_private.approve_preview_v2(
 '36000000-0000-4000-8000-000000000007','request_research',
 'person-c-account','approved-mobile-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 repeat('7',64),
 encode(public.digest(convert_to(repeat('G',100)||'.'||repeat('G',43),
  'UTF8'),'sha256'),'hex'));
SELECT orotitan_mobile_private.approve_preview_v2(
 '36000000-0000-4000-8000-000000000008','request_research',
 'person-c-account','approved-desktop-client',
 'https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp',
 repeat('8',64),
 encode(public.digest(convert_to(repeat('H',100)||'.'||repeat('H',43),
  'UTF8'),'sha256'),'hex'));
RESET ROLE;
SQL
q1="SET ROLE orotitan_d36_research_submitter;
SELECT * FROM orotitan_mobile_private.confirm_scoped_research_v4(
 '36000000-0000-4000-8000-000000000007',
 repeat('G',100)||'.'||repeat('G',43));"
q2="SET ROLE orotitan_d36_research_submitter;
SELECT * FROM orotitan_mobile_private.confirm_scoped_research_v4(
 '36000000-0000-4000-8000-000000000008',
 repeat('H',100)||'.'||repeat('H',43));"
psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$q1" >"$out1" &
a=$!
psql -XAt -v ON_ERROR_STOP=1 -d "$db" -c "$q2" >"$out2" &
b=$!
status=0
wait "$a" || status=1
wait "$b" || status=1
[[ "$status" == 0 ]] || {
 echo 'D36 concurrent PG17 confirmations failed' >&2; exit 1;
}
rows="$(cat "$out1" "$out2" | grep -E '^[a-f0-9-]{36}[|]REQUESTED_NOT_EXECUTED[|][ft]$' || true)"
[[ "$(printf '%s\n' "$rows" | grep -c 'REQUESTED_NOT_EXECUTED')" == 2 ]] || exit 1
[[ "$(printf '%s\n' "$rows" | cut -d '|' -f 1 | sort -u | wc -l)" == 1 ]] || exit 1
[[ "$(printf '%s\n' "$rows" | grep -c '|f$')" == 1 ]] || exit 1
[[ "$(printf '%s\n' "$rows" | grep -c '|t$')" == 1 ]] || exit 1
count="$(psql -XAt -d "$db" -c "SELECT count(*) FROM
 orotitan_mobile_private.research_requests_v4
 WHERE actor_subject='person-c-account' AND contract_pack_sha256=repeat('c',64)")"
[[ "$count" == 1 ]] || exit 1
echo 'D36 real two-session cross-device Research dedupe PASSED (local PG17 only).'
