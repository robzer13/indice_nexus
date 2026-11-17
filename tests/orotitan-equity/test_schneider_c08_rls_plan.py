"""Public offline C08 grant-plan assertions: NO network/DB/secret access."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/orotitan-equity/c08"
SQL = BASE / "schneider-c08-shadow-reader-grants.REVIEW_ONLY.sql"
VERIFY = BASE / "schneider-c08-shadow-reader-verify.READ_ONLY.sql"
DOC = BASE / "SCHNEIDER_C08_SHADOW_READ_ACCESS_REVIEW.md"
RUN = "db00ce77-04eb-492b-a76b-164ecbe9ccfe"
TABLES = (
    "issuers", "securities", "research_dossiers",
    "orotitan_p0e_run_admission_receipts_v1",
)

class C08ReaderGrantPlan(unittest.TestCase):
    def test_explicit_four_table_grants_and_policies(self):
        sql = SQL.read_text()
        self.assertEqual(sql.count("CREATE POLICY c08_schneider_reader_select"), 4)
        self.assertEqual(sql.count("GRANT SELECT ON TABLE public."), 4)
        for table in TABLES:
            self.assertIn(f"ON public.{table} AS PERMISSIVE", sql)
            self.assertIn(f"GRANT SELECT ON TABLE public.{table} TO orotitan_c07_reader;", sql)
        self.assertIn("COMMIT;", sql)

    def test_grants_are_only_select_and_scoped_to_schneider(self):
        sql = SQL.read_text()
        self.assertEqual(sql.count("FOR SELECT TO orotitan_c07_reader"), 4)
        self.assertGreaterEqual(sql.count("r.run_id='" + RUN + "'::uuid"), 4)
        self.assertGreaterEqual(sql.count("FROM public.orotitan_runs r"), 4)
        for token in ("GRANT ALL", "GRANT INSERT", "GRANT UPDATE",
                      "GRANT DELETE", "GRANT EXECUTE", "BYPASSRLS",
                      "DISABLE ROW LEVEL SECURITY", "SECURITY DEFINER",
                      "ALTER DEFAULT PRIVILEGES", "CREATE OR REPLACE FUNCTION"):
            self.assertNotIn(token + " ON TABLE", sql)
        self.assertNotRegex(sql, r"(?im)^\s*GRANT\s+(?:ALL|INSERT|UPDATE|DELETE|TRUNCATE|EXECUTE)\b")
        self.assertNotRegex(sql, r"(?im)^\s*GRANT\s+SELECT\b.*\bTO\s+(?:PUBLIC|anon|authenticated)\b")
        self.assertNotIn("cugpgtzygqqlxetyeven", sql)

    def test_preconditions_fail_closed(self):
        sql = SQL.read_text()
        for gate in ("BLOCKED_C08_ROLE_PRECONDITION",
                     "BLOCKED_C08_EXISTING_READER_SCOPE",
                     "BLOCKED_C08_TABLE_PRECONDITION",
                     "BLOCKED_C08_UNEXPECTED_EXISTING_RLS_POLICY",
                     "BLOCKED_C08_RUN_FK_CLOSURE_PRECONDITION"):
            self.assertIn(gate, sql)
        self.assertIn("pg_auth_members", sql)
        self.assertIn("relrowsecurity", sql)
        self.assertIn("state_version=7", sql)
        self.assertIn("BEGIN;", sql)

    def test_verification_requires_real_reader_login_without_rows(self):
        verify = VERIFY.read_text()
        self.assertIn("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;", verify)
        self.assertIn("current_user = 'orotitan_c07_reader'", verify)
        self.assertIn("session_user = 'orotitan_c07_reader'", verify)
        self.assertNotIn("SET LOCAL ROLE", verify)
        self.assertIn("AS c08_schneider_read_scope_guard;", verify)
        self.assertIn("ROLLBACK;", verify)
        for table in TABLES:
            self.assertIn("FROM public." + table, verify)
        self.assertNotRegex(verify, r"(?im)^\s*(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE)\b")

    def test_shadow_fingerprint_and_incoming_membership_fail_closed(self):
        sql = SQL.read_text()
        self.assertIn("BLOCKED_C08_WRONG_DATABASE", sql)
        self.assertIn("pg_catalog.pg_control_system()", sql)
        self.assertIn("902ab5e7b9a35b605ecf649f8e861c90a7c6b4ff10d71ccc579a454fe4cd2672", sql)
        self.assertIn("current_user <> 'postgres'", sql)
        self.assertIn("session_user <> 'postgres'", sql)
        self.assertIn("current_database() <> 'postgres'", sql)
        self.assertIn("BLOCKED_C08_READER_INCOMING_MEMBERSHIP", sql)
        self.assertIn("m.roleid = reader_oid", sql)
        self.assertIn("administrator.rolname = 'postgres'", sql)
        self.assertIn("NOT m.inherit_option AND NOT m.set_option", sql)
        self.assertIn("incoming_total_count <> 1 OR incoming_admin_count <> 1", sql)

    def test_review_only_no_auto_execution(self):
        text = DOC.read_text()
        self.assertIn("NOT APPLIED", text)
        self.assertIn("independent code/security review", text)
        self.assertIn("No private rows or credentials", text)
        self.assertIn("FUNDAMENTALS_V2=NO_GO", text)

if __name__ == "__main__":
    unittest.main()
