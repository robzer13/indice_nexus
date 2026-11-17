"""Public offline C08 safety invariants; no secrets, DB or network."""
from __future__ import annotations
import ast
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / ".github/workflows/orotitan-schneider-c07-private-preflight.yml"
SQL = ROOT / "scripts/orotitan-equity/schneider-c07-shadow-readonly-snapshot.sql"
PY = ROOT / "scripts/orotitan-equity/schneider-c08-local-pg17-replay.py"
COL = ROOT / "scripts/orotitan-equity/schneider-c07-readonly-collect.py"
RUNNER = ROOT / "tests/postgres/run-registry-v1-migration-tests.sh"
RUN = "db00ce77-04eb-492b-a76b-164ecbe9ccfe"


class C08OfflineReplay(unittest.TestCase):
    def test_python_compiles_and_provides_real_rpc(self):
        src = PY.read_text()
        ast.parse(src)
        for marker in ("checkpoint_orotitan_stage(", "RUN_STATE_VERSION_MISMATCH",
                       "STAGE_STATE_VERSION_MISMATCH", "8:3:2:1",
                       "LOCAL_PG17_SCHNEIDER_CAS_7_2_TO_8_3=PASS"):
            self.assertIn(marker, src)

    def test_snapshot_nine_tables_single_readonly_transaction(self):
        sql = SQL.read_text()
        self.assertEqual(sql.count("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;"), 1)
        self.assertEqual(sql.count("ROLLBACK;"), 1)
        exports = re.findall(r"(?m)^\\o ([a-z0-9_]+\.jsonl)$", sql)
        self.assertEqual(set(exports), {
            "orotitan_runs.jsonl", "orotitan_run_stages.jsonl",
            "orotitan_artifacts.jsonl", "orotitan_artifact_edges.jsonl",
            "orotitan_run_events.jsonl", "issuers.jsonl",
            "securities.jsonl", "research_dossiers.jsonl",
            "orotitan_p0e_run_admission_receipts_v1.jsonl"})
        self.assertEqual(len(exports), 9)
        self.assertGreaterEqual(sql.count(RUN), 9)
        self.assertIn("c08_fk_closure_guard", sql)
        self.assertNotRegex(sql, r"(?im)^\s*(INSERT|UPDATE|DELETE|TRUNCATE|CREATE|ALTER|DROP|GRANT)\s")

    def test_secret_access_scoped_away_from_replay(self):
        flow = WF.read_text()
        job = flow.split("    steps:", 1)[0]
        self.assertNotIn("C07_PRIVATE_READ_TOKEN", job)
        self.assertNotIn("C07_C03_BUNDLE_BASE64", job)
        self.assertNotIn("PGPASSWORD:", job)
        self.assertEqual(len(re.findall(r"(?m)^\s+C07_PRIVATE_READ_TOKEN: ", flow)), 1)
        self.assertEqual(len(re.findall(r"(?m)^\s+C07_C03_BUNDLE_BASE64: ", flow)), 1)
        replay = flow.split("      - name: C08 actual Schneider RPC", 1)[1]
        self.assertNotIn("secrets.", replay)
        self.assertIn("PGHOST=127.0.0.1", replay)
        self.assertIn("PGSSLMODE=disable", replay)
        self.assertIn("BLOCKED_C08_SECRET_ENV_LEAK", replay)
        self.assertNotIn("upload-artifact", flow)

    def test_guarded_manual_main_only(self):
        flow = WF.read_text()
        self.assertIn("on:\n  workflow_dispatch:", flow)
        self.assertNotIn("\n  pull_request:", flow)
        self.assertIn("environment: orotitan-schneider-c07-readonly", flow)
        self.assertIn("refs/heads/main", flow)
        self.assertIn("C08_LOCAL_ONLY", flow)
        self.assertIn("image: postgres:17", flow)
        self.assertIn("c08_local_replay:", flow)

    def test_true_local_restore_keeps_fks(self):
        src = PY.read_text()
        for marker in ("SET CONSTRAINTS ALL DEFERRED", "LOCAL_POSTGRES_ONLY",
                       '"127.0.0.1"', '"PGSSLMODE": "disable"'):
            self.assertIn(marker, src)
        for bad in ("session_replication_role", "DISABLE TRIGGER", "DROP CONSTRAINT"):
            self.assertNotIn(bad, src)
        for table in ("issuers", "securities", "research_dossiers",
                      "orotitan_p0e_run_admission_receipts_v1"):
            self.assertIn('"' + table + '"', src)

    def test_simulation_stays_local_and_private(self):
        src = PY.read_text()
        self.assertIn("LOCAL_ONLY_NOT_REMOTE_AUTHORITY", src)
        self.assertIn("SIMULATION_ONLY_NOT_PERSISTED", src)
        self.assertIn("PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1", src)
        self.assertIn("GITHUB_CONNECTOR_PRIVATE_REREAD_V1", src)
        self.assertIn("SIMULATED_CANONICAL_LOCATOR_EVIDENCE.json", src)
        self.assertIn("SIMULATED_CANONICAL_LOCATOR_EVIDENCE.json", COL.read_text())
        self.assertIn("FUNDAMENTALS_V2=NO_GO", WF.read_text())

    def test_no_duplicate_workflow_blocks_or_unconditional_loader_pass(self):
        flow = WF.read_text()
        step_names = re.findall(r'(?m)^      - name: (.+)$', flow)
        self.assertEqual(len(step_names), len(set(step_names)))
        self.assertEqual(step_names.count('Remove private local working copies'), 1)
        self.assertEqual(step_names.count('Explicit unqualified status'), 1)
        self.assertEqual(step_names.count('C08 actual Schneider RPC against disposable local PostgreSQL 17'), 1)
        self.assertLess(len(flow), 18000)
        self.assertEqual(flow.count('C08 LOCAL_POSTCOMMIT_TRUSTED_LOADER=PASS_LOCAL_ONLY'), 1)
        self.assertIn("grep -Fxq 'C08 LOCAL_POSTCOMMIT_TRUSTED_LOADER=PASS'", flow)
        cleanup = flow.split('      - name: Remove private local working copies', 1)[1]
        self.assertNotIn('PASS_LOCAL_ONLY', cleanup)
        self.assertNotIn('secrets.', cleanup)

    def test_fk_baseline_guard_does_not_corrupt_jsonl(self):
        sql = SQL.read_text()
        self.assertIn('\\o\n-- C08: four FK-parent', sql)
        self.assertLess(sql.index('\\o\n-- C08:'), sql.index('AS c08_fk_closure_guard;'))
        self.assertLess(sql.index('AS c08_fk_closure_guard;'), sql.index('\\o issuers.jsonl'))
    def test_existing_pg17_test_runner_is_only_opt_in(self):
        runner = RUNNER.read_text()
        self.assertIn("OROTITAN_C08_LOCAL_REPLAY", runner)
        self.assertIn("127.0.0.1", runner)
        self.assertIn("schneider-c08-local-pg17-replay.py", runner)
        self.assertIn("schneider-c08-postcommit-loader.ts", runner)

    def test_postcommit_loader_is_pinned_and_prepared_before_private_collection(self):
        flow = WF.read_text()
        self.assertIn("ref: b5162a7e5b49500ed8d4e727a30fed928082516c", flow)
        self.assertIn("npm install --package-lock=false --ignore-scripts", flow)
        self.assertLess(flow.index("Prepare pinned TypeScript tooling before private collection"),
                        flow.index("Verify all sealed historical objects"))
        source = (ROOT / "scripts/orotitan-equity/schneider-c08-postcommit-loader.ts").read_text()
        self.assertIn("loadTrustedSchneiderShadowSnapshot", source)
        self.assertIn("expectedRunStateVersion: 8", source)
        self.assertIn("expectedCurrentStageStateVersion: 3", source)
        self.assertIn("LOCAL_POSTCOMMIT_TRUSTED_LOADER=PASS", source)
        self.assertIn("PGHOST", source)
        self.assertNotIn("https://", source)


if __name__ == "__main__":
    unittest.main()
