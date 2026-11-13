"""Public offline C07 guard tests. NEVER require secrets, network, or Supabase."""
from __future__ import annotations

import base64
from hashlib import sha256
import importlib.util
import io
import zipfile
import os
from pathlib import Path
import re
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
COLLECTOR_PATH = ROOT / "scripts/orotitan-equity/schneider-c07-readonly-collect.py"
WORKFLOW_PATH = ROOT / ".github/workflows/orotitan-schneider-c07-private-preflight.yml"
SNAPSHOT_PATH = ROOT / "scripts/orotitan-equity/schneider-c07-shadow-readonly-snapshot.sql"
spec = importlib.util.spec_from_file_location("c07_private_collector_offline", COLLECTOR_PATH)
assert spec is not None and spec.loader is not None
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class OfflineC07Contract(unittest.TestCase):
    def test_exact_bytes_accept_verified_github_blob_and_sha(self):
        payload = b'{"artifact_type":"SYNTHETIC_C07","version":1}\n'
        proof = {
            "content_sha256": sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "github_blob_sha": collector.blob(payload),
        }
        result = collector.prove(payload, proof, "synthetic")
        self.assertEqual(result["sha256"], proof["content_sha256"])
        self.assertEqual(result["git_blob_sha"], proof["github_blob_sha"])
        self.assertEqual(result["size_bytes"], len(payload))

    def test_wrong_content_same_size_is_rejected(self):
        original = b'{"synthetic":"OK"}'
        tampered = b'{"synthetic":"NO"}'
        self.assertEqual(len(original), len(tampered))
        expected = {
            "content_sha256": sha256(original).hexdigest(),
            "size_bytes": len(original),
            "github_blob_sha": collector.blob(original),
        }
        with self.assertRaisesRegex(RuntimeError, "BLOCKED_C07"):
            collector.prove(tampered, expected, "synthetic-tamper")

    def test_wrong_git_blob_is_rejected(self):
        raw = b"local-only"
        expected = {
            "content_sha256": sha256(raw).hexdigest(),
            "size_bytes": len(raw),
            "github_blob_sha": "0" * 40,
        }
        with self.assertRaisesRegex(RuntimeError, "Git-blob mismatch"):
            collector.prove(raw, expected, "synthetic-tamper")

    def test_immutable_ref_rejected_before_network(self):
        with patch.object(collector.urllib.request, "urlopen", side_effect=AssertionError("NETWORK FORBIDDEN")):
            with self.assertRaisesRegex(RuntimeError, "invalid immutable source commit"):
                collector.github_get("dummy-not-a-token", "synthetic.json", "main")

    def test_immutable_github_get_is_read_only_and_repository_pinned(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self, maximum):
                return b"synthetic-public-test-bytes"

        commit = "f" * 40
        with patch.object(collector.urllib.request, "urlopen", return_value=Response()) as called:
            result = collector.github_get("DUMMY_TEST_TOKEN_NEVER_REAL", "artifacts/example.json", commit)
        self.assertEqual(result, b"synthetic-public-test-bytes")
        request = called.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertIn("/repos/robzer13/real-orotitan/contents/artifacts/example.json", request.full_url)
        self.assertTrue(request.full_url.endswith("?ref=" + commit))

    def test_non_main_or_non_manual_runs_fail_without_network(self):
        cases = [
            {"GITHUB_EVENT_NAME": "pull_request", "GITHUB_REF": "refs/heads/main",
             "GITHUB_REPOSITORY": "robzer13/indice_nexus"},
            {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/heads/feature",
             "GITHUB_REPOSITORY": "robzer13/indice_nexus"},
            {"GITHUB_EVENT_NAME": "workflow_dispatch", "GITHUB_REF": "refs/heads/main",
             "GITHUB_REPOSITORY": "some-other/repository"},
        ]
        with patch.object(collector.urllib.request, "urlopen", side_effect=AssertionError("NETWORK FORBIDDEN")):
            for env in cases:
                with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "manual dispatch"):
                        collector.main()

    def test_unconfigured_protected_inputs_fail_without_network(self):
        env = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_REPOSITORY": "robzer13/indice_nexus",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            collector.urllib.request, "urlopen", side_effect=AssertionError("NETWORK FORBIDDEN")
        ):
            with self.assertRaisesRegex(RuntimeError, "required protected credentials"):
                collector.main()

    def test_invalid_bundle_pin_fails_without_network(self):
        env = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_REPOSITORY": "robzer13/indice_nexus",
            "C07_PRIVATE_READ_TOKEN": "DUMMY_TEST_TOKEN_NEVER_REAL",
            "C07_C03_BUNDLE_BASE64": base64.b64encode(b"synthetic-not-a-zip").decode("ascii"),
            "C07_BUNDLE_SHA256": collector.APPROVED_C03_SHA256,
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            collector.urllib.request, "urlopen", side_effect=AssertionError("NETWORK FORBIDDEN")
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                collector.main()

    def test_unapproved_self_signed_bundle_rejected(self):
        data = b"synthetic-not-the-approved-bundle"
        env = {
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REF": "refs/heads/main",
            "GITHUB_REPOSITORY": "robzer13/indice_nexus",
            "C07_PRIVATE_READ_TOKEN": "DUMMY_TEST_TOKEN_NEVER_REAL",
            "C07_C03_BUNDLE_BASE64": base64.b64encode(data).decode("ascii"),
            "C07_BUNDLE_SHA256": sha256(data).hexdigest(),
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            collector.urllib.request, "urlopen", side_effect=AssertionError("NETWORK FORBIDDEN")
        ):
            with self.assertRaisesRegex(RuntimeError, "unapproved C03 bundle fingerprint"):
                collector.main()

    @staticmethod
    def _synthetic_zip(payloads: dict[str, bytes]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, raw in payloads.items():
                z.writestr(name, raw)
        return buffer.getvalue()

    def test_exact_five_members_allowed_for_bounded_read(self):
        entries = {name: b"{}" for name in collector.EXPECTED_MEMBERS}
        self.assertEqual(collector.read_c03_members(self._synthetic_zip(entries)), entries)

    def test_zip_member_over_expansion_cap_rejected(self):
        entries = {name: b"{}" for name in collector.EXPECTED_MEMBERS}
        entries[collector.SOURCE_PROOF] = b"x" * (collector.MAX_OBJECT + 1)
        with self.assertRaisesRegex(RuntimeError, "uncompressed size"):
            collector.read_c03_members(self._synthetic_zip(entries))

    def test_zip_total_expansion_cap_rejected(self):
        entries = {name: b"x" * 200_000 for name in collector.EXPECTED_MEMBERS}
        self.assertTrue(all(len(x) < collector.MAX_OBJECT for x in entries.values()))
        with self.assertRaisesRegex(RuntimeError, "uncompressed size"):
            collector.read_c03_members(self._synthetic_zip(entries))

    def test_zip_unexpected_members_and_duplicates_rejected(self):
        entries = {name: b"{}" for name in collector.EXPECTED_MEMBERS}
        entries["unexpected.txt"] = b"bad"
        with self.assertRaisesRegex(RuntimeError, "unexpected C03 ZIP members"):
            collector.read_c03_members(self._synthetic_zip(entries))
        buffer = io.BytesIO()
        names = list(collector.EXPECTED_MEMBERS)
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in names[:-1]:
                z.writestr(name, b"{}")
            z.writestr(names[0], b"duplicate")
        with self.assertRaisesRegex(RuntimeError, "unexpected C03 ZIP members"):
            collector.read_c03_members(buffer.getvalue())

    def test_sql_snapshot_guard_has_pinned_nonempty_cas_and_counts(self):
        text = SNAPSHOT_PATH.read_text(encoding="utf-8")
        self.assertIn("SELECT 1 / CASE WHEN", text)
        self.assertLess(text.index("AS c07_registry_baseline_guard"), text.index("\\o orotitan_runs.jsonl"))
        for invariant in (
            "run_status = 'ACTIVE' AND state_version = 7",
            "stage_code = 'DEEP_DIVE' AND lifecycle_status = 'IN_PROGRESS'",
            "state_version = 2 AND active_manifest_kind = 'CHECKPOINT'",
            "active_manifest_version = 1",
            ") = 2", ") = 28", ") = 8", ") = 35",
            "artifact_type = 'FUNDAMENTALS_LOCK' AND version = 2) = 0",
        ):
            self.assertIn(invariant, text)

    def test_protected_workflow_is_manual_main_only_and_does_not_upload(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("on:\n  workflow_dispatch:", text)
        self.assertNotIn("\n  pull_request:", text)
        self.assertIn("permissions:\n  contents: read", text)
        self.assertIn("environment: orotitan-schneider-c07-readonly", text)
        self.assertIn("refs/heads/main", text)
        self.assertIn(collector.APPROVED_C03_SHA256, text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("PGHOST: db.awgsurdyvsyolcgpnygh.supabase.co", text)
        self.assertIn("default_transaction_read_only=on", text)
        self.assertIn("if: always()", text)
        self.assertIn("TRUSTED_LOADER_POSTCOMMIT=NOT_EXECUTED", text)
        self.assertIn("FUNDAMENTALS_V2=NO_GO", text)
        self.assertNotIn("upload-artifact", text)

    def test_snapshot_sql_is_select_only_for_schneider(self):
        text = SNAPSHOT_PATH.read_text(encoding="utf-8")
        self.assertIn("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY;", text)
        self.assertIn("ROLLBACK;", text)
        self.assertEqual(text.count("SELECT row_to_json(t)::text"), 5)
        self.assertEqual(text.count("SELECT * FROM"), 5)
        self.assertGreaterEqual(text.count(collector.RUN), 14)
        sql = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("--"))
        self.assertIsNone(re.search(
            r"(?im)^\s*(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|CALL|EXECUTE|COPY|DO)\b",
            sql,
        ))


if __name__ == "__main__":
    unittest.main()
