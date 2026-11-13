"""Public offline C07 guard tests. NEVER require secrets, network, or Supabase."""
from __future__ import annotations

import base64
from hashlib import sha256
import importlib.util
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
            "C07_BUNDLE_SHA256": "0" * 64,
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
            collector.urllib.request, "urlopen", side_effect=AssertionError("NETWORK FORBIDDEN")
        ):
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                collector.main()

    def test_protected_workflow_is_manual_main_only_and_does_not_upload(self):
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("on:\n  workflow_dispatch:", text)
        self.assertNotIn("\n  pull_request:", text)
        self.assertIn("permissions:\n  contents: read", text)
        self.assertIn("environment: orotitan-schneider-c07-readonly", text)
        self.assertIn("refs/heads/main", text)
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
        self.assertEqual(text.count(collector.RUN), 6)
        sql = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("--"))
        self.assertIsNone(re.search(
            r"(?im)^\s*(?:INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE|CALL|EXECUTE|COPY|DO)\b",
            sql,
        ))


if __name__ == "__main__":
    unittest.main()
