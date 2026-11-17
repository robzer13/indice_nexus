"""Offline regression tests for the public repository boundary (no private data)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "scripts" / "orotitan-equity" / "check_repo_boundary_r0.py"
spec = importlib.util.spec_from_file_location("orotitan_public_boundary", SOURCE)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class PublicBoundaryTests(unittest.TestCase):
    def test_legitimate_application_change_allowed(self):
        self.assertEqual(mod.inspect_path_and_content("app/api/health/route.ts", b"export const GET = () => null;"), [])

    def test_technical_contract_json_allowed(self):
        raw = json.dumps({"contract_version": "V3.1", "expected_run_state_version": 8}).encode()
        self.assertEqual(mod.inspect_path_and_content("contracts/orotitan-equity/schema.json", raw), [])

    def test_private_artifact_directory_denied(self):
        result = mod.inspect_path_and_content(
            "artifacts/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/deep_dive/LOCK__v002.json", b"{}"
        )
        self.assertIn("PRIVATE_DATA_DIRECTORY_IN_PUBLIC_REPO", result)

    def test_private_artifact_basename_denied_anywhere(self):
        result = mod.inspect_path_and_content("docs/FUNDAMENTALS_LOCK__v002.json", b"{}")
        self.assertIn("PRIVATE_ARTIFACT_IN_PUBLIC_REPO", result)

    def test_private_manifest_payload_denied_outside_artifact_directory(self):
        raw = json.dumps({"run_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                          "manifest_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                          "output_artifacts": []}).encode()
        result = mod.inspect_path_and_content("docs/demo.json", raw)
        self.assertIn("PRIVATE_RUN_INSTANCE_JSON_IN_PUBLIC_REPO", result)

    def test_local_environment_and_archive_denied(self):
        self.assertIn("LOCAL_ENVIRONMENT_FILE_IN_PUBLIC_REPO",
                      mod.inspect_path_and_content(".env.production", b"X=1"))
        self.assertEqual(mod.inspect_path_and_content(".env.example", b"VALUE=placeholder"), [])
        self.assertIn("PRIVATE_EXPORT_OR_BINARY_IN_PUBLIC_REPO",
                      mod.inspect_path_and_content("docs/sealed-bundle.zip", b""))

    def test_private_token_pattern_is_rejected_without_printing_token(self):
        raw = b"export GH_TOKEN=ghp_" + b"A" * 36
        self.assertIn("CREDENTIAL_PATTERN_IN_PUBLIC_REPO",
                      mod.inspect_path_and_content("scripts/new.sh", raw))

    def test_pull_request_target_workflow_is_rejected(self):
        raw = b"name: unsafe\non:\n  pull_request_target:\n"
        self.assertIn("PR_TARGET_WORKFLOW_NOT_ALLOWED",
                      mod.inspect_path_and_content(".github/workflows/test.yml", raw))


if __name__ == "__main__":
    unittest.main()
