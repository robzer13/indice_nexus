"""Offline regression tests for the public repository boundary (no private data)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
from unittest.mock import patch
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


    @staticmethod
    def _git(cwd: str, *args: str) -> bytes:
        return subprocess.check_output(
            ["git", *args], cwd=cwd, stderr=subprocess.DEVNULL
        )

    def test_secret_then_cleanup_cannot_hide_prior_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            def git(*args: str) -> bytes:
                return self._git(tmp, *args)

            git("init", "-q")
            git("config", "user.name", "Offline Security Test")
            git("config", "user.email", "ci@example.invalid")
            readme = Path(tmp) / "README.md"
            readme.write_text("public baseline\n", encoding="utf-8")
            git("add", "README.md")
            git("commit", "-qm", "baseline")
            baseline = git("rev-parse", "HEAD").decode().strip()

            # Both disallowed blobs will remain in Git history even when
            # removed from the final tree. No actual credential is used.
            archive = Path(tmp) / "artifacts" / "private.json"
            archive.parent.mkdir(parents=True)
            archive.write_text("{}\n", encoding="utf-8")
            git("add", "artifacts/private.json")
            git("commit", "-qm", "transient private artifact")
            git("rm", "-q", "artifacts/private.json")
            git("commit", "-qm", "cleanup artifact")

            note = Path(tmp) / "notes.txt"
            note.write_bytes(b"token=ghp_" + b"A" * 36 + b"\n")
            git("add", "notes.txt")
            git("commit", "-qm", "transient credential")
            note.write_text("safe text\n", encoding="utf-8")
            git("add", "notes.txt")
            git("commit", "-qm", "sanitize current tree")
            final = git("rev-parse", "HEAD").decode().strip()

            with patch.object(mod, "git", side_effect=git):
                findings = mod.scan_commit_range(baseline, final)
            codes = {code for _, code in findings}
            self.assertIn("PRIVATE_DATA_DIRECTORY_IN_PUBLIC_REPO", codes)
            self.assertIn("CREDENTIAL_PATTERN_IN_PUBLIC_REPO", codes)

    def test_clean_history_with_intermediate_deletion_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            def git(*args: str) -> bytes:
                return self._git(tmp, *args)

            git("init", "-q")
            git("config", "user.name", "Offline Security Test")
            git("config", "user.email", "ci@example.invalid")
            file = Path(tmp) / "README.md"
            file.write_text("public baseline\n", encoding="utf-8")
            git("add", "README.md")
            git("commit", "-qm", "baseline")
            baseline = git("rev-parse", "HEAD").decode().strip()

            file.write_text("public update\n", encoding="utf-8")
            git("add", "README.md")
            git("commit", "-qm", "public change")
            git("rm", "-q", "README.md")
            git("commit", "-qm", "delete public file")
            final = git("rev-parse", "HEAD").decode().strip()
            with patch.object(mod, "git", side_effect=git):
                self.assertEqual(mod.scan_commit_range(baseline, final), [])

    def test_pull_request_target_workflow_is_rejected(self):
        raw = b"name: unsafe\non:\n  pull_request_target:\n"
        self.assertIn("PR_TARGET_WORKFLOW_NOT_ALLOWED",
                      mod.inspect_path_and_content(".github/workflows/test.yml", raw))


if __name__ == "__main__":
    unittest.main()
