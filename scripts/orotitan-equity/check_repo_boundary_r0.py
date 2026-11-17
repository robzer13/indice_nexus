#!/usr/bin/env python3
"""OroTitan Repository Boundary R0: keep PRIVATE research bytes outside PUBLIC indice_nexus.

Only Git metadata and added/modified file bytes are read. Never print content or secrets.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import PurePosixPath
import re
import subprocess
import sys

REPO = "robzer13/indice_nexus"
DENIED_PARTS = frozenset({
    "artifacts", "research-workpapers", "c03_local_simulation",
    "private-artifacts", "private-data", "private-artifact-exports"
})
DENIED_FILES = frozenset({
    "EVIDENCE_REPRISE_C02.json",
    "SIMULATED_CANONICAL_LOCATOR_EVIDENCE.json",
    "OROTITAN_SCHNEIDER_C07_PHASE_B_SECRET_BASE64.txt",
})
ARTIFACT_NAME = re.compile(
    r"^(?:SCORING_INPUTS|FUNDAMENTALS_LOCK|DEEP_DIVE_STAGE_MANIFEST|"
    r"RESEARCH_STAGE_MANIFEST|EVIDENCE_LEDGER|CALCULATION_LEDGER|"
    r"ACQUISITION_CAPITAL_LEDGER|FUNDAMENTAL_RED_TEAM_RECORD|"
    r"FUNDAMENTALS_ANALYTICAL_BLOCK_OUTPUTS|MATERIAL_ASSUMPTION_REGISTER)"
    r"__v[0-9]{3}\.json$"
)
SECRET_PATTERN = re.compile(
    rb"(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"sb_secret_[A-Za-z0-9_-]{20,})"
)
DENIED_SUFFIXES = frozenset({".zip", ".7z", ".sqlite", ".sqlite3", ".db", ".p12", ".pfx", ".key"})
MAX_SCAN_BYTES = 2_000_000


def inspect_path_and_content(path: str, content: bytes) -> list[str]:
    problems = []
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in path:
        return ["UNSAFE_PATH"]
    if any(segment.lower() in DENIED_PARTS for segment in parsed.parts):
        problems.append("PRIVATE_DATA_DIRECTORY_IN_PUBLIC_REPO")
    if parsed.name in DENIED_FILES or ARTIFACT_NAME.fullmatch(parsed.name):
        problems.append("PRIVATE_ARTIFACT_IN_PUBLIC_REPO")
    if parsed.suffix.lower() in DENIED_SUFFIXES:
        problems.append("PRIVATE_EXPORT_OR_BINARY_IN_PUBLIC_REPO")
    if parsed.name == ".env" or (parsed.name.startswith(".env.") and parsed.name != ".env.example"):
        problems.append("LOCAL_ENVIRONMENT_FILE_IN_PUBLIC_REPO")
    if len(content) > MAX_SCAN_BYTES:
        problems.append("UNBOUNDED_CHANGED_FILE")
        return problems
    if SECRET_PATTERN.search(content):
        problems.append("CREDENTIAL_PATTERN_IN_PUBLIC_REPO")
    if path.startswith(".github/workflows/") and b"pull_request_target:" in content:
        problems.append("PR_TARGET_WORKFLOW_NOT_ALLOWED")
    if parsed.suffix.lower() == ".json":
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, ValueError):
            value = None
        if isinstance(value, dict):
            if (
                ("run_id" in value and any(k in value for k in
                    ("output_artifacts", "manifest_id", "artifact_type", "lock_type",
                     "scoring_inputs_artifact", "analytical_block_outputs")))
                or "original_bytes_verified" in value
                or "git_commit_local_manifest" in value
            ):
                problems.append("PRIVATE_RUN_INSTANCE_JSON_IN_PUBLIC_REPO")
    return problems


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)


def changed_files(base: str, head: str) -> list[tuple[str, str]]:
    for name in (base, head):
        if not re.fullmatch(r"[0-9a-f]{40}", name):
            raise ValueError("INVALID_DIFF_PIN")
    parts = git("diff", "--name-status", "-z", "--find-renames", base, head).split(b"\0")
    changes = []
    i = 0
    while i < len(parts) and parts[i]:
        status = parts[i].decode("ascii")
        i += 1
        path = parts[i].decode("utf-8")
        i += 1
        if status.startswith(("R", "C")):
            path = parts[i].decode("utf-8")
            i += 1
        changes.append((status, path))
    return changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    if os.getenv("GITHUB_REPOSITORY", REPO) != REPO:
        print("BLOCKED_R0_WRONG_REPOSITORY", file=sys.stderr)
        return 1
    violations = []
    for status, path in changed_files(args.base, args.head):
        if status == "D":
            continue
        try:
            data = git("show", args.head + ":" + path)
        except subprocess.CalledProcessError:
            violations.append((path, "MISSING_CHANGED_FILE"))
            continue
        for code in inspect_path_and_content(path, data):
            violations.append((path, code))
    for path, code in violations:
        print("BLOCKED_R0 " + code + " " + path, file=sys.stderr)
    if violations:
        return 1
    print("OROTITAN_REPOSITORY_BOUNDARY_R0=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
