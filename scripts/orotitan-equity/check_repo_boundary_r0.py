#!/usr/bin/env python3
"""OroTitan Repository Boundary R0: keep PRIVATE research bytes outside PUBLIC indice_nexus.

Only Git metadata and added/modified file bytes are read. Never print content or secrets.
"""
from __future__ import annotations

import argparse
import hashlib
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


MAX_HISTORY_COMMITS = 2000
MAX_CHANGED_BLOBS = 5000


def introduced_changes(base: str, head: str):
    """Yield each added/changed blob in every commit, not only the final diff.

    This includes merged histories and blobs later overwritten or deleted.
    """
    for ref in (base, head):
        if not re.fullmatch(r"[0-9a-f]{40}", ref):
            raise ValueError("INVALID_DIFF_PIN")
    try:
        git("merge-base", "--is-ancestor", base, head)
        commits = git("rev-list", "--reverse", f"{base}..{head}").decode("ascii").splitlines()
    except (subprocess.CalledProcessError, UnicodeDecodeError):
        raise ValueError("UNVERIFIED_GIT_HISTORY") from None
    if len(commits) > MAX_HISTORY_COMMITS:
        raise ValueError("HISTORY_TOO_LARGE_FOR_AUTOMATIC_ATTESTATION")

    observed: set[tuple[str, str, str | None]] = set()
    for commit in commits:
        parents = git("rev-list", "--parents", "-n", "1", commit).decode("ascii").split()
        if not parents or parents[0] != commit or len(parents) < 2:
            raise ValueError("UNEXPECTED_ROOT_OR_GIT_HISTORY")
        for parent in parents[1:]:
            fields = git(
                "diff", "--name-status", "-z", "--find-renames", parent, commit
            ).split(b"\0")
            index = 0
            while index < len(fields) and fields[index]:
                status = fields[index].decode("ascii")
                index += 1
                if index >= len(fields) or not fields[index]:
                    raise ValueError("INVALID_GIT_CHANGE_RECORD")
                old_path = fields[index].decode("utf-8")
                index += 1
                if status.startswith(("R", "C")):
                    if index >= len(fields) or not fields[index]:
                        raise ValueError("INVALID_GIT_RENAME_RECORD")
                    path = fields[index].decode("utf-8")
                    index += 1
                    source_path = old_path
                else:
                    path = old_path
                    source_path = None
                if status == "D":
                    continue
                entry = (commit, path, source_path)
                if entry not in observed:
                    observed.add(entry)
                    if len(observed) > MAX_CHANGED_BLOBS:
                        raise ValueError("TOO_MANY_CHANGED_BLOBS")
                    yield entry


def scan_commit_range(base: str, head: str) -> list[tuple[str, str]]:
    """Fail closed if any intermediate commit introduced non-public bytes."""
    violations: list[tuple[str, str]] = []
    for commit, path, old_path in introduced_changes(base, head):
        obj = f"{commit}:{path}"
        try:
            if git("cat-file", "-t", obj).strip() != b"blob":
                violations.append((path, "NON_BLOB_GIT_OBJECT"))
                continue
            size = int(git("cat-file", "-s", obj))
            if size > MAX_SCAN_BYTES:
                violations.append((path, "UNBOUNDED_CHANGED_FILE"))
                continue
            data = git("show", obj)
        except (subprocess.CalledProcessError, ValueError):
            violations.append((path, "GIT_BLOB_CANNOT_BE_VERIFIED"))
            continue
        for candidate_path in (path, old_path) if old_path else (path,):
            for code in inspect_path_and_content(candidate_path, data):
                violations.append((candidate_path, code))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args()
    if os.getenv("GITHUB_REPOSITORY", REPO) != REPO:
        print("BLOCKED_R0_WRONG_REPOSITORY", file=sys.stderr)
        return 1
    try:
        violations = scan_commit_range(args.base, args.head)
    except (ValueError, subprocess.CalledProcessError, UnicodeDecodeError):
        print("BLOCKED_R0_UNVERIFIED_HISTORY", file=sys.stderr)
        return 1
    for path, code in violations:
        # A file path may itself contain private information; never log it.
        safe_id = hashlib.sha256(path.encode("utf-8")).hexdigest()[:12]
        print(f"BLOCKED_R0 {code} path_sha256={safe_id}", file=sys.stderr)
    if violations:
        return 1
    print("OROTITAN_REPOSITORY_BOUNDARY_R0=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
