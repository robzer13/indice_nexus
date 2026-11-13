#!/usr/bin/env python3
"""C07 Phase B: read-only, fail-closed collection; NOT a Schneider SQL replay.

Runs only from manually dispatched, protected main-branch GitHub Actions.
Reads one sealed-candidate simulation bundle from an environment secret and
19 exact historical GitHub objects from the separate private repository.
Never uploads their bytes or prints their contents.
"""
from __future__ import annotations

import base64
from collections import Counter
from hashlib import sha1, sha256
import io
import json
import os
from pathlib import Path
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

RUN = "db00ce77-04eb-492b-a76b-164ecbe9ccfe"
PRIVATE_REPO = "robzer13/real-orotitan"
SOURCE_PROOF = "EVIDENCE_REPRISE_C02.json"
SIMULATED = "c03_local_simulation/SIMULATED_CANONICAL_LOCATOR_EVIDENCE.json"
CANDIDATES = (
    "SCORING_INPUTS__v001.json",
    "FUNDAMENTALS_LOCK__v002.json",
    "DEEP_DIVE_STAGE_MANIFEST__v002.json",
)
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Z0-9_]+\Z")
MAX_ZIP = 400_000
MAX_OBJECT = 300_000

def fail(message: str) -> None:
    raise RuntimeError("BLOCKED_C07: " + message)

def blob(raw: bytes) -> str:
    return sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()

def prove(raw: bytes, expected: dict, label: str) -> dict:
    digest = sha256(raw).hexdigest()
    if (digest != expected.get("content_sha256")
            or len(raw) != int(expected.get("size_bytes", -1))
            or blob(raw) != expected.get("github_blob_sha", expected.get("git_blob_sha"))):
        fail("exact-byte size/SHA-256/Git-blob mismatch for " + label)
    return {"sha256": digest, "size_bytes": len(raw), "git_blob_sha": blob(raw)}

def github_get(token: str, path: str, commit: str) -> bytes:
    if not SHA40.fullmatch(commit):
        fail("invalid immutable source commit")
    url = ("https://api.github.com/repos/" + PRIVATE_REPO + "/contents/"
           + urllib.parse.quote(path, safe="/") + "?ref=" + commit)
    request = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "orotitan-c07-readonly-attestation",
    }, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read(MAX_OBJECT + 1)
    except (urllib.error.URLError, TimeoutError) as error:
        fail("private immutable GitHub GET unavailable (" + type(error).__name__ + ")")
    if len(payload) > MAX_OBJECT:
        fail("private artifact exceeds size limit")
    try:
        meta = json.loads(payload)
        if isinstance(meta, dict) and meta.get("encoding") == "base64" and "content" in meta:
            return base64.b64decode(meta["content"], validate=False)
    except (ValueError, UnicodeDecodeError):
        pass
    return payload

def main() -> None:
    if (os.getenv("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or os.getenv("GITHUB_REF") != "refs/heads/main"
            or os.getenv("GITHUB_REPOSITORY") != "robzer13/indice_nexus"):
        fail("manual dispatch from public repository default branch required")
    token = os.getenv("C07_PRIVATE_READ_TOKEN", "")
    encoded = os.getenv("C07_C03_BUNDLE_BASE64", "")
    bundle_sha = os.getenv("C07_BUNDLE_SHA256", "")
    if not token or not encoded or not SHA64.fullmatch(bundle_sha):
        fail("required protected credentials / bundle pin missing")
    try:
        bundle = base64.b64decode(encoded, validate=True)
    except ValueError:
        fail("invalid protected bundle encoding")
    if len(bundle) > MAX_ZIP or sha256(bundle).hexdigest() != bundle_sha:
        fail("protected bundle SHA-256 mismatch")
    temp = Path(os.environ["RUNNER_TEMP"]) / "orotitan-schneider-c07-private"
    temp.mkdir(mode=0o700, parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(bundle)) as z:
        if len(z.namelist()) != 5:
            fail("expected minimal five-member C03 bundle")
        evidence = json.loads(z.read(SOURCE_PROOF))
        simulation = json.loads(z.read(SIMULATED))
        originals = evidence.get("original_bytes_verified", [])
        if (evidence.get("run_id") != RUN or simulation.get("run_id") != RUN
                or simulation.get("status") != "SIMULATION_ONLY_NOT_PERSISTED"
                or len(originals) != 19
                or Counter(x.get("stage") for x in originals)
                   != {"RESEARCH": 8, "DEEP_DIVE": 11}):
            fail("historical fixture set or simulation state is inconsistent")
        ids = {(x["artifact_id"], x["version"]) for x in originals}
        if len(ids) != 19:
            fail("duplicate historical identity/version")
        candidate_proofs = []
        for name in CANDIDATES:
            raw = z.read("c03_local_simulation/candidate_exports/" + name)
            expected = simulation["artifacts"][name]
            data = prove(raw, expected, "candidate")
            (temp / name).write_bytes(raw)
            candidate_proofs.append({"artifact_type": name, **data, "authority": "LOCAL_SIMULATION_ONLY"})
    historical_dir = temp / "historical"
    historical_dir.mkdir(mode=0o700)
    historical_proofs = []
    for x in originals:
        stage = x["stage"]
        artifact_type = x["artifact_type"]
        version = x["version"]
        if not NAME.fullmatch(artifact_type) or not isinstance(version, int) or version < 1:
            fail("unsafe original artifact path")
        name = f"{artifact_type}__v{version:03d}.json"
        path = f"artifacts/{RUN}/{stage.lower()}/{name}"
        raw = github_get(token, path, x["github_commit_sha"])
        if len(raw) > MAX_OBJECT:
            fail("source oversized")
        data = prove(raw, x, "historical")
        target = historical_dir / stage.lower()
        target.mkdir(mode=0o700, exist_ok=True)
        (target / name).write_bytes(raw)
        historical_proofs.append({
            "artifact_id": x["artifact_id"], "version": version,
            "stage": stage, "type": artifact_type, "source_commit": x["github_commit_sha"],
            **data,
        })
    summary = {
        "run_id": RUN, "historical_verified": len(historical_proofs),
        "candidate_verified": len(candidate_proofs),
        "historical_proofs": historical_proofs,
        "candidate_proofs": candidate_proofs,
        "postgresql_schneider_rpc": "NOT_EXECUTED",
        "trusted_loader_postcommit": "NOT_EXECUTED",
        "remote_persistence": False, "admission": "NO_GO",
    }
    (temp / "C07_PRIVATE_PROVENANCE_ONLY.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("C07 PRIVATE PRE-FLIGHT: 19/19 historical exact-byte proofs verified.")
    print("C07 PRIVATE PRE-FLIGHT: 3/3 local simulated candidates verified.")
    print("C07 SCHNEIDER SQL REPLAY=NOT_EXECUTED; V3.1 POSTCOMMIT=NOT_EXECUTED; NO_GO.")
    # No artifact upload: GitHub Actions logs are public for this repository.

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("C07 PRIVATE PRE-FLIGHT: " + str(exc), file=sys.stderr)
        sys.exit(1)
