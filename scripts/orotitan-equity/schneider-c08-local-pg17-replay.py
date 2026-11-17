#!/usr/bin/env python3
"""C08: actual Schneider RPC replay in a disposable LOCAL PostgreSQL 17 only.

Private snapshot + verified immutable bytes never leave the protected runner.
The three C03 outputs are LOCALLY SIMULATED, NOT GitHub-attested authority.
All psql output is private, never printed; public logs show fixed verdicts only.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

RUN = "db00ce77-04eb-492b-a76b-164ecbe9ccfe"
REPO = "robzer13/real-orotitan"
EXPECTED = {
    "issuers": 1, "securities": 1, "research_dossiers": 1,
    "orotitan_runs": 1, "orotitan_run_stages": 2,
    "orotitan_artifacts": 28, "orotitan_artifact_edges": 8,
    "orotitan_run_events": 35,
    "orotitan_p0e_run_admission_receipts_v1": 1,
}
CANDIDATES = {
    ("SCORING_INPUTS", 1): "SCORING_INPUTS__v001.json",
    ("FUNDAMENTALS_LOCK", 2): "FUNDAMENTALS_LOCK__v002.json",
    ("DEEP_DIVE_STAGE_MANIFEST", 2): "DEEP_DIVE_STAGE_MANIFEST__v002.json",
}
SHA40 = re.compile(r"^[a-f0-9]{40}$")
SHA64 = re.compile(r"^[a-f0-9]{64}$")
UUID = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$")
ALLOWED_TABLES = tuple(EXPECTED)
PRIVATE_FOLDER = "orotitan-schneider-c07-private"


class Blocked(Exception):
    pass


def guard(ok: bool, category: str) -> None:
    if not ok:
        raise Blocked(category)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def decode(raw: bytes) -> dict:
    value = json.loads(raw.decode("utf-8"))
    guard(isinstance(value, dict), "JSON_OBJECT_EXPECTED")
    return value


def raw_rows(path: Path, count: int) -> list[dict]:
    guard(path.is_file() and 0 < path.stat().st_size <= 2_500_000, "SNAPSHOT_FILE_MISSING")
    rows = [decode(line) for line in path.read_bytes().splitlines() if line.strip()]
    guard(len(rows) == count, "SNAPSHOT_COUNT_MISMATCH")
    return rows


def q(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def j(value: object) -> str:
    return q(json.dumps(value, ensure_ascii=True, separators=(",", ":"))) + "::jsonb"


class Replay:
    def __init__(self, database: str, root: Path):
        guard(re.fullmatch(r"orotitan_registry_v1_[0-9]+", database) is not None,
              "UNRECOGNIZED_LOCAL_DATABASE")
        guard(os.environ.get("PGHOST") == "127.0.0.1" and
              os.environ.get("PGPORT") == "5432" and
              os.environ.get("PGUSER") == "postgres" and
              os.environ.get("PGPASSWORD") == "postgres",
              "LOCAL_POSTGRES_ONLY")
        guard(root.name == PRIVATE_FOLDER and root.is_dir(), "PRIVATE_INPUT_DIR_INVALID")
        self.root = root
        self.db = database
        # Explicit allowlist: do NOT forward SHADOW credentials, private GitHub token,
        # GitHub OIDC env, DB SSL root, or any workflow secret to local psql.
        self.env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": os.environ.get("HOME", "/tmp"),
                    "LANG": "C.UTF-8", "PGHOST": "127.0.0.1",
                    "PGPORT": "5432", "PGUSER": "postgres",
                    "PGPASSWORD": "postgres", "PGDATABASE": database,
                    "PGSSLMODE": "disable"}
        self.rows = {name: raw_rows(root / (name + ".jsonl"), count)
                     for name, count in EXPECTED.items()}
        self.raw_candidate = {}
        for key, name in CANDIDATES.items():
            path = root / name
            guard(path.is_file() and path.stat().st_size < 300_000,
                  "CANDIDATE_BYTES_MISSING")
            self.raw_candidate[key] = path.read_bytes()
        self.manifest = decode(self.raw_candidate[("DEEP_DIVE_STAGE_MANIFEST", 2)])
        simulation_file = root / "SIMULATED_CANONICAL_LOCATOR_EVIDENCE.json"
        guard(simulation_file.is_file(), "LOCAL_C03_SIMULATION_PROOF_MISSING")
        self.local_simulation = decode(simulation_file.read_bytes())
        guard(self.local_simulation.get("run_id") == RUN
              and self.local_simulation.get("status") == "SIMULATION_ONLY_NOT_PERSISTED",
              "LOCAL_C03_SIMULATION_STATUS")
        self.historical = {(x["artifact_id"], int(x["version"])): x
                           for x in self.rows["orotitan_artifacts"]}
        self.events = self.rows["orotitan_run_events"]
        self.verify_snapshot()

    def sql(self, statement: str, stage: str) -> str:
        command = ["psql", "-X", "-w", "-A", "-t", "-q",
                   "-v", "ON_ERROR_STOP=1", "-d", self.db]
        try:
            result = subprocess.run(command, input=statement, text=True,
                                    capture_output=True, env=self.env,
                                    timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise Blocked(stage + "_EXECUTION_UNAVAILABLE") from None
        if result.returncode:
            # Never disclose psql stderr (can contain private SQL payloads).
            raise Blocked(stage + "_SQL_REJECTED")
        return result.stdout.strip()

    def sql_expected_error(self, statement: str, expected: str, stage: str) -> None:
        command = ["psql", "-X", "-w", "-A", "-t", "-q",
                   "-v", "ON_ERROR_STOP=1", "-d", self.db]
        result = subprocess.run(command, input=statement, text=True,
                                capture_output=True, env=self.env,
                                timeout=120, check=False)
        guard(result.returncode != 0 and expected in result.stderr,
              stage + "_FAIL_CLOSED_EXPECTATION_MISSED")

    def verify_snapshot(self) -> None:
        run = self.rows["orotitan_runs"][0]
        stages = self.rows["orotitan_run_stages"]
        guard(run["run_id"] == RUN and run["state_version"] == 7
              and run["run_status"] == "ACTIVE" and run["current_stage"] == "DEEP_DIVE",
              "RUN_CAS_BASELINE")
        deep = [r for r in stages if r["stage_code"] == "DEEP_DIVE"]
        guard(len(deep) == 1 and deep[0]["state_version"] == 2
              and deep[0]["active_manifest_kind"] == "CHECKPOINT"
              and deep[0]["active_manifest_version"] == 1,
              "DEEP_DIVE_CAS_BASELINE")
        guard(sum(x["event_type"] == "PERSISTENCE_ATTESTED" for x in self.events) == 28,
              "HISTORICAL_ATTESTATION_BASELINE")
        guard(self.manifest.get("run_id") == RUN
              and self.manifest.get("stage") == "DEEP_DIVE"
              and self.manifest.get("manifest_kind") == "CHECKPOINT"
              and len(self.manifest.get("output_artifacts", [])) == 11
              and len(self.manifest.get("parent_manifests", [])) == 2,
              "MANIFEST_V2_SHAPE")
        sim = self.raw_candidate
        refs = {(x["artifact_type"], int(x["version"])): x
                for x in self.manifest["output_artifacts"]}
        guard(len(refs) == 11, "DUPLICATE_MANIFEST_OUTPUT")
        for key in list(CANDIDATES)[:2]:
            guard(key in refs and refs[key]["content_sha256"] == sha(sim[key])
                  and refs[key]["size_bytes"] == len(sim[key])
                  and refs[key]["storage_ref"]["blob_sha"] == git_blob(sim[key]),
                  "CANDIDATE_LOCATOR_PROOF")
        guard(all(r.get("run_id") == RUN for r in self.rows["orotitan_run_events"]),
              "CROSS_RUN_EVENT")
        guard(len(self.historical) == 28, "DUPLICATE_HISTORICAL_ARTIFACT")

    def restore(self) -> None:
        # Local mirror of observed P0E receipt relation; no remote DDL. The nine
        # snapshot files are imported without rewriting columns or disabling FKs.
        mirror = r"""
CREATE TABLE public.orotitan_p0e_run_admission_receipts_v1 (
  run_id uuid primary key references public.orotitan_runs(run_id) on delete restrict,
  creation_idempotency_key text not null unique,
  issuer_id uuid not null references public.issuers(issuer_id) on delete restrict,
  security_id uuid not null references public.securities(security_id) on delete restrict,
  dossier_id uuid not null references public.research_dossiers(dossier_id) on delete restrict,
  data_cutoff date not null,
  profile_id text not null check (profile_id='OROTITAN_V3_1'),
  contract_set_sha256 text not null,
  verification_receipt jsonb not null,
  authority text not null check (authority='USER_GO_P0E_SHADOW_INITIAL_V3_1'),
  publication_authorized boolean not null default false
    check (publication_authorized=false),
  created_at timestamptz not null default now()
);
"""
        statements = ["BEGIN;", "SET CONSTRAINTS ALL DEFERRED;", mirror]
        order = ("issuers", "securities", "research_dossiers", "orotitan_runs",
                 "orotitan_run_stages", "orotitan_artifacts", "orotitan_artifact_edges",
                 "orotitan_run_events", "orotitan_p0e_run_admission_receipts_v1")
        for table in order:
            rows = self.rows[table]
            values = ",".join("(" + q(base64.b64encode(json.dumps(r).encode()).decode())
                              + ")" for r in rows)
            statements.append(
                "INSERT INTO public." + table +
                " SELECT (jsonb_populate_record(NULL::public." + table +
                ", convert_from(decode(x.data,'base64'),'UTF8')::jsonb)).*" +
                " FROM (VALUES " + values + ") AS x(data);"
            )
        statements.append(
            "DO $$ BEGIN "
            "IF (SELECT count(*) FROM public.orotitan_runs WHERE run_id='"
            + RUN + "'::uuid AND state_version=7) <> 1 "
            "OR (SELECT count(*) FROM public.orotitan_artifacts WHERE run_id='"
            + RUN + "'::uuid) <> 28 THEN "
            "RAISE EXCEPTION 'LOCAL_SNAPSHOT_NOT_RESTORED'; END IF; END $$;"
        )
        statements.append("COMMIT;")
        self.sql("\n".join(statements), "RESTORE")
        print("C08 LOCAL_PG17_AUTHENTIC_9_TABLE_RESTORE=PASS", flush=True)

    def event_for(self, row: dict) -> dict:
        matching = [
            e for e in self.events if e["event_type"] == "PERSISTENCE_ATTESTED"
            and e["stage_code"] == row["stage_code"]
            and e["payload"].get("artifact_id") == row["artifact_id"]
            and int(e["payload"].get("version", -1)) == int(row["version"])
        ]
        guard(len(matching) == 1, "HISTORICAL_ATTESTATION_NOT_UNIQUE")
        event = matching[0]
        payload = event["payload"]
        guard(payload.get("content_sha256") == row["content_sha256"]
              and payload.get("github_blob_sha") == row["github_blob_sha"]
              and payload.get("github_commit_sha") == row["github_commit_sha"],
              "HISTORICAL_EVENT_IDENTITY")
        return event

    def metadata_for(self, ref: dict, local: bool) -> tuple[dict, bytes, dict | None]:
        key = ref["artifact_type"], int(ref["version"])
        if local:
            raw = self.raw_candidate[key]
            storage = ref["storage_ref"]
            meta = dict(
                artifact_id=ref["artifact_id"], version=ref["version"],
                artifact_type=ref["artifact_type"],
                logical_name=ref["artifact_type"].lower(),
                authority_class="CHECKPOINT_STAGE_OUTPUT",
                authority_state="NON_AUTHORITATIVE",
                artifact_status="SEALED", availability_state="AVAILABLE",
                media_type="application/json", size_bytes=len(raw),
                content_sha256=sha(raw), hash_algorithm="SHA-256",
                storage_backend="PRIVATE_GITHUB",
                storage_uri=storage["storage_uri"],
                github_repository=storage["repository"], github_path=storage["path"],
                github_commit_sha=storage["commit_sha"],
                github_blob_sha=storage["blob_sha"]
            )
            guard(meta["github_repository"] == REPO
                  and SHA40.fullmatch(meta["github_commit_sha"]) is not None,
                  "CANDIDATE_SIMULATED_REPO")
            return meta, raw, None
        key_id = ref["artifact_id"], int(ref["version"])
        guard(key_id in self.historical, "HISTORICAL_OUTPUT_REGISTRY_MISSING")
        row = self.historical[key_id]
        guard(row["artifact_type"] == key[0] and row["stage_code"] == "DEEP_DIVE"
              and row["authority_class"] == "CHECKPOINT_STAGE_OUTPUT",
              "HISTORICAL_OUTPUT_AUTHORITY")
        path = self.root / "historical" / "deep_dive" / (
            row["artifact_type"] + "__v" + str(row["version"]).zfill(3) + ".json")
        guard(path.is_file() and path.stat().st_size < 300_000,
              "HISTORICAL_EXACT_BYTES_MISSING")
        raw = path.read_bytes()
        fields = ("artifact_id", "version", "artifact_type", "logical_name",
                  "authority_class", "authority_state", "artifact_status",
                  "availability_state", "media_type", "size_bytes",
                  "hash_algorithm", "content_sha256", "storage_backend",
                  "storage_uri", "github_repository", "github_path",
                  "github_commit_sha", "github_blob_sha")
        meta = {k: row[k] for k in fields}
        guard(meta["size_bytes"] == len(raw)
              and meta["content_sha256"] == sha(raw)
              and meta["github_blob_sha"] == git_blob(raw)
              and meta["github_repository"] == REPO,
              "HISTORICAL_BYTES_MISMATCH")
        return meta, raw, self.event_for(row)

    def stage_local_candidate(self, meta: dict, raw: bytes) -> dict:
        guard(meta["github_blob_sha"] == git_blob(raw)
              and meta["content_sha256"] == sha(raw)
              and meta["size_bytes"] == len(raw)
              and meta["github_repository"] == REPO,
              "LOCAL_CANDIDATE_HASH")
        metadata = dict(meta)
        self.sql("SELECT public.stage_orotitan_persistence_candidate("
                 + q(RUN) + "::uuid,'DEEP_DIVE'," + j(metadata) + ");",
                 "STAGE_LOCAL")
        verified_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "attestation_schema_version": "1.0",
            "verification_method": "GITHUB_CONNECTOR_PRIVATE_REREAD_V1",
            "trust_boundary": "SUPABASE_MANAGEMENT_PLANE",
            "storage_backend": "PRIVATE_GITHUB",
            "run_id": RUN, "stage_code": "DEEP_DIVE",
            "artifact_id": meta["artifact_id"], "version": meta["version"],
            "artifact_type": meta["artifact_type"],
            "github_repository": REPO,
            "github_path": meta["github_path"],
            "github_commit_sha": meta["github_commit_sha"],
            "github_blob_sha": meta["github_blob_sha"],
            "storage_uri": meta["storage_uri"],
            "size_bytes": meta["size_bytes"],
            "content_sha256": meta["content_sha256"],
            "commit_path_resolved": True,
            "verified_at": verified_at,
        }
        # CRITICAL: this payload is an explicitly LOCAL-ONLY test simulation.
        # No remote GitHub candidate commit is being asserted or created.
        eid = self.sql("SELECT public.attest_orotitan_persistence_locator("
                       + q(RUN) + "::uuid,'DEEP_DIVE'," + j(payload) + ")::text;",
                       "ATTEST_LOCAL")
        guard(UUID.fullmatch(eid) is not None, "LOCAL_RECEIPT_NOT_CREATED")
        return {"event_id": eid, "payload": payload}

    def receipt(self, meta: dict, raw: bytes, event: dict) -> dict:
        payload = event["payload"]
        return {
            "receipt_schema_version": "1.1",
            "verification_method": "PRIVATE_GITHUB_ATTESTED_REREAD_EXACT_BYTES_V1",
            "storage_backend": "PRIVATE_GITHUB",
            "run_id": RUN, "stage_code": "DEEP_DIVE",
            "artifact_id": meta["artifact_id"], "version": meta["version"],
            "artifact_type": meta["artifact_type"],
            "github_repository": meta["github_repository"],
            "github_path": meta["github_path"],
            "github_commit_sha": meta["github_commit_sha"],
            "github_blob_sha": meta["github_blob_sha"],
            "attestation_event_id": event["event_id"],
            "commit_path_resolved": True,
            "verified_content_base64": base64.b64encode(raw).decode(),
            "verified_at": payload["verified_at"],
        }

    def make_manifest_meta(self) -> tuple[dict, bytes]:
        raw = self.raw_candidate[("DEEP_DIVE_STAGE_MANIFEST", 2)]
        simulation_path = self.root / "C07_PRIVATE_PROVENANCE_ONLY.json"
        guard(simulation_path.is_file(), "PROVENANCE_ONLY_PROOF_MISSING")
        proof = decode(simulation_path.read_bytes())
        guard(proof.get("historical_verified") == 19 and proof.get("candidate_verified") == 3,
              "PROVENANCE_COUNTS_MISMATCH")
        # Project-local Git locator is intentionally simulated and never promoted.
        commit = self.local_simulation.get("git_commit_local_manifest", "")
        guard(SHA40.fullmatch(commit) is not None, "SIMULATED_COMMIT_NOT_PINNED")
        path = "artifacts/" + RUN + "/deep_dive/DEEP_DIVE_STAGE_MANIFEST__v002.json"
        return {
            "artifact_id": self.manifest["manifest_id"], "version": 2,
            "artifact_type": "DEEP_DIVE_STAGE_MANIFEST",
            "logical_name": "deep_dive_stage_manifest",
            "authority_class": "CHECKPOINT_STAGE_OUTPUT",
            "authority_state": "NON_AUTHORITATIVE",
            "artifact_status": "SEALED", "availability_state": "AVAILABLE",
            "media_type": "application/json", "size_bytes": len(raw),
            "hash_algorithm": "SHA-256", "content_sha256": sha(raw),
            "storage_backend": "PRIVATE_GITHUB",
            "storage_uri": "github://" + REPO + "@" + commit + "/" + path,
            "github_repository": REPO, "github_path": path,
            "github_commit_sha": commit, "github_blob_sha": git_blob(raw),
        }, raw

    def candidate_bundle(self) -> dict:
        registrations = []
        for ref in self.manifest["output_artifacts"]:
            key = ref["artifact_type"], int(ref["version"])
            locally_simulated = key in CANDIDATES
            guard(key != ("DEEP_DIVE_STAGE_MANIFEST", 2),
                  "MANIFEST_CANNOT_BE_ITS_OWN_OUTPUT")
            meta, raw, event = self.metadata_for(ref, locally_simulated)
            expected_ref = ref["storage_ref"]
            guard(meta["artifact_id"] == ref["artifact_id"]
                  and meta["content_sha256"] == ref["content_sha256"]
                  and meta["size_bytes"] == ref["size_bytes"]
                  and meta["github_path"] == expected_ref["path"]
                  and meta["github_blob_sha"] == expected_ref["blob_sha"]
                  and meta["github_commit_sha"] == expected_ref["commit_sha"]
                  and meta["storage_uri"] == expected_ref["storage_uri"],
                  "OUTPUT_REF_MISMATCH")
            if locally_simulated:
                event = self.stage_local_candidate(meta, raw)
            assert event is not None
            registered = dict(meta)
            registered["authority_state"] = "CHECKPOINT"
            registered["canonical_json_content"] = decode(raw)
            registered["persistence_receipt"] = self.receipt(meta, raw, event)
            registrations.append(registered)
        manifest_meta, manifest_raw = self.make_manifest_meta()
        manifest_event = self.stage_local_candidate(manifest_meta, manifest_raw)
        manifest_registration = dict(manifest_meta)
        manifest_registration["authority_state"] = "CHECKPOINT"
        manifest_registration["canonical_json_content"] = self.manifest
        manifest_registration["persistence_receipt"] = self.receipt(
            manifest_meta, manifest_raw, manifest_event)
        edges = []
        for ref in self.manifest["output_artifacts"]:
            edges.append({
                "child_run_id": RUN,
                "child_artifact_id": self.manifest["manifest_id"],
                "child_version": 2,
                "parent_run_id": RUN,
                "parent_artifact_id": ref["artifact_id"],
                "parent_version": ref["version"],
                "relation_type": "CONSUMES",
            })
        prev = [x for x in self.manifest["parent_manifests"]
                if x.get("relation_type") == "SUPERSEDES_PRIOR_CHECKPOINT"]
        guard(len(prev) == 1 and int(prev[0]["version"]) == 1,
              "SUPERSEDES_PARENT_MISSING")
        edges.append({
            "child_run_id": RUN,
            "child_artifact_id": self.manifest["manifest_id"],
            "child_version": 2,
            "parent_run_id": RUN,
            "parent_artifact_id": prev[0]["artifact_id"],
            "parent_version": 1,
            "relation_type": "SUPERSEDES",
        })
        return {
            "manifest": self.manifest,
            "manifest_registration": manifest_registration,
            "output_artifacts": registrations,
            "edges": edges,
        }

    def negative_cas(self) -> None:
        bad = ("SELECT public.checkpoint_orotitan_stage(" + q(RUN) +
               "::uuid,'DEEP_DIVE',6,2,'{}'::jsonb,'{}'::jsonb,"
               "'[]'::jsonb,'[]'::jsonb,'IN_PROGRESS',"
               + q("c08-invalid-run-cas") + "," + q("f"*64) + ",'PILOTAGE');")
        self.sql_expected_error(bad, "RUN_STATE_VERSION_MISMATCH", "NEGATIVE_RUN_CAS")
        bad2 = bad.replace("'DEEP_DIVE',6,2,", "'DEEP_DIVE',7,1,")
        self.sql_expected_error(bad2, "STAGE_STATE_VERSION_MISMATCH",
                                "NEGATIVE_STAGE_CAS")
        print("C08 LOCAL_PG17_REAL_RUN_CAS_REJECT=PASS", flush=True)

    def positive_replay(self, b: dict) -> None:
        fingerprint = sha(json.dumps(b, sort_keys=True).encode())
        statement = ("BEGIN;\nSELECT public.checkpoint_orotitan_stage(" +
                     q(RUN) + "::uuid,'DEEP_DIVE',7,2," +
                     j(b["manifest"]) + "," +
                     j(b["manifest_registration"]) + "," +
                     j(b["output_artifacts"]) + "," +
                     j(b["edges"]) + ",'IN_PROGRESS'," +
                     q("c08-local-schneider-v2-replay") + "," +
                     q(fingerprint) + ",'DEEP_DIVE_WORKER');\nCOMMIT;")
        self.sql(statement, "POSITIVE_REAL_RPC")
        result = self.sql(
            "SELECT r.state_version::text || ':' || s.state_version::text || ':' || "
            "s.active_manifest_version::text || ':' || "
            "(SELECT count(*) FROM public.orotitan_artifacts a "
            " WHERE a.run_id=r.run_id AND a.artifact_type='FUNDAMENTALS_LOCK' "
            " AND a.version=2 AND a.authority_state='CHECKPOINT')::text "
            "FROM public.orotitan_runs r JOIN public.orotitan_run_stages s "
            "ON s.run_id=r.run_id AND s.stage_code='DEEP_DIVE' "
            "WHERE r.run_id=" + q(RUN) + "::uuid;",
            "POSTCOMMIT")
        guard(result == "8:3:2:1", "CAS_8_3_OR_LOCK_V2_NOT_COMMITTED")
        print("C08 LOCAL_PG17_SCHNEIDER_CAS_7_2_TO_8_3=PASS", flush=True)

    def run(self) -> None:
        version = self.sql("SHOW server_version_num;", "SERVER_VERSION")
        guard(version.isdigit() and 170000 <= int(version) < 180000,
              "POSTGRESQL_17_REQUIRED")
        self.restore()
        self.negative_cas()
        bundle = self.candidate_bundle()
        self.positive_replay(bundle)
        print("C08 LOCAL_SIMULATED_C03_RECEIPTS=LOCAL_ONLY_NOT_REMOTE_AUTHORITY", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", required=True)
    ap.add_argument("--input-dir", required=True, type=Path)
    args = ap.parse_args()
    Replay(args.database, args.input_dir.resolve()).run()


if __name__ == "__main__":
    try:
        main()
    except (Blocked, KeyError, ValueError, TypeError, AssertionError, OSError,
            UnicodeDecodeError, json.JSONDecodeError) as exc:
        # No paths, values, SQL error text, payloads, or secrets in PUBLIC logs.
        category = str(exc) if isinstance(exc, Blocked) else "INPUT_OR_RUNTIME_INVALID"
        if not re.fullmatch(r"[A-Z0-9_]+", category):
            category = "INPUT_OR_RUNTIME_INVALID"
        print("BLOCKED_C08_" + category, file=sys.stderr)
        sys.exit(1)
