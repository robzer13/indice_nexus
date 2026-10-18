import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { loadTrustedPilotageDecision, TrustedLoaderError,
  type ArtifactRow, type RunRow, type StageRow, type TrustedPilotagePort,
} from "../lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1";
import { projectV31CheckpointForSchema, CheckpointCompatError } from
  "../lib/orotitan-equity/v3/pilotage-checkpoint-compat-v3-1";
import { validateStageManifest } from "../lib/orotitan-equity/v1/stage-manifest";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

const ids = {
  run: "11111111-1111-4111-8111-111111111111",
  issuer: "22222222-2222-4222-8222-222222222222",
  security: "33333333-3333-4333-8333-333333333333",
  dossier: "44444444-4444-4444-8444-444444444444",
  manifest: "55555555-5555-4555-8555-555555555555",
  output: "66666666-6666-4666-8666-666666666666",
};
const repo = "example/immutable-private-store";
const commit = "a".repeat(40);
const env = {projectRef:"cugpgtzygqqlxetyeven",projectName:"orotitan-screener",
  region:"eu-west-3",status:"ACTIVE_HEALTHY"};
const bytesSha = (b: Uint8Array) => createHash("sha256").update(b).digest("hex");
const blobSha = (b: Uint8Array) => createHash("sha1")
  .update(Buffer.from("blob " + b.length + "\0")).update(b).digest("hex");
const toBuffer = (o: unknown) => Buffer.from(JSON.stringify(o));
type Fixture = {
  run: RunRow;
  stage: StageRow;
  rows: ArtifactRow[];
  raw: Record<string, unknown>;
  content: Map<string, Buffer>;
  port: TrustedPilotagePort;
};
const k = (r: {artifact_id:string;version:number}) => r.artifact_id+":"+r.version;
function artifact(id: string, version: number, type: string, body: Buffer,
  parentId: string|null, state="CHECKPOINT"): ArtifactRow {
  const path = "runs/" + ids.run + "/" + id + "-v" + version + ".json";
  return {
    artifact_id:id, version, run_id:ids.run, stage_code:"RESEARCH",
    artifact_type:type, authority_class:"CHECKPOINT_STAGE_OUTPUT",
    authority_state:state, artifact_status:"SEALED", availability_state:"AVAILABLE",
    media_type:"application/json", size_bytes:body.length,hash_algorithm:"SHA-256",
    content_sha256:bytesSha(body),storage_backend:"PRIVATE_GITHUB",
    storage_uri:"github://"+repo+"@"+commit+"/"+path,
    github_repository:repo,github_path:path,github_commit_sha:commit,
    github_blob_sha:blobSha(body),manifest_artifact_id:parentId,
    manifest_version:parentId===null?null:4,
  };
}
function fixture(): Fixture {
  const researchPin = pins.contract_pins.research_stage;
  const run: RunRow = {
    run_id:ids.run,issuer_id:ids.issuer,security_id:ids.security,dossier_id:ids.dossier,
    baseline_snapshot_id:null,canonical_mode:"ANALYZE",run_type:"INITIAL",
    run_status:"ACTIVE",current_stage:"RESEARCH",data_cutoff:"2026-10-12",
    process_version:"3.1",pilotage_contract_version:"3.1",
    contract_pins:pins.contract_pins,contract_set_sha256:pins.contract_set_sha256,state_version:6,
  };
  const stage: StageRow = {
    run_id:ids.run,stage_code:"RESEARCH",stage_revision:1,
    stage_contract_name:researchPin.name,stage_contract_version:researchPin.version,
    stage_contract_sha256:researchPin.content_sha256,
    lifecycle_status:"IN_PROGRESS",handoff_gate_name:"READY_FOR_DEEP_DIVE",
    handoff_gate_state:"NO",active_manifest_artifact_id:ids.manifest,
    active_manifest_version:4,active_manifest_kind:"CHECKPOINT",
    blocker_summary:["Missing independent customer evidence","Denominator not reconciled"],
    state_version:3,
  };
  const outputBody = toBuffer({type:"RESEARCH_SOURCE_MANIFEST",not_admitted:true});
  const output = artifact(ids.output,1,"RESEARCH_SOURCE_MANIFEST",outputBody,ids.manifest);
  const previousBody = toBuffer({
    manifest_id:ids.manifest,manifest_kind:"CHECKPOINT",stage:"RESEARCH",
    stage_revision:1,run_id:ids.run,critical_blockers:["Earlier unresolved issue"],
  });
  const previous = artifact(ids.manifest,3,"RESEARCH_STAGE_MANIFEST",previousBody,null,"SUPERSEDED");
  const raw = {
    manifest_schema_version:"1.0.0",manifest_id:ids.manifest,manifest_kind:"CHECKPOINT",
    run_id:ids.run,stage:"RESEARCH",stage_revision:1,
    issuer_id:ids.issuer,security_id:ids.security,dossier_id:ids.dossier,
    canonical_mode:"ANALYZE",run_type:"INITIAL",data_cutoff:"2026-10-12",
    baseline_snapshot_id:null,process_version:"3.1",pilotage_contract_version:"3.1",
    contract_pins:pins.contract_pins,
    stage_contract:{name:researchPin.name,version:researchPin.version,
      content_sha256:researchPin.content_sha256},
    contract_set_sha256:pins.contract_set_sha256,
    input_artifacts:[],
    output_artifacts:[{
      artifact_id:output.artifact_id,version:output.version,
      artifact_type:output.artifact_type,content_sha256:output.content_sha256,
      authority_class:output.authority_class,media_type:output.media_type,
      size_bytes:output.size_bytes,
      storage_ref:{
        backend:output.storage_backend,repository:output.github_repository,
        path:output.github_path,commit_sha:output.github_commit_sha,
        blob_sha:output.github_blob_sha,storage_uri:output.storage_uri,
      },
    }],
    stage_status:"IN_PROGRESS",contract_status_code:"BLOCKED_INSUFFICIENT_INPUT",
    handoff_gate:{name:"READY_FOR_DEEP_DIVE",state:"NO"},
    critical_blockers:stage.blocker_summary,
    open_material_limitations:["Unreconciled independent source"],
    parent_manifests:[{artifact_id:ids.manifest,version:3,
      relation_type:"SUPERSEDES_PRIOR_CHECKPOINT"}],
    started_at:"2026-10-12T10:03:55.417717+00:00",completed_at:null,
  };
  const current = artifact(ids.manifest,4,"RESEARCH_STAGE_MANIFEST",toBuffer(raw),null);
  const rows = [previous,output,current];
  const content = new Map(rows.map((row,i) => [k(row),
    i===0?previousBody:i===1?outputBody:toBuffer(raw)]));
  const port: TrustedPilotagePort = {
    async getRun(){return {...run};},
    async getStages(){return [{...stage}];},
    async getArtifacts(){return rows.map(x=>({...x}));},
    async getIssuer(){return {issuer_id:ids.issuer,display_name:"Example Company",legal_name:null};},
    async readImmutableBytes(row){
      const bytes=content.get(k(row));
      if(!bytes)throw new Error("fixture has no content for "+k(row));
      return bytes;
    },
  };
  return {run,stage,rows,raw,content,port};
}
function resignCurrent(f: Fixture) {
  const row=f.rows.find(x=>x.artifact_id===ids.manifest&&x.version===4)!;
  const b=toBuffer(f.raw);
  row.size_bytes=b.length;row.content_sha256=bytesSha(b);row.github_blob_sha=blobSha(b);
  f.content.set(k(row),b);
}
async function load(f: Fixture) {
  return loadTrustedPilotageDecision({
    port:f.port,environment:env,runId:ids.run,
    expectedRunStateVersion:6,expectedCurrentStageStateVersion:3,
  });
}
async function rejects(f: Fixture,code: string) {
  await assert.rejects(load(f),(e:unknown)=>e instanceof TrustedLoaderError &&
    e.code===code,code);
}

test("real-shaped V4 RESEARCH checkpoint is verified without handoff/publication",async()=>{
  const f=fixture();
  const rawBefore=JSON.stringify(f.raw);
  const result=await load(f);
  assert.notEqual(result.outcome,"HANDOFF");
  assert.notEqual(result.outcome,"AWAIT_EXPLICIT_GO_PUBLISH");
  assert.equal(result.publicationAuthorized,false);
  assert.equal(result.mutationPerformed,false);
  assert.equal(JSON.stringify(f.raw),rawBefore);
  // The loader first applies its existing, separately pinned RAW_CANONICAL_V1
  // locator projection. Mimic that *schema-only* step without mutating the raw.
  const schemaInput = JSON.parse(JSON.stringify(f.raw)) as Record<string,unknown>;
  for (const key of ["investment_policy","execution_patch"]) {
    const pin=(schemaInput.contract_pins as Record<string,{locator:Record<string,unknown>}>)[key];
    if (pin.locator.locator_format === "RAW_CANONICAL_V1") delete pin.locator.locator_format;
  }
  const projected=projectV31CheckpointForSchema({
    manifest:schemaInput,run:f.run,stage:f.stage,activeManifest:f.rows[2],
    verifiedParents:[{artifactId:ids.manifest,version:3,sha256:f.rows[0].content_sha256,
      runId:ids.run,stageCode:"RESEARCH"}],
  });
  assert.equal((projected.critical_blockers as {summary:string}[])[0].summary,
    "Missing independent customer evidence");
  assert.deepEqual(f.raw.critical_blockers,f.stage.blocker_summary);
  assert.equal((projected.parent_manifests as {content_sha256:string}[])[0].content_sha256,
    f.rows[0].content_sha256);
  assert.equal(validateStageManifest(projected,{
    runId:f.run.run_id,stage:"RESEARCH",stageRevision:1,issuerId:f.run.issuer_id,
    securityId:f.run.security_id,dossierId:f.run.dossier_id,
    canonicalMode:f.run.canonical_mode,runType:f.run.run_type,dataCutoff:f.run.data_cutoff,
    baselineSnapshotId:f.run.baseline_snapshot_id,processVersion:"3.1",
    pilotageContractVersion:"3.1",contractSetSha256:f.run.contract_set_sha256,
    stageContractName:f.stage.stage_contract_name,
    stageContractVersion:f.stage.stage_contract_version,
    stageContractSha256:f.stage.stage_contract_sha256,
  }).ok,true);
});
test("tampered previous checkpoint bytes fail closed",async()=>{
  const f=fixture();
  const r=f.rows[0],original=f.content.get(k(r))!;
  const tampered=Buffer.from(original);tampered[8]^=1;
  f.content.set(k(r),tampered);
  await rejects(f,"ARTIFACT_SHA256_MISMATCH");
});
test("parent missing from Registry fails, no synthetic SHA",async()=>{
  const f=fixture();f.rows.splice(0,1);
  await rejects(f,"CHECKPOINT_PARENT_REFERENCE_UNVERIFIED");
});
test("incorrect parent version or relation refused",async()=>{
  const f=fixture();
  (f.raw.parent_manifests as Record<string,unknown>[])[0].version=2;
  resignCurrent(f);
  await rejects(f,"CHECKPOINT_PARENT_REFERENCE_UNVERIFIED");
  const x=fixture();
  (x.raw.parent_manifests as Record<string,unknown>[])[0].relation_type="FORGED_PUBLISH";
  resignCurrent(x);
  await rejects(x,"CHECKPOINT_PARENT_SHAPE_INVALID");
});
test("stage contract conflict refused even if hash of the manifest is valid",async()=>{
  const f=fixture();
  (f.raw.stage_contract as Record<string,unknown>).content_sha256="f".repeat(64);
  resignCurrent(f);
  await rejects(f,"CHECKPOINT_STAGE_CONTRACT_NOT_PINNED");
});
test("unsupported blocker or unknown schema key fails closed",async()=>{
  const f=fixture();(f.raw.critical_blockers as string[])[0]="";
  f.stage.blocker_summary=f.raw.critical_blockers;
  resignCurrent(f);
  await rejects(f,"CHECKPOINT_BLOCKER_STRING_INVALID");
  const f2=fixture();f2.raw.unapproved_extension="YES";
  resignCurrent(f2);
  await rejects(f2,"MANIFEST_SCHEMA_OR_CONTRACT_INVALID");
});
test("FINAL manifests are never coerced from legacy CHECKPOINT strings",async()=>{
  const f=fixture();
  const row=f.rows[2];
  f.raw.manifest_kind="FINAL";f.stage.active_manifest_kind="FINAL";
  row.authority_class="AUTHORITATIVE_STAGE_OUTPUT";row.authority_state="AUTHORITATIVE";
  resignCurrent(f);
  await rejects(f,"MANIFEST_SCHEMA_OR_CONTRACT_INVALID");
});
test("a foreign run cannot project a matching checkpoint shape",()=>{
  const f=fixture();
  assert.throws(()=>projectV31CheckpointForSchema({
    manifest:f.raw,run:{...f.run,process_version:"2.0"},stage:f.stage,
    activeManifest:f.rows[2],verifiedParents:[],
  }),(e:unknown)=>e instanceof CheckpointCompatError &&
    e.code==="CHECKPOINT_PROJECTION_SCOPE_INVALID");
});
