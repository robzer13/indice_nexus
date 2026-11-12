import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { loadTrustedSchneiderShadowSnapshot, TrustedLoaderError,
  type ArtifactRow, type RunRow, type StageRow, type TrustedPilotagePort,
} from "../lib/orotitan-equity/v3/pilotage-trusted-loader-v3-1";
import { RESEARCH_V2_FINAL_OUTPUTS } from "../lib/orotitan-equity/v1/research-required-outputs";
import pins from "../contracts/orotitan-equity/v3/contract-pin-pack-v3/OROTITAN_CONTRACT_PIN_PACK_V3_0_2.json";

const RUN = "db00ce77-04eb-492b-a76b-164ecbe9ccfe";
const ISSUER = "22222222-2222-4222-8222-222222222222";
const SECURITY = "33333333-3333-4333-8333-333333333333";
const DOSSIER = "44444444-4444-4444-8444-444444444444";
const BASE = "2026-10-30";
const RESEARCH_MANIFEST = "55555555-5555-4555-8555-555555555555";
const DEEP_DIVE_MANIFEST = "66666666-6666-4666-8666-666666666666";
const REPO = "example/cryptographic-private-fixture";
const COMMIT = "a".repeat(40);
const ENV = {projectRef:"awgsurdyvsyolcgpnygh",projectName:"orotitan-vnext-shadow",
  region:"eu-west-3",status:"ACTIVE_HEALTHY"};

const json = (obj:unknown):Buffer => Buffer.from(JSON.stringify(obj),"utf8");
const sha = (data:Uint8Array) => createHash("sha256").update(data).digest("hex");
const blob = (data:Uint8Array) => createHash("sha1")
  .update(Buffer.from("blob "+data.length+"\0")).update(data).digest("hex");
const id = (n:number) => "77777777-7777-4777-8777-"+n.toString(16).padStart(12,"0");
const key = (a:Pick<ArtifactRow,"artifact_id"|"version">) => a.artifact_id+":"+a.version;
const ref = (a:ArtifactRow) =>
  ({artifact_id:a.artifact_id,version:a.version,content_sha256:a.content_sha256});

function row(stage:"RESEARCH"|"DEEP_DIVE", type:string, identifier:string,
  version:number, bytes:Buffer, parent:string|null, manifestVersion:number|null,
  mode:"FINAL"|"CHECKPOINT"|"SUPERSEDED"="CHECKPOINT"):ArtifactRow {
  const path="runs/"+RUN+"/"+identifier+"-v"+version+".json";
  return {run_id:RUN,stage_code:stage,artifact_id:identifier,version,artifact_type:type,
    authority_class:mode==="FINAL"?"AUTHORITATIVE_STAGE_OUTPUT":"CHECKPOINT_STAGE_OUTPUT",
    authority_state:mode==="FINAL"?"AUTHORITATIVE":mode==="SUPERSEDED"?"SUPERSEDED":"CHECKPOINT",
    artifact_status:"SEALED",availability_state:"AVAILABLE",media_type:"application/json",
    size_bytes:bytes.length,hash_algorithm:"SHA-256",content_sha256:sha(bytes),
    storage_backend:"PRIVATE_GITHUB",
    storage_uri:"github://"+REPO+"@"+COMMIT+"/"+path,github_repository:REPO,
    github_path:path,github_commit_sha:COMMIT,github_blob_sha:blob(bytes),
    manifest_artifact_id:parent,manifest_version:manifestVersion};
}
function manifestOutput(a:ArtifactRow){
  return {artifact_id:a.artifact_id,version:a.version,artifact_type:a.artifact_type,
    content_sha256:a.content_sha256,authority_class:a.authority_class,
    media_type:a.media_type,size_bytes:a.size_bytes,
    storage_ref:{backend:a.storage_backend,repository:a.github_repository,
      path:a.github_path,commit_sha:a.github_commit_sha,blob_sha:a.github_blob_sha,
      storage_uri:a.storage_uri}};
}
function baseManifest(stage:"RESEARCH"|"DEEP_DIVE",revision:number){
  return {
    manifest_schema_version:"1.0.0",manifest_id:stage==="RESEARCH"?RESEARCH_MANIFEST:DEEP_DIVE_MANIFEST,
    manifest_kind:stage==="RESEARCH"?"FINAL":"CHECKPOINT",
    run_id:RUN,stage,stage_revision:revision,issuer_id:ISSUER,security_id:SECURITY,
    dossier_id:DOSSIER,canonical_mode:"ANALYZE",run_type:"INITIAL",data_cutoff:BASE,
    baseline_snapshot_id:null,process_version:"3.1",pilotage_contract_version:"3.1",
    contract_pins:pins.contract_pins,contract_set_sha256:pins.contract_set_sha256,
    stage_contract:stage==="RESEARCH"?pins.contract_pins.research_stage:pins.contract_pins.deep_dive_stage,
    input_artifacts:[],output_artifacts:[],
    stage_status:stage==="RESEARCH"?"COMPLETE":"IN_PROGRESS",
    contract_status_code:null,handoff_gate:{name:stage==="RESEARCH"?"READY_FOR_DEEP_DIVE":"READY_FOR_INTEGRATION",
      state:stage==="RESEARCH"?"YES":"NOT_EVALUATED"},
    critical_blockers:[],open_material_limitations:[],
    parent_manifests:[],started_at:"2026-11-10T12:00:00Z",
    completed_at:stage==="RESEARCH"?"2026-11-10T13:00:00Z":null};
}
type Fixture = {run:RunRow;research:StageRow;deepDive:StageRow;rows:ArtifactRow[];
  data:Map<string,Buffer>;port:TrustedPilotagePort;researchManifest:ArtifactRow;
  checkpointV1:ArtifactRow;currentManifest:ArtifactRow;current:Record<string,unknown>};
function fixture(activeVersion:1|2):Fixture {
  const run:RunRow={run_id:RUN,issuer_id:ISSUER,security_id:SECURITY,dossier_id:DOSSIER,
    baseline_snapshot_id:null,canonical_mode:"ANALYZE",run_type:"INITIAL",run_status:"ACTIVE",
    current_stage:"DEEP_DIVE",data_cutoff:BASE,process_version:"3.1",pilotage_contract_version:"3.1",
    contract_pins:pins.contract_pins,contract_set_sha256:pins.contract_set_sha256,state_version:7};
  const research:StageRow={
    run_id:RUN,stage_code:"RESEARCH",stage_revision:1,
    stage_contract_name:pins.contract_pins.research_stage.name,
    stage_contract_version:pins.contract_pins.research_stage.version,
    stage_contract_sha256:pins.contract_pins.research_stage.content_sha256,
    lifecycle_status:"COMPLETE",handoff_gate_name:"READY_FOR_DEEP_DIVE",handoff_gate_state:"YES",
    active_manifest_artifact_id:RESEARCH_MANIFEST,active_manifest_version:1,
    active_manifest_kind:"FINAL",blocker_summary:[],state_version:3};
  const deepDive:StageRow={
    run_id:RUN,stage_code:"DEEP_DIVE",stage_revision:1,
    stage_contract_name:pins.contract_pins.deep_dive_stage.name,
    stage_contract_version:pins.contract_pins.deep_dive_stage.version,
    stage_contract_sha256:pins.contract_pins.deep_dive_stage.content_sha256,
    lifecycle_status:"IN_PROGRESS",handoff_gate_name:"READY_FOR_INTEGRATION",
    handoff_gate_state:"NOT_EVALUATED",active_manifest_artifact_id:DEEP_DIVE_MANIFEST,
    active_manifest_version:activeVersion,active_manifest_kind:"CHECKPOINT",
    blocker_summary:[],state_version:2};
  const rows:ArtifactRow[]=[];
  const data=new Map<string,Buffer>();
  function add(stage:"RESEARCH"|"DEEP_DIVE",type:string,identifier:string,version:number,
    bytes:Buffer,parent:string|null,manifestVersion:number|null,
    mode:"FINAL"|"CHECKPOINT"|"SUPERSEDED"="CHECKPOINT"){
    const a=row(stage,type,identifier,version,bytes,parent,manifestVersion,mode);
    rows.push(a);data.set(key(a),bytes);return a;
  }
  const researchOutputs=RESEARCH_V2_FINAL_OUTPUTS.map((type,i)=>
    add("RESEARCH",type,id(i+1),1,json({type,source:"real byte-verified fixture"}),
      RESEARCH_MANIFEST,1,"FINAL"));
  const researchJson={...baseManifest("RESEARCH",1),
    output_artifacts:researchOutputs.map(manifestOutput)};
  const researchManifest=add("RESEARCH","RESEARCH_STAGE_MANIFEST",RESEARCH_MANIFEST,1,
    json(researchJson),null,null,"FINAL");

  const checkpointParent={manifest_id:researchManifest.artifact_id,version:1,
    manifest_kind:"FINAL",content_sha256:researchManifest.content_sha256};
  const oldManifestJson={
    ...baseManifest("DEEP_DIVE",1),parent_manifests:[checkpointParent],
  };
  const checkpointV1=add("DEEP_DIVE","DEEP_DIVE_STAGE_MANIFEST",DEEP_DIVE_MANIFEST,1,
    json(oldManifestJson),null,null,activeVersion===1?"CHECKPOINT":"SUPERSEDED");

  const DD = (type:string,n:number,body:unknown) =>
    add("DEEP_DIVE",type,id(n),1,json(body),DEEP_DIVE_MANIFEST,activeVersion);
  const evidence=DD("EVIDENCE_LEDGER",50,{entries:[{claim:"Grounded statement"}]});
  const taxonomy=DD("CONTROLLED_TAXONOMY_RECORD",51,{taxonomy:"industrial"});
  const description=DD("BUSINESS_DESCRIPTION_SHORT_ARTIFACT",52,{description:"Equipment and software"});
  const scoring=DD("SCORING_INPUTS",53,{scores:{OQS:null,OVS:null},final_scoring_performed:false});
  const redTeam=DD("FUNDAMENTAL_RED_TEAM_RECORD",54,{status:"COMPLETE",adverse_test:"PRESENT"});
  const lockId=id(55);
  const lockVersion=activeVersion;
  const lockJson={
    lock_type:"FUNDAMENTALS_LOCK",artifact:{artifact_id:lockId,version:lockVersion},
    run_id:RUN,data_cutoff:BASE,stage_revision:1,
    research_final_manifest:ref(researchManifest),evidence_lineage:[ref(evidence)],
    red_team_status:"COMPLETE",taxonomy_record:ref(taxonomy),
    business_description_artifact:ref(description),scoring_inputs_artifact:ref(scoring),
    limitations:["Unknown All-in ROIC"],invalidation_triggers:["Material fact contradiction"],
    ready_for_valuation:"YES",
  };
  const lock=add("DEEP_DIVE","FUNDAMENTALS_LOCK",lockId,lockVersion,
    json(lockJson),DEEP_DIVE_MANIFEST,activeVersion);
  const current=activeVersion===2?{
    ...baseManifest("DEEP_DIVE",1),
    parent_manifests:[
      {artifact_id:DEEP_DIVE_MANIFEST,version:1,relation_type:"SUPERSEDES_PRIOR_CHECKPOINT"},
      checkpointParent,
    ],
    output_artifacts:[evidence,taxonomy,description,scoring,redTeam,lock].map(manifestOutput),
  }:{
    ...baseManifest("DEEP_DIVE",1),
    parent_manifests:[checkpointParent],
    output_artifacts:[evidence,taxonomy,description,scoring,redTeam,lock].map(manifestOutput),
  };
  if(activeVersion===1){
    // Use active V1 bytes with outputs instead of the minimal ancestor shell.
    const b=json(current);
    checkpointV1.size_bytes=b.length;checkpointV1.content_sha256=sha(b);
    checkpointV1.github_blob_sha=blob(b);data.set(key(checkpointV1),b);
  }
  const currentManifest=activeVersion===1?checkpointV1:
    add("DEEP_DIVE","DEEP_DIVE_STAGE_MANIFEST",DEEP_DIVE_MANIFEST,2,
      json(current),null,null);
  const port:TrustedPilotagePort={
    async getRun(){return {...run};},
    async getStages(){return [{...research},{...deepDive}];},
    async getArtifacts(){return rows.map(x=>({...x}));},
    async getIssuer(){return {issuer_id:ISSUER,display_name:"Synthetic Schneider fixture",legal_name:null};},
    async readImmutableBytes(artifact){
      const b=data.get(key(artifact));
      if(!b) throw new Error("unexpected bytes request");
      return b;
    },
  };
  return {run,research,deepDive,rows,data,port,researchManifest,checkpointV1,currentManifest,current};
}
async function load(f:Fixture){
  return loadTrustedSchneiderShadowSnapshot({port:f.port,environment:ENV,runId:RUN,
    expectedRunStateVersion:7,expectedCurrentStageStateVersion:2});
}
function rehash(f:Fixture,a:ArtifactRow,body:unknown){
  const b=json(body);a.content_sha256=sha(b);a.github_blob_sha=blob(b);a.size_bytes=b.length;
  f.data.set(key(a),b);
}
async function fails(f:Fixture,code:string){
  await assert.rejects(load(f),(e:unknown)=>
    e instanceof TrustedLoaderError && e.code===code,code);
}
test("V1 Deep Dive checkpoint verifies its exact Research FINAL bytes",async()=>{
  const f=fixture(1);
  const s=await load(f);
  assert.equal(s.stages.DEEP_DIVE?.manifest?.version,1);
  assert.equal(s.stages.DEEP_DIVE?.gates.READY_FOR_VALUATION,true);
  assert.equal(s.stages.DEEP_DIVE?.consumedPredecessorRevision,1);
});
test("V2 Deep Dive checkpoint cryptographically verifies both independent parents",async()=>{
  const f=fixture(2);
  const s=await load(f);
  assert.equal(s.stages.DEEP_DIVE?.manifest?.version,2);
  assert.equal(s.stages.DEEP_DIVE?.gates.READY_FOR_VALUATION,true);
  assert.equal(s.stages.DEEP_DIVE?.consumedPredecessorRevision,1);
  assert.equal(f.currentManifest.content_sha256,sha(f.data.get(key(f.currentManifest))!));
});
test("tampering the prior CHECKPOINT bytes fails closed",async()=>{
  const f=fixture(2);
  const b=Buffer.from(f.data.get(key(f.checkpointV1))!);b[12]^=1;
  f.data.set(key(f.checkpointV1),b);
  await fails(f,"ARTIFACT_SHA256_MISMATCH");
});
test("tampering Research FINAL bytes fails closed separately",async()=>{
  const f=fixture(2);
  const b=Buffer.from(f.data.get(key(f.researchManifest))!);b[17]^=1;
  f.data.set(key(f.researchManifest),b);
  await fails(f,"ARTIFACT_SHA256_MISMATCH");
});
test("forged Research FINAL digest in parent is rejected even when manifest is rehashed",async()=>{
  const f=fixture(2);
  (f.current.parent_manifests as Record<string,unknown>[])[1].content_sha256="f".repeat(64);
  rehash(f,f.currentManifest,f.current);
  await fails(f,"CHECKPOINT_RESEARCH_FINAL_HASH_MISMATCH");
});
test("missing Research FINAL parent fails closed",async()=>{
  const f=fixture(2);
  f.current.parent_manifests=[(f.current.parent_manifests as unknown[])[0]];
  rehash(f,f.currentManifest,f.current);
  await fails(f,"CHECKPOINT_PARENT_SHAPE_INVALID");
});
test("missing previous CHECKPOINT parent fails closed",async()=>{
  const f=fixture(2);
  f.current.parent_manifests=[(f.current.parent_manifests as unknown[])[1]];
  rehash(f,f.currentManifest,f.current);
  await fails(f,"CHECKPOINT_PARENT_SHAPE_INVALID");
});
test("duplicate Research parents cannot bypass predecessor proof",async()=>{
  const f=fixture(2);
  const second=(f.current.parent_manifests as Record<string,unknown>[])[1];
  f.current.parent_manifests=[second,second];
  rehash(f,f.currentManifest,f.current);
  await fails(f,"CHECKPOINT_PARENT_LINEAGES_INCOMPLETE");
});
test("forged previous-checkpoint relation rejected",async()=>{
  const f=fixture(2);
  (f.current.parent_manifests as Record<string,unknown>[])[0].relation_type="ARBITRARY";
  rehash(f,f.currentManifest,f.current);
  await fails(f,"CHECKPOINT_PARENT_SHAPE_INVALID");
});
test("wrong previous-checkpoint version rejected",async()=>{
  const f=fixture(2);
  (f.current.parent_manifests as Record<string,unknown>[])[0].version=2;
  rehash(f,f.currentManifest,f.current);
  await fails(f,"CHECKPOINT_PARENT_SHAPE_INVALID");
});
test("wrong Research manifest kind rejected",async()=>{
  const f=fixture(2);
  (f.current.parent_manifests as Record<string,unknown>[])[1].manifest_kind="CHECKPOINT";
  rehash(f,f.currentManifest,f.current);
  await fails(f,"DEEP_DIVE_PARENT_KIND_INVALID");
});
