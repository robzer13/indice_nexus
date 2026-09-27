import { execFileSync } from "node:child_process";
import os from "node:os";

const MODEL = "qwen3.5:4b-q4_K_M";
const EXPECTED_DIGEST =
  "2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd";
const CONTEXT_TOKENS = 8192;
const OLLAMA = "http://127.0.0.1:11434";

function runText(command: string, args: readonly string[], timeout = 15_000): string | null {
  try {
    return execFileSync(command, [...args], {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
      timeout,
      maxBuffer: 2 * 1024 * 1024,
    }).trim();
  } catch {
    return null;
  }
}
function gib(bytes:number){ return Math.round((bytes/1024**3)*100)/100; }
function num(v:string|undefined){ if(v===undefined) return null; const n=Number(v.trim()); return Number.isFinite(n)?n:null; }
function gpuSnapshot(){
  const raw=runText("nvidia-smi",[
    "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,pci.bus_id",
    "--format=csv,noheader,nounits",
  ]);
  const f=raw?.split(/\r?\n/)[0]?.split(",").map(v=>v.trim())??[];
  return {raw,name:f[0]??null,driverVersion:f[1]??null,memoryTotalMiB:num(f[2]),memoryFreeMiB:num(f[3]),memoryUsedMiB:num(f[4]),pciBusId:f[5]??null};
}
function ollamaPs(){
  const raw=runText("ollama",["ps"]);
  const lines=raw?.split(/\r?\n/).map(x=>x.trim()).filter(Boolean)??[];
  return {reachable:raw!==null,raw,rows:lines.length>1?lines.slice(1):[]};
}
function ramSnapshot(){ return {totalRamGiB:gib(os.totalmem()),freeRamGiB:gib(os.freemem())}; }
async function apiJson(path:string,body:Record<string,unknown>){
  const r=await fetch(`${OLLAMA}${path}`,{method:"POST",headers:{"content-type":"application/json"},body:JSON.stringify(body)});
  const t=await r.text(); if(!r.ok) throw new Error(`HTTP_${r.status}:${t}`);
  return JSON.parse(t) as Record<string,unknown>;
}
async function main(){
  const auth=JSON.parse(await (await import("node:fs/promises")).readFile(
    "calibration/vnext/OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_CONTEXT8192_LOAD_SMOKE_AUTH_001.json","utf8"
  ));
  if(auth.status!=="AUTHORIZED_SINGLE_LOAD_ONLY_UNCONSUMED" ||
     auth.authority.load_smoke_authorized!==true ||
     auth.authority.inference_authorized!==false ||
     auth.authority.authorized_run_count!==1){
    throw new Error("QWEN3_5_4B_CONTEXT8192_LOAD_SMOKE_NOT_AUTHORIZED");
  }

  const tags=await fetch(`${OLLAMA}/api/tags`);
  if(!tags.ok) throw new Error(`TAGS_HTTP_${tags.status}`);
  const data=await tags.json() as {models?:Array<{name?:string;digest?:string}>};
  const model=data.models?.find(x=>x.name===MODEL);
  if(!model || model.digest!==EXPECTED_DIGEST) throw new Error("QWEN3_5_4B_IDENTITY_MISMATCH");

  const before={ram:ramSnapshot(),gpu:gpuSnapshot(),ollamaPs:ollamaPs()};
  const load=await apiJson("/api/generate",{model:MODEL,stream:false,keep_alive:"2m",options:{num_ctx:CONTEXT_TOKENS}});
  const loadOnlyConfirmed=load.response==="" && load.done===true && (load.eval_count===undefined || load.eval_count===0);
  await new Promise(r=>setTimeout(r,1000));
  const loaded={ram:ramSnapshot(),gpu:gpuSnapshot(),ollamaPs:ollamaPs()};
  const unload=await apiJson("/api/generate",{model:MODEL,stream:false,keep_alive:0});
  await new Promise(r=>setTimeout(r,1000));
  const after={ram:ramSnapshot(),gpu:gpuSnapshot(),ollamaPs:ollamaPs()};

  console.log(JSON.stringify({
    format:"OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_CONTEXT8192_LOAD_SMOKE_V0.1",
    status:loadOnlyConfirmed?"PASS_LOAD_ONLY_MEASURED":"FAIL_LOAD_ONLY_GUARD",
    mode:"LOCAL_MODEL_LOAD_WITHOUT_INFERENCE",
    target:{model:MODEL,digest:EXPECTED_DIGEST,contextTokens:CONTEXT_TOKENS,promptProvided:false,inferenceRequested:false},
    before,
    load:{response:load.response??null,done:load.done??null,doneReason:load.done_reason??null,evalCount:load.eval_count??null,loadOnlyConfirmed},
    loaded,
    unload:{response:unload.response??null,done:unload.done??null,doneReason:unload.done_reason??null},
    after,
    safety:{networkScope:"LOOPBACK_ONLY",modelDownloadExecuted:false,promptProvided:false,semanticInferenceExecuted:false,productionMutation:false,publicationAuthority:false}
  },null,2));
  if(!loadOnlyConfirmed) process.exitCode=1;
}
main().catch(e=>{
  console.error(JSON.stringify({format:"OROTITAN_GATE18_PHASE_C_QWEN3_5_4B_CONTEXT8192_LOAD_SMOKE_V0.1",status:"BLOCKED",error:e instanceof Error?e.message:String(e),safety:{semanticInferenceExecuted:false}},null,2));
  process.exitCode=1;
});
