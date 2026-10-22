import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod/v4";
import { isAllowedMcpContentLength } from "@/lib/orotitan-equity/v3/mcp-content-length-v1";

import {
  authorizeMcpBearer, mcpEnabled, mcpWriteEnabled, McpBridgeError,
} from "@/lib/orotitan-equity/v3/mcp-bridge-access-v1";
import {
  mcpOAuthConfig, verifyMcpOAuthBearer, oauthChallenge,
  MCP_SCOPE_WRITE, type McpOAuthIdentity,
} from "@/lib/orotitan-equity/v3/mcp-oauth-v1";
import {
  getMcpRunContext, getMcpRunContextDelegated, getMcpNextPrompt,
  getMcpNextPromptDelegated, getMcpRecoveryWorklistDelegated,
  listMcpSubmissions, submitMcpWorkProduct,
} from "@/lib/orotitan-equity/v3/mcp-bridge-service-v1";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const NAME = "orotitan-equity-research";
const VERSION = "0.1.0";

function content(payload: unknown) {
  return {content:[{type:"text" as const,text:JSON.stringify(payload)}]};
}

function failure(error: unknown) {
  // Never echo provider errors, SQL text, request payloads or secrets to ChatGPT.
  let code="OROTITAN_BRIDGE_UNAVAILABLE";
  if(error instanceof McpBridgeError) code=error.code;
  else if(error && typeof error==="object" && "code" in error &&
    typeof (error as {code?:unknown}).code==="string" &&
    /^[A-Z][A-Z0-9_]{1,120}$/.test((error as {code:string}).code)) {
    code=(error as {code:string}).code;
  }
  return {isError:true as const,content:[{type:"text" as const,
    text:JSON.stringify({ok:false,code,admitted:false})}]};
}

function createServer(delegatedJwt: string | null): McpServer {
  const server=new McpServer({name:NAME,version:VERSION},{capabilities:{tools:{}}});
  const runSchema=z.object({
    run_id:z.string().uuid().describe("Exact OroTitan Registry run UUID"),
  }).strict();

  server.registerTool("get_orotitan_run",{
    title:"Consulter un dossier OroTitan",
    description:"Lire le run, l'étape et les blocages depuis Supabase. Observation non validée, jamais une preuve autoritative.",
    inputSchema:runSchema,
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:false},
  },async({run_id})=>{
    try{return content(delegatedJwt
      ? await getMcpRunContextDelegated(run_id,delegatedJwt,process.env)
      : await getMcpRunContext(run_id,process.env));}
    catch(err){return failure(err);}
  });

  server.registerTool("get_orotitan_next_prompt",{
    title:"Obtenir le prompt vérifié de l'étape",
    description:"Pilotage en lecture seule. OAuth : état et blocages Registry non vérifiés, aucun prompt d'exécution. Un prompt exact exige une vérification indépendante des preuves immuables et des contrats.",
    inputSchema:runSchema,
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:false},
  },async({run_id})=>{
    try{
      return content(delegatedJwt
        ? await getMcpNextPromptDelegated(run_id,delegatedJwt,process.env)
        : await getMcpNextPrompt(run_id,process.env));
    }
    catch(err){return failure(err);}
  });

  server.registerTool("get_orotitan_blocker_worklist",{
    title:"Préparer la reprise d'un dossier bloqué",
    description:"Retourne les tâches de recherche et critères de clôture pour chaque blocage Registry observé. Lecture déléguée OAuth/RLS seulement : aucun dépôt, aucune validation de preuve, aucun passage d'étape.",
    inputSchema:runSchema,
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:false},
  },async({run_id})=>{
    try {
      // No privileged fallback. Even private test-bearer mode cannot use this
      // delegated-only operation without a verified user JWT.
      if(!delegatedJwt) throw new McpBridgeError("RECOVERY_OAUTH_REQUIRED");
      return content(await getMcpRecoveryWorklistDelegated(run_id,delegatedJwt,process.env));
    } catch(err) {return failure(err);}
  });

  server.registerTool("list_orotitan_submissions",{
    title:"Consulter les fichiers transmis à OroTitan",
    description:"Lister seulement les livrables en quarantaine privée, reçus mais non vérifiés. Aucune preuve d'admission ou de phase terminée.",
    inputSchema:runSchema,
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:false},
  },async({run_id})=>{
    try{
      if(delegatedJwt)throw new McpBridgeError("DELEGATED_QUARANTINE_READ_DISABLED");
      return content(await listMcpSubmissions(run_id,process.env));
    }
    catch(err){return failure(err);}
  });

  // Separate opt-in, never advertised or permitted by default.
  if(mcpWriteEnabled(process.env)) {
    server.registerTool("submit_orotitan_work_product",{
      title:"Transmettre un livrable textuel non vérifié à OroTitan",
      description:"Déposer un texte, JSON objet, Markdown ou CSV dans la quarantaine privée du run. Aucune admission, aucun Registry write et aucune validation de phase. Action d'écriture demandant confirmation.",
      inputSchema:z.object({
        run_id:z.string().uuid(),
        expected_stage:z.enum(["RESEARCH","DEEP_DIVE","INTEGRATION"]),
        expected_run_state_version:z.number().int().positive(),
        filename:z.string().min(1).max(160),
        content:z.string().min(1).max(100_000),
      }).strict(),
      annotations:{readOnlyHint:false,destructiveHint:false,idempotentHint:true,openWorldHint:false},
    },async({run_id,expected_stage,expected_run_state_version,filename,content:text})=>{
      try{return content(await submitMcpWorkProduct({
        runId:run_id,expectedStage:expected_stage,
        expectedRunStateVersion:expected_run_state_version,
        filename,content:text,
      },process.env));}
      catch(err){return failure(err);}
    });
  }

  return server;
}

// Stateless MCP handlers are constructed per HTTP request. The verified JWT
// is captured in request-local closures; no process-global bearer/context.
function handlerForRequest(request:Request,identity:McpOAuthIdentity|null) {
  const bearer=identity?request.headers.get("authorization")?.slice(7)??null:null;
  return createMcpHandler(()=>createServer(bearer),{legacy:"stateless"});
}

function plain(status:number,message:string,headers:Record<string,string>={}) {
  return new Response(message,{status,headers:{"cache-control":"no-store",...headers}});
}

type GateResult = { denied: Response | null; identity: McpOAuthIdentity | null };

async function gate(request:Request):Promise<GateResult> {
  function deny(response:Response):GateResult {return {denied:response,identity:null};}
  if(!mcpEnabled(process.env)) return deny(plain(404,"Not found"));
  const site=process.env.OROTITAN_MCP_PUBLIC_ORIGIN;
  let url:URL;
  try{url=new URL(request.url);}catch{return deny(plain(400,"Invalid URL"));}
  if(!site || !/^https:\/\/[^/]+$/.test(site) || url.origin!==site) {
    return deny(plain(503,"MCP endpoint not configured"));
  }
  const origin=request.headers.get("origin");
  if(origin && origin!==site)return deny(plain(403,"Origin rejected"));

  if(process.env.OROTITAN_MCP_AUTH_MODE==="OAUTH") {
    const config=mcpOAuthConfig(process.env);
    if(!config)return deny(plain(503,"OAuth not configured"));
    try {
      const identity=await verifyMcpOAuthBearer(
        request.headers.get("authorization"),config);
      return {denied:null,identity};
    } catch {
      return deny(plain(401,"Authentication required",{
        "www-authenticate":oauthChallenge(config),
      }));
    }
  }
  // Static bearer can only be exercised on a private development server.
  // Never expose this shortcut in a production deployment.
  if(process.env.OROTITAN_MCP_AUTH_MODE==="TEST_BEARER" &&
    process.env.NODE_ENV!=="production" &&
    authorizeMcpBearer(request.headers.get("authorization"),process.env)) {
    return {denied:null,identity:null};
  }
  return deny(plain(401,"OAuth required"));
}

export async function POST(request:Request):Promise<Response> {
  const {denied,identity}=await gate(request);
  if(denied)return denied;
  if(!isAllowedMcpContentLength(request.headers.get("content-length"))) {
    return plain(413,"Request too large");
  }
  // Scope-aware tool dispatch: the write tool is NEVER callable with only
  // orotitan.read, even if the caller bypasses the ChatGPT consent UI.
  if(mcpWriteEnabled(process.env)) {
    let parsed:unknown;
    try{parsed=await request.clone().json();}
    catch{return plain(400,"Invalid JSON-RPC body");}
    // Batch requests could smuggle a write past a single-call scope check.
    if (!parsed || typeof parsed!=="object" || Array.isArray(parsed)) {
      return plain(400,"Single JSON-RPC request required");
    }
    if(parsed && typeof parsed==="object" && !Array.isArray(parsed)) {
      const call=parsed as {method?:unknown;params?:unknown};
      const params=call.params && typeof call.params==="object" ?
        call.params as {name?:unknown} : null;
      if(call.method==="tools/call" && params?.name==="submit_orotitan_work_product" &&
        (!identity || !identity.scopes.has(MCP_SCOPE_WRITE))) {
        const config=mcpOAuthConfig(process.env);
        return plain(403,"Insufficient scope",config ?
          {"www-authenticate":oauthChallenge(config)} : {});
      }
    }
  }
  return handlerForRequest(request,identity).fetch(request);
}
export async function GET(request:Request):Promise<Response> {
  const {denied,identity}=await gate(request);
  return denied ?? handlerForRequest(request,identity).fetch(request);
}
export async function DELETE(request:Request):Promise<Response> {
  const {denied,identity}=await gate(request);
  return denied ?? handlerForRequest(request,identity).fetch(request);
}
