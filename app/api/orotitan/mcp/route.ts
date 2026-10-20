import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod/v4";

import {
  authorizeMcpBearer, mcpEnabled, mcpWriteEnabled, McpBridgeError,
} from "@/lib/orotitan-equity/v3/mcp-bridge-access-v1";
import {
  getMcpRunContext, getMcpNextPrompt, listMcpSubmissions, submitMcpWorkProduct,
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

function createServer(): McpServer {
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
    try{return content(await getMcpRunContext(run_id,process.env));}
    catch(err){return failure(err);}
  });

  server.registerTool("get_orotitan_next_prompt",{
    title:"Obtenir le prompt vérifié de l'étape",
    description:"Vérifier les preuves immuables GitHub, le contrat et l'état Registry. Renvoie le prompt exact uniquement si le handoff est admis; ne lance aucune IA, transition ni publication.",
    inputSchema:runSchema,
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:false},
  },async({run_id})=>{
    try{return content(await getMcpNextPrompt(run_id,process.env));}
    catch(err){return failure(err);}
  });

  server.registerTool("list_orotitan_submissions",{
    title:"Consulter les fichiers transmis à OroTitan",
    description:"Lister seulement les livrables en quarantaine privée, reçus mais non vérifiés. Aucune preuve d'admission ou de phase terminée.",
    inputSchema:runSchema,
    annotations:{readOnlyHint:true,destructiveHint:false,openWorldHint:false},
  },async({run_id})=>{
    try{return content(await listMcpSubmissions(run_id,process.env));}
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

// Official SDK handles Streamable HTTP: 2026 protocol plus stateless legacy.
const handler=createMcpHandler(createServer,{legacy:"stateless"});

function plain(status:number,message:string,headers:Record<string,string>={}) {
  return new Response(message,{status,headers:{"cache-control":"no-store",...headers}});
}

function gate(request:Request):Response|null {
  if(!mcpEnabled(process.env)) return plain(404,"Not found");
  const site=process.env.OROTITAN_MCP_PUBLIC_ORIGIN;
  let url:URL;
  try{url=new URL(request.url);}catch{return plain(400,"Invalid URL");}
  // Explicit HTTPS origin, no preview/wildcard/attacker-selected Host trust.
  if(!site || !/^https:\/\/[^/]+$/.test(site) || url.origin!==site) {
    return plain(503,"MCP endpoint not configured");
  }
  const origin=request.headers.get("origin");
  if(origin && origin!==site)return plain(403,"Origin rejected");
  if(!authorizeMcpBearer(request.headers.get("authorization"),process.env)) {
    return plain(401,"Authentication required",{"www-authenticate":'Bearer realm="orotitan-mcp"'});
  }
  return null;
}

export async function POST(request:Request):Promise<Response> {
  const denied=gate(request);
  if(denied)return denied;
  return handler(request);
}
export async function GET(request:Request):Promise<Response> {
  const denied=gate(request);
  return denied ?? handler(request);
}
export async function DELETE(request:Request):Promise<Response> {
  const denied=gate(request);
  return denied ?? handler(request);
}
