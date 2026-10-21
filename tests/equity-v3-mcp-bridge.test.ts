import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { isAllowedMcpContentLength } from "../lib/orotitan-equity/v3/mcp-content-length-v1";
import {
  allowedMcpRuns, authorizeMcpBearer, mcpEnabled, mcpWriteEnabled,
  requireMcpRun, McpBridgeError,
} from "../lib/orotitan-equity/v3/mcp-bridge-access-v1";

const besi="48b9426b-4d0d-4858-9d87-efcd13ac415d";
const other="11111111-1111-4111-8111-111111111111";
const token="s".repeat(64);
const good:NodeJS.ProcessEnv={
  NODE_ENV:"test",
  OROTITAN_MCP_ENABLED:"true",
  OROTITAN_MCP_ALLOWED_RUN_IDS:besi,
  OROTITAN_MCP_READ_TOKEN:token,
  OROTITAN_MCP_WRITE_ENABLED:"false",
  OROTITAN_MANUAL_INTAKE_ENABLED:"false",
};

test("MCP is disabled by default and denies requests without a valid bearer",()=>{
  assert.equal(mcpEnabled({NODE_ENV:"test"}),false);
  assert.equal(authorizeMcpBearer("Bearer "+token,{NODE_ENV:"test"}),false);
  assert.equal(authorizeMcpBearer(null,good),false);
  assert.equal(authorizeMcpBearer("Bearer "+token.slice(1),good),false);
  assert.equal(authorizeMcpBearer("Basic "+token,good),false);
  assert.equal(authorizeMcpBearer("Bearer "+token,{...good,OROTITAN_MCP_READ_TOKEN:"weak"}),false);
  assert.equal(authorizeMcpBearer("Bearer "+token,{...good,OROTITAN_MCP_ALLOWED_RUN_IDS:""}),false);
  assert.equal(authorizeMcpBearer("Bearer "+token,good),true);
  assert.equal(authorizeMcpBearer("Bearer "+token,{...good,OROTITAN_MCP_ENABLED:"false"}),false);
});

test("MCP allows only explicit exact run IDs and no wildcard or invalid UUID",()=>{
  assert.deepEqual([...allowedMcpRuns(good)],[besi]);
  assert.equal(requireMcpRun(besi,good),besi);
  assert.throws(()=>requireMcpRun(other,good),(e:unknown)=>e instanceof McpBridgeError&&e.code==="RUN_NOT_ALLOWED");
  assert.throws(()=>requireMcpRun("../"+besi,good),(e:unknown)=>e instanceof McpBridgeError);
  assert.equal(allowedMcpRuns({...good,OROTITAN_MCP_ALLOWED_RUN_IDS:"*"}).size,0);
  assert.equal(allowedMcpRuns({...good,OROTITAN_MCP_ALLOWED_RUN_IDS:besi+","+besi}).size,0);
});

test("MCP write tools are disabled for the Supabase OAuth pilot regardless of flags",()=>{
  assert.equal(mcpWriteEnabled(good),false);
  assert.equal(mcpWriteEnabled({...good,OROTITAN_MCP_WRITE_ENABLED:"true"}),false);
  assert.equal(mcpWriteEnabled({...good,OROTITAN_MANUAL_INTAKE_ENABLED:"true"}),false);
  assert.equal(mcpWriteEnabled({...good,OROTITAN_MCP_ENABLED:"false",
    OROTITAN_MCP_WRITE_ENABLED:"true",OROTITAN_MANUAL_INTAKE_ENABLED:"true"}),false);
  assert.equal(mcpWriteEnabled({...good,OROTITAN_MCP_WRITE_ENABLED:"true",
    OROTITAN_MANUAL_INTAKE_ENABLED:"true"}),false);
});

test("MCP POST Content-Length accepts numeric payload sizes and rejects malformed or oversized headers",()=>{
  assert.equal(isAllowedMcpContentLength(null),true); // chunked or streaming body
  for(const value of ["0","1","250","4096","250000","000250"]) {
    assert.equal(isAllowedMcpContentLength(value),true,`expected ${value} to pass`);
  }
  for(const value of ["","-1","+1","1.5"," 250","250 ","2e3","250001",
    "999999999999999999999999999999999","\\\\d","NaN"]) {
    assert.equal(isAllowedMcpContentLength(value),false,`expected ${value} to fail`);
  }
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  assert.match(route,/isAllowedMcpContentLength\(request\.headers\.get\("content-length"\)\)/);
});

test("Public OAuth read uses only caller JWT and publishable key, never service_role",()=>{
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  const service=readFileSync(new URL("../lib/orotitan-equity/v3/mcp-bridge-service-v1.ts",import.meta.url),"utf8");
  assert.match(route,/createMcpHandler\(\(\)=>createServer\(bearer,identity\)/);
  assert.match(route,/handlerForRequest\(request,identity\)/);
  assert.match(route,/getMcpRunContextDelegated\(run_id,delegatedJwt,process\.env\)/);
  assert.match(route,/getMcpNextPromptDelegated\(run_id,delegatedJwt,process\.env,identity\)/);
  assert.match(route,/if\(delegatedJwt\)throw new McpBridgeError\("DELEGATED_QUARANTINE_READ_DISABLED"\)/);
  assert.match(service,/global:\{headers:\{Authorization:"Bearer "\+jwt\}\}/);
  assert.match(service,/SUPABASE_PUBLISHABLE_KEY/);
  assert.match(service,/requireMcpRun\(runIdRaw,env\)/);
  const delegated=service.split("function delegatedClient(")[1].split("async function getRun(")[0];
  assert.doesNotMatch(delegated,/SUPABASE_SERVICE_ROLE_KEY|createServerSupabaseClient/);
});

test("MCP route uses official Streamable HTTP SDK and guards the full endpoint",()=>{
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  const service=readFileSync(new URL("../lib/orotitan-equity/v3/mcp-bridge-service-v1.ts",import.meta.url),"utf8");
  assert.match(route,/createMcpHandler/);
  assert.match(route,/new McpServer/);
  for(const tool of ["get_orotitan_run","get_orotitan_next_prompt",
    "list_orotitan_submissions","submit_orotitan_work_product"]){
    assert.ok(route.includes(tool));
  }
  assert.match(route,/authorizeMcpBearer/);
  assert.match(route,/OROTITAN_MCP_PUBLIC_ORIGIN/);
  assert.match(route,/if\(mcpWriteEnabled\(process\.env\)\)/);
  assert.match(service,/loadTrustedPilotageDecision/);
  assert.match(service,/observeSupabaseEnvironment/);
  assert.match(service,/requireMcpRun/);
  assert.match(service,/intakeObjectPath/);
  assert.match(service,/upsert:false/);
  assert.doesNotMatch(service,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
  assert.doesNotMatch(route,/\.rpc\s*\(|\.insert\s*\(|\.update\s*\(|\.delete\s*\(/);
  assert.doesNotMatch(service,/GO PUBLISH|start_orotitan_stage|finalize_orotitan_stage/);
});
