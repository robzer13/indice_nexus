import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import test from "node:test";

const get=(file:string)=>readFileSync(new URL("../mcp-public-gateway/"+file,import.meta.url),"utf8");

test("isolated public gateway reveals availability but no Registry data",()=>{
  const home=get("app/page.tsx");
  const health=get("app/health/route.ts");
  const mcp=get("app/api/orotitan/mcp/route.ts");
  const metadata=get("app/.well-known/oauth-protected-resource/route.ts");
  assert.match(health,/mcpEnabled: false/);
  assert.match(health,/registryAccessible: false/);
  assert.match(health,/writesEnabled: false/);
  assert.match(mcp,/status: 503/);
  assert.match(metadata,/status: 404/);
  assert.match(home,/Accès aux dossiers désactivé/);
});

test("public MCP gateway is independent from original app auth and production secrets",()=>{
  const pkg=JSON.parse(get("package.json")) as {scripts:Record<string,string>;dependencies:Record<string,string>};
  assert.equal(pkg.dependencies.next,"16.3.3");
  assert.equal(pkg.scripts.build,"next build");
  const content=["app/page.tsx","app/health/route.ts","app/api/orotitan/mcp/route.ts",
    "app/.well-known/oauth-protected-resource/route.ts","app/layout.tsx"].map(get).join("\n");
  for(const forbidden of ["SUPABASE_SERVICE_ROLE_KEY","OROTITAN_ADMIN_PASSWORD",
    "createServerSupabaseClient","orotitan_runs","publish_orotitan",
    "approveAuthorization","submit_orotitan_work_product"]){
    assert.equal(content.includes(forbidden),false,forbidden);
  }
});
