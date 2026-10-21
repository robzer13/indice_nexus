import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { isIsolatedMcpPilot, isPilotPublicPath, PILOT_ORIGIN } from "../lib/orotitan-equity/v3/mcp-pilot-boundary-v1";

const admitted:NodeJS.ProcessEnv={
  OROTITAN_MCP_PUBLIC_ORIGIN:PILOT_ORIGIN,
  OROTITAN_OAUTH_SITE_ORIGIN:PILOT_ORIGIN,
  OROTITAN_MCP_AUTH_MODE:"OAUTH",
};

test("only explicitly configured isolated pilot uses public gateway view",()=>{
  assert.equal(isIsolatedMcpPilot(admitted),true);
  for(const env of [
    {},{...admitted,OROTITAN_MCP_PUBLIC_ORIGIN:"https://orotitan-screener.vercel.app"},
    {...admitted,OROTITAN_OAUTH_SITE_ORIGIN:"https://orotitan-screener.vercel.app"},
    {...admitted,OROTITAN_MCP_AUTH_MODE:"TEST_BEARER"},
  ])assert.equal(isIsolatedMcpPilot(env),false);
});

test("pilot route isolation permits only OAuth/MCP endpoints and static Next assets",()=>{
  for(const path of ["/","/oauth/consent","/api/orotitan/oauth/eligibility","/api/orotitan/mcp",
    "/.well-known/oauth-protected-resource","/_next/static/chunks/app.js","/_next/image","/favicon.ico"]){
    if(path==="/_next/image")assert.equal(isPilotPublicPath(path),false);
    else assert.equal(isPilotPublicPath(path),true,path);
  }
  for(const path of ["/admin","/admin/orotitan","/screener","/company/besi",
    "/api/admin","/api/orotitan/pilotage","/api/orotitan/mcp/extra",
    "/oauth/consent/other","/.well-known/oauth-protected-resource-extra",
    "/_next/data","/api/orotitan/oauth","//oauth/consent"]){
    assert.equal(isPilotPublicPath(path),false,path);
  }
});

test("root dashboard branches before privileged getCompanyStates, keeping main code",()=>{
  const source=readFileSync(new URL("../app/page.tsx",import.meta.url),"utf8");
  assert.match(source,/if \(isIsolatedMcpPilot\(process\.env\)\)/);
  assert.ok(source.indexOf("if (isIsolatedMcpPilot(process.env))") <
    source.indexOf("const companies = await getCompanyStates();"));
  assert.match(source,/Aucune modification, validation ou publication/);
  const layout=readFileSync(new URL("../app/layout.tsx",import.meta.url),"utf8");
  assert.match(layout,/if \(isIsolatedMcpPilot\(process\.env\)\)/);
  const proxy=readFileSync(new URL("../proxy.ts",import.meta.url),"utf8");
  assert.match(proxy,/isPilotPublicPath\(request\.nextUrl\.pathname\)/);
  assert.match(proxy,/status: 404/);
});
