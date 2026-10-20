import assert from "node:assert/strict";
import test from "node:test";
import {
  createLocalJWKSet, exportJWK, generateKeyPair, SignJWT,
} from "jose";
import {
  mcpOAuthConfig, verifyMcpOAuthBearer, oauthResourceMetadata,
  oauthChallenge, MCP_SCOPE_READ, MCP_SCOPE_WRITE, McpOAuthError,
} from "../lib/orotitan-equity/v3/mcp-oauth-v1";

const base: NodeJS.ProcessEnv = {
  NODE_ENV:"test",
  OROTITAN_MCP_ENABLED:"true",
  OROTITAN_MCP_AUTH_MODE:"OAUTH",
  OROTITAN_MCP_PUBLIC_ORIGIN:"https://orotitan.example.org",
  OROTITAN_MCP_OAUTH_ISSUER:"https://id.example.org/oauth",
  OROTITAN_MCP_OAUTH_JWKS_URI:"https://id.example.org/oauth/jwks",
  OROTITAN_MCP_OAUTH_ALLOWED_SUBJECTS:"user-123",
};
const config=mcpOAuthConfig(base)!;

test("MCP OAuth resource metadata and config fail closed",()=>{
  assert.equal(config.issuer,"https://id.example.org/oauth");
  assert.equal(config.audience,"https://orotitan.example.org/api/orotitan/mcp");
  assert.equal(config.resourceMetadataUrl,"https://orotitan.example.org/.well-known/oauth-protected-resource");
  assert.deepEqual(oauthResourceMetadata(config),{
    resource:config.audience,
    authorization_servers:[config.issuer],
    scopes_supported:[MCP_SCOPE_READ,MCP_SCOPE_WRITE],
  });
  assert.match(oauthChallenge(config),/resource_metadata="https:/);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_ENABLED:"false"}),null);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_AUTH_MODE:"TEST_BEARER"}),null);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_OAUTH_ISSUER:"http://id.example.org"}),null);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_OAUTH_JWKS_URI:"https://attacker.example.com/jwks"}),null);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_OAUTH_ALLOWED_SUBJECTS:"*"}),null);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_OAUTH_ALLOWED_SUBJECTS:"user-123,user-123"}),null);
  assert.equal(mcpOAuthConfig({...base,OROTITAN_MCP_PUBLIC_ORIGIN:"https://orotitan.example.org/login"}),null);
});

test("MCP OAuth independently verifies RS256 signature, issuer, audience, subject and scope",async()=>{
  const {publicKey,privateKey}=await generateKeyPair("RS256",{extractable:true});
  const jwk=await exportJWK(publicKey);
  const keys=createLocalJWKSet({keys:[{...jwk,kid:"testkey",alg:"RS256",use:"sig"}]});
  const issue=async(input?:{
    issuer?:string;audience?:string;subject?:string;scope?:string;exp?:string;
  })=>{
    let builder=new SignJWT({scope:input?.scope??MCP_SCOPE_READ})
      .setProtectedHeader({alg:"RS256",kid:"testkey"})
      .setIssuer(input?.issuer??config.issuer)
      .setAudience(input?.audience??config.audience)
      .setSubject(input?.subject??"user-123")
      .setIssuedAt();
    builder=builder.setExpirationTime(input?.exp??"10m");
    return builder.sign(privateKey);
  };
  const good=await issue();
  const identity=await verifyMcpOAuthBearer("Bearer "+good,config,keys);
  assert.equal(identity.subject,"user-123");
  assert.equal(identity.scopes.has(MCP_SCOPE_READ),true);
  assert.equal(identity.scopes.has(MCP_SCOPE_WRITE),false);
  const fails=async(value:string)=>{
    await assert.rejects(()=>verifyMcpOAuthBearer("Bearer "+value,config,keys),
      (error:unknown)=>error instanceof McpOAuthError);
  };
  await fails(await issue({audience:"https://other.example.org/api/orotitan/mcp"}));
  await fails(await issue({issuer:"https://another.example.org"}));
  await fails(await issue({subject:"user-999"}));
  await fails(await issue({scope:MCP_SCOPE_WRITE}));
  await fails(await issue({exp:"-30m"}));
  await fails(good.slice(0,-5)+"BADxx");
  await assert.rejects(()=>verifyMcpOAuthBearer(null,config,keys),
    (error:unknown)=>error instanceof McpOAuthError);
});

test("MCP OAuth discovery route and dispatch require validated metadata and scope",()=>{
  const {readFileSync}=require("node:fs") as typeof import("node:fs");
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  const meta=readFileSync(new URL("../app/.well-known/oauth-protected-resource/route.ts",import.meta.url),"utf8");
  assert.match(route,/verifyMcpOAuthBearer/);
  assert.match(route,/oauthChallenge/);
  assert.match(route,/securitySchemes:\[\{type:"oauth2",scopes:\[MCP_SCOPE_READ\]\}\]/);
  assert.match(route,/params\?\.name==="submit_orotitan_work_product"/);
  assert.match(route,/identity\.scopes\.has\(MCP_SCOPE_WRITE\)/);
  assert.match(route,/process\.env\.NODE_ENV!=="production"/);
  assert.match(meta,/oauthResourceMetadata/);
});
