import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { delegatedPilotageStatus, type DelegatedPilotageObservation } from
  "../lib/orotitan-equity/v3/mcp-delegated-pilotage-v1";

const base:DelegatedPilotageObservation={
  runId:"48b9426b-4d0d-4858-9d87-efcd13ac415d",
  issuerName:"Besi",ticker:"BESI",runStatus:"ACTIVE",registryStage:"RESEARCH",
  stageLifecycle:"IN_PROGRESS",contractStatusCode:"BLOCKED_INSUFFICIENT_INPUT",
  handoffGateName:"READY_FOR_DEEP_DIVE",handoffGateState:"NO",
  activeManifestKind:"CHECKPOINT",processVersion:"3.1",pilotageVersion:"3.1",
  runStateVersion:6,stageStateVersion:4,
  blockers:[{code:"G003",text:"Commercial qualification evidence missing",resolution:null}],
};

test("D29a: observed Research blockers are useful status, never a prompt",()=>{
  const result=delegatedPilotageStatus(base);
  assert.equal(result.status,"BLOCKED_OBSERVED");
  assert.equal(result.runId,base.runId);
  assert.deepEqual(result.blockers,base.blockers);
  assert.equal(result.runStateVersion,6);
  assert.equal(result.stageStateVersion,4);
  assert.equal(result.trust,"REGISTRY_OBSERVATION_NOT_ARTIFACT_VERIFICATION");
  assert.equal(result.prompt,null);
  assert.equal(result.verifiedPrompt,null);
  assert.equal(result.stateFingerprintSha256,null);
  assert.equal(result.verificationPerformed,false);
  assert.equal(result.nextStepAuthorized,false);
  assert.equal(result.publicationAuthorized,false);
  assert.equal(result.mutationPerformed,false);
  assert.match(result.guidance,/Do not advance stage/);
});

test("D29a: even an apparently final YES cannot turn Registry observation into evidence",()=>{
  const result=delegatedPilotageStatus({...base,stageLifecycle:"COMPLETE",
    contractStatusCode:"READY",handoffGateState:"YES",activeManifestKind:"FINAL",
    blockers:[]});
  assert.equal(result.status,"UNVERIFIED_NO_PROMPT");
  assert.equal(result.prompt,null);
  assert.equal(result.verifiedPrompt,null);
  assert.equal(result.nextStepAuthorized,false);
  assert.equal(result.publicationAuthorized,false);
  assert.equal(result.verificationPerformed,false);
});

test("D29a: unsupported versions, unexecutable runs and missing stage fail closed",()=>{
  const cases=[
    [{...base,processVersion:"2.0"},"UNSUPPORTED_VERSION"],
    [{...base,pilotageVersion:"2.0"},"UNSUPPORTED_VERSION"],
    [{...base,runStatus:"PUBLISHED"},"RUN_NOT_EXECUTABLE_UPSTREAM"],
    [{...base,stageStateVersion:null},"STAGE_OBSERVATION_MISSING"],
    [{...base,registryStage:null},"STAGE_OBSERVATION_MISSING"],
    [{...base,stageLifecycle:"UNKNOWN"},"STAGE_OBSERVATION_MISSING"],
  ] as const;
  for(const [input,status] of cases){
    const r=delegatedPilotageStatus(input);
    assert.equal(r.status,status);
    assert.equal(r.prompt,null);
    assert.equal(r.nextStepAuthorized,false);
  }
});

test("D29a: OAuth never calls the privileged immutable-proof verifier",()=>{
  const route=readFileSync(new URL("../app/api/orotitan/mcp/route.ts",import.meta.url),"utf8");
  const service=readFileSync(new URL("../lib/orotitan-equity/v3/mcp-bridge-service-v1.ts",import.meta.url),"utf8");
  const helper=readFileSync(new URL("../lib/orotitan-equity/v3/mcp-delegated-pilotage-v1.ts",import.meta.url),"utf8");
  assert.match(route,/getMcpNextPromptDelegated\(run_id,delegatedJwt,process\.env\)/);
  assert.match(route,/getMcpNextPrompt\(run_id,process\.env\)/);
  assert.match(service,/getMcpRunContextDelegated\(runIdRaw,jwt,env\)/);
  const delegated=service.split("export async function getMcpNextPromptDelegated(")[1]
    .split("/**\n * Reuses D2 trusted immutable bytes verification")[0];
  assert.doesNotMatch(delegated,/loadTrustedPilotageDecision|createServerSupabaseClient|observeSupabaseEnvironment|getMcpNextPrompt\(/);
  assert.doesNotMatch(helper,/VERIFIED_FROM_IMMUTABLE_BYTES|prompt:\s*["\x27\x60]|service_role|GO PUBLISH/);
  assert.match(route,/if\(delegatedJwt\)throw new McpBridgeError\("DELEGATED_QUARANTINE_READ_DISABLED"\)/);
});
