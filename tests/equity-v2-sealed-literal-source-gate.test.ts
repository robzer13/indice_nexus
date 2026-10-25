import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import {
  validateFicoStyleExactLiteralsFromSealedSources,
  type CertifiedLiteralGateInput,
  type RegistrySealedSource,
} from "../lib/orotitan-equity/v2/sealed-literal-source-gate";

const runId = "00000000-0000-4000-8000-000000000001";
const dossierId = "00000000-0000-4000-8000-000000000002";
const issuerId = "00000000-0000-4000-8000-000000000003";
const securityId = "00000000-0000-4000-8000-000000000004";
const cutoff = "2026-09-15";

function seal(body: Record<string, unknown>, artifactId: string): RegistrySealedSource {
  const utf8Body=JSON.stringify(body);
  return { utf8Body,registeredSha256:createHash("sha256").update(utf8Body,"utf8").digest("hex"),
    registeredSizeBytes:Buffer.byteLength(utf8Body,"utf8"),
    registeredArtifactId:artifactId,registeredVersion:1 };
}
function fixture(): CertifiedLiteralGateInput {
  const lockId="00000000-0000-4000-8000-000000000005";
  const readyId="00000000-0000-4000-8000-000000000006";
  const f={
    run_id:runId,issuer_id:issuerId,security_id:securityId,dossier_id:dossierId,
    data_cutoff:cutoff,stage:"DEEP_DIVE",version:1,artifact_type:"FUNDAMENTALS_LOCK",
    artifact:{artifact_id:lockId,version:1},canonical_fundamental_verdicts:{RETURN_INTERPRETABILITY:"MEDIUM"},
  };
  const a={
    run_id:runId,data_cutoff:cutoff,stage:"DEEP_DIVE",version:1,artifact_type:"READINESS_NEXT_ACTION_ARTIFACT",
    refresh_class:"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED",
  };
  const candidate={
    issuer_id:issuerId,security_id:securityId,data_lock:{data_cutoff:cutoff},
    l2_research_fundamentals:{analytical_metrics:{return_interpretability:"MEDIUM"}},
    l4_operational_state:{refresh_class:"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED"},
  };
  return {runId,issuerId,securityId,dossierId,dataCutoff:cutoff,
    frozenFundamentalsLock:seal(f,lockId),frozenReadiness:seal(a,readyId),candidate};
}

test("authentic literals proceed to full schema gate rather than false certification",()=>{
  const input=fixture(),before=JSON.stringify(input);
  const x=validateFicoStyleExactLiteralsFromSealedSources(input);
  assert.equal(x.ok,false);
  if(!x.ok)assert.equal(x.stage,"SEMANTIC_SCHEMA"); // synthetic fixture is intentionally not full ANALYZE
  assert.equal(JSON.stringify(input),before);
});
test("altered source SHA or length cannot enter app validator",()=>{
  for(const v of ["hash","size","body"]) {
    const input=fixture();
    const lock={...input.frozenFundamentalsLock};
    if(v==="hash")lock.registeredSha256="1".repeat(64);
    if(v==="size")lock.registeredSizeBytes++;
    if(v==="body")lock.utf8Body+=" ";
    const x=validateFicoStyleExactLiteralsFromSealedSources({...input,frozenFundamentalsLock:lock});
    assert.equal(x.ok,false);
    if(!x.ok){assert.equal(x.stage,"SOURCE_PROVENANCE"); assert.deepEqual(x.errors,["SEALED_SOURCE_HASH_OR_IDENTITY_MISMATCH"]);}
  }
});
test("source-candidate rewrites, wrong cutoff or issuer fail closed",()=>{
  for(const key of ["interpret","refresh","cutoff","issuer"]) {
    const input=fixture();const candidate=structuredClone(input.candidate) as Record<string, unknown>;
    const l2=candidate.l2_research_fundamentals as Record<string, unknown>;
    const metrics=l2.analytical_metrics as Record<string, unknown>;
    const l4=candidate.l4_operational_state as Record<string, unknown>;
    const lock=candidate.data_lock as Record<string, unknown>;
    if(key==="interpret")metrics.return_interpretability="INTERPRETABLE_WITH_LIMITATIONS";
    if(key==="refresh")l4.refresh_class="NOT_APPLICABLE";
    if(key==="cutoff")lock.data_cutoff="2026-01-01";
    if(key==="issuer")candidate.issuer_id="00000000-0000-4000-8000-000000000007";
    const x=validateFicoStyleExactLiteralsFromSealedSources({...input,candidate});
    assert.equal(x.ok,false);
    if(!x.ok)assert.equal(x.stage,"SOURCE_PROVENANCE");
  }
});
test("registered source hash with unsupported literal still fails closed",()=>{
  const input=fixture();const modified=JSON.parse(input.frozenReadiness.utf8Body);
  modified.refresh_class="UNREVIEWED_FUTURE_STATE";
  const fakeButInternallyConsistent=seal(modified,input.frozenReadiness.registeredArtifactId);
  const x=validateFicoStyleExactLiteralsFromSealedSources({...input,frozenReadiness:fakeButInternallyConsistent});
  assert.equal(x.ok,false);if(!x.ok)assert.deepEqual(x.errors,["SOURCE_OUTSIDE_ADMITTED_EXACT_LITERAL_SCOPE"]);
});
