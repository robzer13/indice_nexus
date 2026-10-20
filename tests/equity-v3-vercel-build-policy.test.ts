import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

type VercelConfig = {
  crons?: Array<{path:string;schedule:string}>;
  git?: {deploymentEnabled?: Record<string,boolean>|boolean};
};

test("Vercel automatic build suppression targets only the long-lived Research branch",()=>{
  const config=JSON.parse(readFileSync(new URL("../vercel.json",import.meta.url),"utf8")) as VercelConfig;
  assert.deepEqual(config.git?.deploymentEnabled,{
    "fix/orotitan-v31-research-handoff-artifact-contract-20261019":false,
  });
  assert.notEqual(config.git?.deploymentEnabled,false,
    "Production deployments must not be globally disabled");
  assert.deepEqual(config.crons,[{
    path:"/api/cron/prices",
    schedule:"15 22 * * 1-5",
  }],"Preserve the existing production price refresh schedule");
});
