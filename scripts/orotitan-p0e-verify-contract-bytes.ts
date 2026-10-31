/**
 * Manual P0-E immutable-byte audit. No database access, no Registry write.
 * Usage: NODE_OPTIONS='--conditions=react-server' npx tsx
 *        scripts/orotitan-p0e-verify-contract-bytes.ts
 */
import {verifyAllRunFactoryPinnedBytes} from
 '../lib/orotitan-equity/product/runfactory-p0e-byte-gate';

async function main(){
 const report=await verifyAllRunFactoryPinnedBytes('OROTITAN_V3_1');
 const verifiedAt=new Date().toISOString();
 const proof={...report,checkedAt:verifiedAt};
 console.log(JSON.stringify(proof,null,2));
}
void main().catch(error=>{
 console.error(error instanceof Error?error.message:String(error));
 process.exitCode=1;
});
