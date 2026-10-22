import http from "node:http";
import { fileURLToPath } from "node:url";
import { decodeSignablePayload,loadSignerPolicy,SignerPolicyError } from "./policy.mjs";
import { signWithCloudKms } from "./kms.mjs";

async function collectJson(req,maxBytes) {
  const type=req.headers["content-type"]??"";
  if(typeof type!=="string" || !/^application\/json(?:\s*;\s*charset=utf-8)?$/i.test(type))
    throw new SignerPolicyError("UNSUPPORTED_MEDIA_TYPE");
  const chunks=[];let total=0;
  for await(const x of req){
    total+=x.length;
    if(total>maxBytes)throw new SignerPolicyError("REQUEST_TOO_LARGE");
    chunks.push(x);
  }
  try{return JSON.parse(Buffer.concat(chunks).toString("utf8"));}
  catch{throw new SignerPolicyError("MALFORMED_REQUEST_JSON");}
}
function send(res,status,obj) {
  const body=JSON.stringify(obj);
  res.writeHead(status,{"Content-Type":"application/json; charset=utf-8",
    "Cache-Control":"no-store","X-Content-Type-Options":"nosniff"});
  res.end(body);
}

/** IAM authorization MUST be required on the Cloud Run service itself.
 * No user-supplied HTTP header or OAuth access token grants signing authority.
 */
export function createSignerServer(policy,signMessage){
  const server=http.createServer(async(req,res)=>{
    if(req.url==="/healthz" && req.method==="GET"){
      res.writeHead(204,{"Cache-Control":"no-store"});res.end();return;
    }
    if(req.url!=="/v1/sign-d2-handoff" || req.method!=="POST"){
      send(res,404,{error:"not_found"});return;
    }
    try{
      const input=await collectJson(req,26000);
      const message=decodeSignablePayload(input,policy,Math.floor(Date.now()/1000));
      const sig=await signMessage(message,policy);
      if(!Buffer.isBuffer(sig) || sig.length!==64)
        throw new Error("INVALID_SIGNER_OUTPUT");
      send(res,200,{
        version:1,keyId:policy.keyId,
        payloadBase64Url:input.payloadBase64Url,
        signatureBase64Url:sig.toString("base64url"),
      });
    }catch(err){
      if(err instanceof SignerPolicyError){
        // Never echo private prompt, request body, token or key material.
        send(res,403,{error:"request_denied"});
      }else{
        send(res,503,{error:"signing_unavailable"});
      }
    }
  });
  server.requestTimeout=15000;
  server.headersTimeout=6000;
  server.keepAliveTimeout=3000;
  server.maxRequestsPerSocket=10;
  return server;
}
if(process.argv[1] && fileURLToPath(import.meta.url)===process.argv[1]){
  const policy=loadSignerPolicy(process.env); // fail before listening
  const port=Number(process.env.PORT??"8080");
  if(!Number.isSafeInteger(port) || port<1 || port>65535)
    throw new Error("INVALID_CLOUD_RUN_PORT");
  const server=createSignerServer(policy,signWithCloudKms);
  server.listen(port,"0.0.0.0",()=>console.info("D29b private signer listening"));
}
