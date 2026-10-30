/**
 * Synthetic PDF response for the disposable PG17 / loopback browser test ONLY.
 * Never call this from a hosted Preview or Production environment.
 * The bytes below are NOT a primary source or valid investment evidence.
 */
import {ASML_Q2_2026_OFFICIAL_PRESENTATION,ASML_Q2_2026_PUBLIC_RELEASE}
  from './research-recovery-official-source-v1';

export const isRecoveryCiBrowserFixture = (env:NodeJS.ProcessEnv):boolean=>
  env.CI==='true' &&
  env.NODE_ENV==='development' &&
  !env.VERCEL_ENV &&
  env.OROTITAN_PRODUCT_BROWSER_FIXTURE==='1' &&
  env.OROTITAN_RECOVERY_E2E_FIXTURE==='1' &&
  env.OROTITAN_SHARED_INBOX_TEST_ENABLED==='1' &&
  env.OROTITAN_SHARED_INBOX_TEST_PROJECT_REF==='local-browser-fixture' &&
  env.OROTITAN_SHARED_INBOX_TEST_URL==='http://127.0.0.1:48761' &&
  env.NEXT_PUBLIC_SUPABASE_URL==='http://127.0.0.1:48761' &&
  typeof env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY==='string' &&
  env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY.length>=40 &&
  env.SUPABASE_SERVICE_ROLE_KEY===env.OROTITAN_SHARED_INBOX_TEST_SERVICE_KEY;

export const recoverySyntheticFetch:typeof fetch = async (input,options)=>{
  if(!isRecoveryCiBrowserFixture(process.env))throw Error('RECOVERY_FIXTURE_DENIED');
  if(options?.redirect!=='error')throw Error('RECOVERY_FIXTURE_REDIRECT_DENIED');
  const target=String(input);
  if(target===ASML_Q2_2026_PUBLIC_RELEASE.url){
    const response=new Response('CI SYNTHETIC REJECTED HTML',{status:403,
      headers:{'content-type':'text/html'}});
    Object.defineProperty(response,'url',{value:target});
    return response;
  }
  if(target===ASML_Q2_2026_OFFICIAL_PRESENTATION.url){
    const bytes=Buffer.concat([Buffer.from('%PDF-1.7\n'),
      Buffer.from('OroTitan CI synthetic fixture NO SOURCE EVIDENCE\n'.repeat(400)),
      Buffer.from('%%EOF\n')]);
    const response=new Response(new Uint8Array(bytes),{status:200,
      headers:{'content-type':'application/pdf'}});
    Object.defineProperty(response,'url',{value:target});
    return response;
  }
  throw Error('RECOVERY_FIXTURE_URL_DENIED');
};
