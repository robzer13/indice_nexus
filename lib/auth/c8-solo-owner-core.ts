import type {User} from '@supabase/supabase-js';
import {
 SHADOW_RESEARCH_ORIGIN,SHADOW_RESEARCH_PROJECT_REF,
 type HostedShadowEnv,
} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';

const UUID=/^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const AMENDMENT=/^https:\/\/github\.com\/robzer13\/indice_nexus\/blob\/[a-f0-9]{40}\/docs\/audits\/[\w.-]+\.md$/;
export type SoloOwnerConfig=Readonly<{url:string;publishableKey:string;
 ownerUserId:string;amendmentRef:string;amendmentSha256:string}>;
export type SoloOwnerAttestation=Readonly<{authProvider:'SUPABASE_AUTH';
 ownerUserId:string;governanceMode:'SOLO_OWNER';amendmentRef:string;
 amendmentSha256:string}>;

function legacyJwtRole(key:string):string|null{
 const parts=key.split('.');if(parts.length!==3)return null;
 try{return String((JSON.parse(Buffer.from(parts[1],'base64url').toString()) as {role?:unknown}).role??'');}
 catch{return 'INVALID';}
}
export function getSoloOwnerConfig(env:HostedShadowEnv):SoloOwnerConfig{
 const key=env.OROTITAN_C8_OWNER_AUTH_PUBLISHABLE_KEY?.trim()??'';
 const legacyRole=legacyJwtRole(key);
 const ownerUserId=env.OROTITAN_C8_SOLO_OWNER_USER_ID?.trim()??'';
 const amendmentRef=env.OROTITAN_C8_SOLO_AMENDMENT_REF?.trim()??'';
 const amendmentSha256=env.OROTITAN_C8_SOLO_AMENDMENT_SHA256?.trim()??'';
 if(env.OROTITAN_C8_OWNER_AUTH_ENABLED!=='1'||env.OROTITAN_C8_SOLO_AMENDMENT_RATIFIED!=='1'||
    env.OROTITAN_EXPECTED_SUPABASE_PROJECT_REF!==SHADOW_RESEARCH_PROJECT_REF||
    env.OROTITAN_C8_OWNER_AUTH_URL?.replace(/\/$/,'')!==SHADOW_RESEARCH_ORIGIN||
    !UUID.test(ownerUserId)||!AMENDMENT.test(amendmentRef)||
    !/^[a-f0-9]{64}$/.test(amendmentSha256)||key.length<20||
    key.startsWith('sb_secret_')||(legacyRole!==null&&legacyRole!=='anon'))
  throw Error('OWNER_AUTH_CONFIGURATION_DENIED');
 return {url:SHADOW_RESEARCH_ORIGIN,publishableKey:key,ownerUserId,
  amendmentRef,amendmentSha256};
}
export function assertSoloOwnerUser(user:Pick<User,'id'|'email_confirmed_at'|'is_anonymous'>,
 config:SoloOwnerConfig):SoloOwnerAttestation{
 if(user.id!==config.ownerUserId||!user.email_confirmed_at||user.is_anonymous===true)
  throw Error('OWNER_ATTESTATION_DENIED');
 return {authProvider:'SUPABASE_AUTH',ownerUserId:user.id,governanceMode:'SOLO_OWNER',
  amendmentRef:config.amendmentRef,amendmentSha256:config.amendmentSha256};
}
