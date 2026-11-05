import 'server-only';
import {cookies} from 'next/headers';
import {createClient} from '@supabase/supabase-js';
import type {HostedShadowEnv} from '@/lib/orotitan-equity/product/hosted-shadow-research-gate';
import {assertSoloOwnerUser,getSoloOwnerConfig,type SoloOwnerAttestation} from './c8-solo-owner-core';
export {assertSoloOwnerUser,getSoloOwnerConfig} from './c8-solo-owner-core';
export type {SoloOwnerAttestation,SoloOwnerConfig} from './c8-solo-owner-core';

export const C8_OWNER_COOKIE='orotitan_c8_owner_access_token';
export function createSoloOwnerAuthClient(env:HostedShadowEnv=process.env){
 const config=getSoloOwnerConfig(env);
 return {config,client:createClient(config.url,config.publishableKey,{
  auth:{persistSession:false,autoRefreshToken:false,detectSessionInUrl:false},
 })};
}

export async function requireSoloOwnerAttestation(
 env:HostedShadowEnv=process.env,
):Promise<SoloOwnerAttestation>{
 const token=(await cookies()).get(C8_OWNER_COOKIE)?.value;
 if(!token||token.length>4096)throw Error('OWNER_SESSION_REQUIRED');
 const {config,client}=createSoloOwnerAuthClient(env);
 const {data,error}=await client.auth.getUser(token);
 if(error||!data.user)throw Error('OWNER_SESSION_INVALID');
 return assertSoloOwnerUser(data.user,config);
}
