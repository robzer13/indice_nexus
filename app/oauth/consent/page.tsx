import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { OroTitanOauthConsent } from "@/components/admin/orotitan-oauth-consent";
import {
  isAuthorizationId, oauthConsentConfig,
} from "@/lib/orotitan-equity/v3/mcp-consent-policy-v1";
import { assertProductionUrl } from "@/lib/orotitan-equity/v3/pilotage-trusted-loader-server";

export const metadata:Metadata={title:"Consentement ChatGPT — OroTitan"};
export const dynamic="force-dynamic";

type Props={searchParams:Promise<{authorization_id?:string|string[]}>};

export default async function OauthConsentPage({searchParams}:Props){
  const cfg=oauthConsentConfig(process.env);
  // Never expose a live OAuth consent UI before the issuer, a real Auth user
  // and the separated rollout flags have been reviewed and configured.
  if(!cfg)notFound();
  try{assertProductionUrl();}catch{notFound();}
  const {authorization_id:id}=await searchParams;
  if(!isAuthorizationId(id))notFound();
  return <OroTitanOauthConsent
    authorizationId={id}
    supabaseUrl={cfg.supabaseUrl}
    publishableKey={cfg.publishableKey}
    siteOrigin={cfg.siteOrigin}
    allowedClientIds={cfg.allowedClientIds}
  />;
}
