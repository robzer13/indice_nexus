/**
 * OroTitan P0-D — deterministic audited SHADOW identity onboarding command.
 * Does NOT authenticate remote sources: this is explicitly a manual
 * operator attestation with two source URLs and a reviewed-at timestamp.
 * Only server code under exact hosted-SHADOW and admin session may call RPC.
 */
import {createHash} from 'node:crypto';
import {assertIsin} from './runfactory-identity-preflight-v1';

export type P0DInput=Readonly<{
 isin:string;ticker:string;exchange:string;
 displayName:string;legalName:string;issuerCountry:string;
 reportingCurrency:string;tradingCurrency:string;
 primaryListing:boolean;quoteUnit:'MAJOR'|'MINOR';priceDecimals:number;
 legacyCompanyId:string|null;existingIssuerId:string|null;
 issuerSourceUrl:string;listingSourceUrl:string;
 reviewedAt:string;confirmedIdentity:boolean;confirmation:string;
}>;
export type P0DRpcArgs=Readonly<{
 p_idempotency_key:string;p_isin:string;p_ticker:string;p_exchange:string;
 p_display_name:string;p_legal_name:string;p_issuer_country:string;
 p_reporting_currency:string;p_trading_currency:string;
 p_primary_listing:boolean;p_quote_unit:'MAJOR'|'MINOR';p_price_decimals:number;
 p_legacy_company_id:string|null;p_existing_issuer_id:string|null;
 p_evidence:Readonly<{
 issuer_source_url:string;listing_source_url:string;
 method:'OPERATOR_PRIMARY_SOURCE_REVIEW';
 reviewed_by:'ADMIN_SESSION_MANUAL_ATTESTATION_V1';observed_at:string;
 }>;
 p_operator_attestation:'APPROVE_SHADOW_CANONICAL_IDENTITY_V1';
}>;
const invalid=(code:string):never=>{throw Error(code);};
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
function id(v:unknown):string|null{
 if(v===null)return null;
 if(typeof v!=='string'||!UUID.test(v))invalid('P0D_ID_INVALID');
 return v.toLowerCase();
}
function txt(v:unknown,max:number):string{
 if(typeof v!=='string'||!v.trim()||v.length>max||/[\u0000-\u001F\u007F]/.test(v))
  invalid('P0D_TEXT_INVALID');
 return v.trim();
}
function sourceUrl(v:unknown):string{
 if(typeof v!=='string'||v.length>2048)invalid('P0D_SOURCE_INVALID');
 let url:URL;
 try{url=new URL(v);}catch{return invalid('P0D_SOURCE_INVALID');}
 if(url.protocol!=='https:'||url.username||url.password||url.port||
    !/^[a-z0-9-]+(?:\.[a-z0-9-]+)+$/i.test(url.hostname)||
    url.hostname.endsWith('.local')||
    /^\d+(?:\.\d+){3}$/.test(url.hostname)||
    !url.pathname.startsWith('/')||url.pathname==='/'||
    url.hash)
   invalid('P0D_SOURCE_INVALID');
 return url.toString();
}
export function buildP0DOnboardingCommand(input:P0DInput,now=new Date()):
 P0DRpcArgs{
 if(!input||input.confirmedIdentity!==true)invalid('P0D_REVIEW_REQUIRED');
 const isin=assertIsin(input.isin);
 if(input.confirmation!=='ONBOARD '+isin)invalid('P0D_CONFIRMATION_INVALID');
 const ticker=txt(input.ticker,30).toUpperCase(),
  exchange=txt(input.exchange,30).toUpperCase();
 if(!/^[A-Z0-9._-]{1,30}$/.test(ticker)||
    !/^[A-Z0-9._-]{2,30}$/.test(exchange))
  invalid('P0D_LISTING_INVALID');
 const displayName=txt(input.displayName,180),
  legalName=txt(input.legalName,180);
 const issuerCountry=txt(input.issuerCountry,2).toUpperCase(),
  reportingCurrency=txt(input.reportingCurrency,3).toUpperCase(),
  tradingCurrency=txt(input.tradingCurrency,3).toUpperCase();
 if(!/^[A-Z]{2}$/.test(issuerCountry)||
    !/^[A-Z]{3}$/.test(reportingCurrency)||
    !/^[A-Z]{3}$/.test(tradingCurrency)||
    (input.quoteUnit!=='MAJOR'&&input.quoteUnit!=='MINOR')||
    !Number.isInteger(input.priceDecimals)||input.priceDecimals<0||
    input.priceDecimals>6||typeof input.primaryListing!=='boolean')
  invalid('P0D_METADATA_INVALID');
 const issuerSourceUrl=sourceUrl(input.issuerSourceUrl);
 const listingSourceUrl=sourceUrl(input.listingSourceUrl);
 if(new URL(issuerSourceUrl).hostname===new URL(listingSourceUrl).hostname)
  invalid('P0D_SOURCES_NOT_INDEPENDENT');
 if(typeof input.reviewedAt!=='string'||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(input.reviewedAt))
  invalid('P0D_REVIEW_TIMESTAMP_INVALID');
 const reviewed=Date.parse(input.reviewedAt);
 if(!Number.isFinite(reviewed)||reviewed>now.getTime()+60_000||
    reviewed<now.getTime()-7*86400_000)
  invalid('P0D_REVIEW_TIMESTAMP_INVALID');
 const p_evidence={
  issuer_source_url:issuerSourceUrl,listing_source_url:listingSourceUrl,
  method:'OPERATOR_PRIMARY_SOURCE_REVIEW' as const,
  reviewed_by:'ADMIN_SESSION_MANUAL_ATTESTATION_V1' as const,
  observed_at:new Date(reviewed).toISOString(),
 };
 const payload={
  p_isin:isin,p_ticker:ticker,p_exchange:exchange,
  p_display_name:displayName,p_legal_name:legalName,
  p_issuer_country:issuerCountry,
  p_reporting_currency:reportingCurrency,p_trading_currency:tradingCurrency,
  p_primary_listing:input.primaryListing,p_quote_unit:input.quoteUnit,
  p_price_decimals:input.priceDecimals,
  p_legacy_company_id:id(input.legacyCompanyId),
  p_existing_issuer_id:id(input.existingIssuerId),
  p_evidence,
 };
 const digest=createHash('sha256').update(
  'OROTITAN_P0D_IDENTITY_COMMAND_V1\n'+JSON.stringify(payload)
 ).digest('hex');
 return {
  p_idempotency_key:'orotitan:p0d:shadow:'+digest,
  ...payload,
  p_operator_attestation:'APPROVE_SHADOW_CANONICAL_IDENTITY_V1',
 };
}
