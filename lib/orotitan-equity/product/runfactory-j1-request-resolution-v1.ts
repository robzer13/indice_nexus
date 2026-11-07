/** J1: resolve a human's company request against observed canonical SHADOW records.
 * Never infer an issuer or primary exchange from ambiguous matches.
 * This function is pure and MUST NOT create an identity or authorize a Run.
 */
export type J1CompanyIdentity=Readonly<{
  isin:string;exchange:string;ticker:string;
  securityId:string;issuerId:string;issuerName:string;
}>;
export type J1Resolution=
  | Readonly<{status:'RESOLVED';identity:J1CompanyIdentity}>
  | Readonly<{status:'AMBIGUOUS'|'NOT_FOUND';identity:null}>;

export function resolveJ1CompanyIdentity(
  query:string,observed:readonly J1CompanyIdentity[],
):J1Resolution{
  const normalized=query.trim().normalize('NFKC').toLocaleUpperCase('en-US');
  const isinQuery=/^[A-Z]{2}[A-Z0-9]{9}[0-9]$/.test(normalized);
  const matches=observed.filter(x=>{
    const name=x.issuerName.trim().normalize('NFKC').toLocaleUpperCase('en-US');
    return (isinQuery?x.isin.toUpperCase()===normalized:
      x.ticker.toUpperCase()===normalized||
      name===normalized||name.startsWith(normalized+' '));
  });
  // Duplicate/multi-listing identities are intentionally not auto-selected.
  if(matches.length===0)return {status:'NOT_FOUND',identity:null};
  if(matches.length!==1)return {status:'AMBIGUOUS',identity:null};
  return {status:'RESOLVED',identity:matches[0]};
}
