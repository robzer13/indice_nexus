/**
 * OroTitan - issuer-agnostic, pre-admission PDF numeric triage.
 * Pure deterministic processing. NO evidence validation, storage, Registry,
 * issuer-authentication or historical-byte certification.
 */
export type NumericProbe = Readonly<{
 claimId:string; numeric:string; derived?:boolean;
 contextHints?:readonly string[];
}>;
export type NumericCandidate = Readonly<{
 claimId:string;excerpt:string;match:'NUMERIC_TOKEN_ONLY';
}>;

export function findNumericCandidates(
 rawPageText:string,probes:readonly NumericProbe[],limit=12,
):NumericCandidate[]{
 if(typeof rawPageText!=='string'||rawPageText.length>1_000_000||
    !Number.isSafeInteger(limit)||limit<1||limit>100)
  throw Error('RESEARCH_NUMERIC_INPUT_DENIED');
 const clean=rawPageText.replace(/\s+/g,' ');
 const normalized=clean.replace(/(\d)[, ](?=\d{3}(?:\D|$))/g,'$1');
 const seen=new Set<string>();
 const matches:NumericCandidate[]=[];
 for(const probe of probes){
  if(!probe||typeof probe.claimId!=='string'||
     !/^[A-Z0-9_-]{2,64}$/.test(probe.claimId)||
     typeof probe.numeric!=='string'||
     !/^-?(?:0|[1-9]\d*)(?:\.\d{1,8})?$/.test(probe.numeric))
   throw Error('RESEARCH_NUMERIC_PROBE_DENIED');
  if(seen.has(probe.claimId))throw Error('RESEARCH_NUMERIC_DUPLICATE_CLAIM');
  seen.add(probe.claimId);
  if(probe.derived===true)continue;
  const escaped=probe.numeric.replace(/[.*+?^$\{\}()|[\]\\]/g,'\\$&');
  const occurrence=new RegExp('(?<!\\d)'+escaped+'(?!\\d)').exec(normalized);
  if(!occurrence)continue;
  const excerpt=normalized.slice(
   Math.max(0,occurrence.index-95),
   Math.min(normalized.length,occurrence.index+probe.numeric.length+150),
  );
  matches.push({claimId:probe.claimId,excerpt:excerpt.slice(0,270),
   match:'NUMERIC_TOKEN_ONLY'});
  if(matches.length>=limit)break;
 }
 return matches;
}

/** Context hints are signals only. A negative hint NEVER disproves a claim. */
export function inspectNumericContext(
 candidate:NumericCandidate,alternatives:readonly string[],
):'HINT_PRESENT'|'HINT_ABSENT'|'HINT_NOT_CONFIGURED'{
 if(!alternatives.length)return 'HINT_NOT_CONFIGURED';
 return alternatives.some(x=>
  x.trim().length>0&&candidate.excerpt.toLocaleLowerCase('en-US')
   .includes(x.toLocaleLowerCase('en-US')))?'HINT_PRESENT':'HINT_ABSENT';
}
