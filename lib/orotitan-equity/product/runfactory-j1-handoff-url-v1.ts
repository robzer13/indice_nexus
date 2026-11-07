/** Extract the single canonical Run ID from a Next.js request URL.
 *
 * Vercel/ChatGPT/browser links may append non-authoritative query metadata.
 * Only the one required runId controls Registry lookup. Extra params must
 * never influence the identity, and duplicate runId values are rejected.
 */
const UUID=/^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;

export function parseJ1HandoffRunId(url:string):string|null {
 try{
  const params=new URL(url).searchParams;
  const ids=params.getAll('runId');
  if(ids.length!==1||!UUID.test(ids[0]))return null;
  return ids[0].toLowerCase();
 }catch{return null;}
}
