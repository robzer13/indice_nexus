'use client';

import {useState} from 'react';
import type {P0CDecision} from
 '@/lib/orotitan-equity/product/runfactory-canonical-onboarding-candidate-v1';
import type {P0DInput} from
 '@/lib/orotitan-equity/product/runfactory-p0d-onboarding-command-v1';
import type {P0DReviewDraft} from
 '@/lib/orotitan-equity/product/runfactory-onboarding-review-draft-v1';
import {buildUntrustedRunPreflightHandoff} from
 '@/lib/orotitan-equity/product/runfactory-run-preflight-handoff-v1';

type InspectResponse={ok:true;result:P0CDecision};
type CommitResponse={ok:true;environment:'SHADOW_ONLY';outcome:string;
 issuerId:string;securityId:string;dossierId:string;
 persistedAndReadBack:true;operatorAttestationRecorded:true;
 officialSourceBytesAutomaticallyAuthenticated:false;
 runCreated:false;stageAdvanced:false;productionTouched:false;
};
type FormState={
 isin:string;ticker:string;exchange:string;displayName:string;legalName:string;
 issuerCountry:string;reportingCurrency:string;tradingCurrency:string;
 primaryListing:boolean;quoteUnit:'MAJOR'|'MINOR';priceDecimals:number;
 issuerSourceUrl:string;listingSourceUrl:string;
};
const EMPTY:FormState={
 isin:'',ticker:'',exchange:'',displayName:'',legalName:'',
 issuerCountry:'',reportingCurrency:'',tradingCurrency:'',
 primaryListing:true,quoteUnit:'MAJOR',priceDecimals:2,
 issuerSourceUrl:'',listingSourceUrl:'',
};
export function AtomicOnboardingClient({
 initialValues,
}:{initialValues:P0DReviewDraft|null}){
 const [values,setValues]=useState<FormState>(initialValues??EMPTY);
 const [checked,setChecked]=useState(false);
 const [reviewedAt,setReviewedAt]=useState('');
 const [confirmation,setConfirmation]=useState('');
 const [reuseIssuer,setReuseIssuer]=useState(false);
 const [candidate,setCandidate]=useState<P0CDecision|null>(null);
 const [commitResult,setCommitResult]=useState<CommitResponse|null>(null);
 const [error,setError]=useState('');
 const [busy,setBusy]=useState<'inspect'|'commit'|null>(null);
 function edit<K extends keyof FormState>(key:K,value:FormState[K]){
  setValues(prev=>({...prev,[key]:value}));
  setChecked(false);setReviewedAt('');setConfirmation('');
  setCommitResult(null);setError('');
  if(['isin','ticker','exchange','displayName'].includes(key))
   setCandidate(null);
 }
 async function inspect(){
  if(busy||!values.isin||!values.ticker||!values.exchange)return;
  setBusy('inspect');setCandidate(null);setCommitResult(null);setError('');
  try{
   const response=await fetch('/admin/research-factory/onboarding/inspect',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({isin:values.isin,ticker:values.ticker,
     exchange:values.exchange,issuerDisplayName:values.displayName||null,
     evidenceUrl:values.issuerSourceUrl||null}),
   });
   const data:unknown=await response.json();
   if(!response.ok||!data||typeof data!=='object'||!('ok' in data)||data.ok!==true)
    throw Error(data&&typeof data==='object'&&'error' in data?
     String(data.error):'Préflight indisponible.');
   const proposed=(data as InspectResponse).result;
   if(proposed.registryMutated!==false||proposed.operationAuthorized!==false)
    throw Error('Préflight incohérent — aucune création autorisée.');
   setCandidate(proposed);
  }catch(e){setError(e instanceof Error?e.message:'Erreur de préflight.');}
  finally{setBusy(null);}
 }
 const blocked=candidate?.status.startsWith('BLOCKED')??true;
 async function commit(){
  if(busy||!candidate||blocked||!checked||!reviewedAt)return;
  setBusy('commit');setError('');setCommitResult(null);
  try{
   const request:P0DInput={
    isin:values.isin,ticker:values.ticker,exchange:values.exchange,
    displayName:values.displayName,legalName:values.legalName,
    issuerCountry:values.issuerCountry,
    reportingCurrency:values.reportingCurrency,
    tradingCurrency:values.tradingCurrency,
    primaryListing:values.primaryListing,quoteUnit:values.quoteUnit,
    priceDecimals:values.priceDecimals,
    legacyCompanyId:candidate.legacyCompanyId,
    existingIssuerId:(candidate.existingIssuerId&&
     (candidate.status==='EXISTING_SECURITY_REUSE_CANDIDATE'||reuseIssuer))?
     candidate.existingIssuerId:null,
    issuerSourceUrl:values.issuerSourceUrl,
    listingSourceUrl:values.listingSourceUrl,
    reviewedAt,confirmedIdentity:checked,confirmation,
   };
   const response=await fetch('/admin/research-factory/onboarding/commit',{
    method:'POST',credentials:'same-origin',cache:'no-store',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(request),
   });
   const raw:unknown=await response.json();
   if(!response.ok||!raw||typeof raw!=='object'||!('ok' in raw)||raw.ok!==true)
    throw Error(raw&&typeof raw==='object'&&'error' in raw?
     String(raw.error):'Échec de la transaction — aucun résultat confirmé.');
   const result=raw as CommitResponse;
   if(result.environment!=='SHADOW_ONLY'||
      result.persistedAndReadBack!==true||result.runCreated!==false||
      result.stageAdvanced!==false||result.productionTouched!==false||
      result.officialSourceBytesAutomaticallyAuthenticated!==false)
    throw Error('Intégrité du reçu de transaction non confirmée.');
   setCommitResult(result);
  }catch(e){setError(e instanceof Error?e.message:'Erreur de transaction.');}
  finally{setBusy(null);}
 }
 const fields:[
  keyof FormState,string,string
 ][]=[
  ['isin','ISIN (transaction)','FR0000120321'],
  ['ticker','Ticker','OR'],['exchange','MIC marché','XPAR'],
  ['displayName','Nom affiché de l’émetteur','L’Oréal'],
  ['legalName','Dénomination légale (vérifier)','L’Oréal S.A.'],
  ['issuerCountry','Pays ISO2','FR'],
  ['reportingCurrency','Devise de reporting ISO3','EUR'],
  ['tradingCurrency','Devise de transaction ISO3','EUR'],
  ['issuerSourceUrl','Source officielle émetteur','https://www.loreal-finance.com/fr/laction'],
  ['listingSourceUrl','Source officielle bourse','https://live.euronext.com/fr/product/equities/fr0000120321-xpar/overview'],
 ];
 return <div className="space-y-6">
  <section className="rounded-lg border border-slate-700 p-5 space-y-4">
   <h2 className="text-xl font-semibold text-white">1. Identité du titre et sources</h2>
   <p className="text-sm text-slate-300">
    Revue initiale obligatoire une seule fois par émetteur. Les deux liens
    doivent provenir des sites officiels que vous avez personnellement consultés.
    Le système ne télécharge ni n’authentifie leurs octets à cette étape.
   </p>
   <div className="grid gap-3 md:grid-cols-2">
    {fields.map(([key,label,placeholder])=><label key={key}
     className="space-y-1 text-sm text-slate-200">
     <span>{label}</span>
     <input disabled={busy!==null||commitResult!==null}
      value={String(values[key])} placeholder={placeholder}
      onChange={e=>edit(key,e.target.value as never)}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
    </label>)}
   </div>
   <div className="grid gap-3 md:grid-cols-3">
    <label className="space-y-1 text-sm text-slate-200">
     <span>Unité de cotation</span>
     <select value={values.quoteUnit} disabled={busy!==null||commitResult!==null}
      onChange={e=>edit('quoteUnit',e.target.value as 'MAJOR'|'MINOR')}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white">
      <option value="MAJOR">MAJOR</option><option value="MINOR">MINOR</option>
     </select>
    </label>
    <label className="space-y-1 text-sm text-slate-200">
     <span>Décimales prix (0–6)</span>
     <input type="number" min={0} max={6} value={values.priceDecimals}
      disabled={busy!==null||commitResult!==null}
      onChange={e=>edit('priceDecimals',Number(e.target.value))}
      className="w-full rounded-md border border-slate-600 bg-slate-900 p-3 text-white"/>
    </label>
    <label className="flex items-center gap-2 text-sm text-slate-200">
     <input type="checkbox" checked={values.primaryListing}
      disabled={busy!==null||commitResult!==null}
      onChange={e=>edit('primaryListing',e.target.checked)}/>
     Cotation principale
    </label>
   </div>
   <button type="button" onClick={()=>void inspect()} disabled={busy!==null||
    commitResult!==null||!values.isin||!values.ticker||!values.exchange}
    className="rounded-md bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-50">
    {busy==='inspect'?'Contrôle…':'2. Contrôler les doublons dans SHADOW'}
   </button>
  </section>
  {candidate&&<section className="space-y-3 rounded-lg border border-slate-700 p-5">
   <h2 className="text-xl font-semibold text-white">Résultat du préflight</h2>
   <p className="font-mono text-sm text-amber-200">{candidate.status}</p>
   <p className="text-xs text-slate-300">
    ISIN : {candidate.isin} · Titre : {candidate.ticker} · Marché : {candidate.exchange}
   </p>
   <p className="text-xs text-slate-400">
    Émetteur existant : {candidate.existingIssuerId??'NON RÉSOLU'}.
    Dossier existant : {candidate.existingDossierId??'AUCUN'}.
   </p>
   {candidate.existingIssuerId&&
    candidate.status==='EXISTING_ISSUER_NAME_REVIEW'&&
    <label className="flex items-start gap-2 text-sm text-amber-100">
     <input type="checkbox" checked={reuseIssuer}
      onChange={e=>{setReuseIssuer(e.target.checked);setChecked(false);}}
      disabled={busy!==null||commitResult!==null}/>
     Je confirme que cet émetteur existant représente exactement la même
     personne morale après vérification des documents.
    </label>}
   {blocked?<p className="text-rose-300">
    Collision ou incohérence détectée. Transaction désactivée.
   </p>:<p className="text-sm text-slate-300">
    Contrôles préparatoires non bloquants. La transaction vérifiera de nouveau
    toutes les identités sous verrou PostgreSQL, sans faire confiance à cet écran.
   </p>}
  </section>}
  {candidate&&!blocked&&!commitResult&&<section className="space-y-4 rounded-lg border border-amber-700 p-5">
   <h2 className="text-xl font-semibold text-white">3. Autorisation manuelle — SHADOW uniquement</h2>
   <p className="text-sm text-slate-200">
    Vérifiez les sources officielles dans votre navigateur. En validant,
    vous autorisez la création ou la réutilisation des enregistrements
    <code> issuers / securities / research_dossiers </code>
    en base de TEST uniquement. Aucun run Research ne sera créé.
   </p>
   <p className="text-xs text-slate-400">
    La preuve documentaire sera enregistrée comme
    ATTESTATION DE REVUE HUMAINE, pas comme source authentifiée automatiquement.
   </p>
   <div className="space-y-1 text-sm">
    <a className="block break-all text-cyan-300 underline" href={values.issuerSourceUrl}
     target="_blank" rel="noopener noreferrer">
     Consulter la source émetteur
    </a>
    <a className="block break-all text-cyan-300 underline" href={values.listingSourceUrl}
     target="_blank" rel="noopener noreferrer">
     Consulter la source de cotation
    </a>
   </div>
   <label className="flex items-start gap-3 text-sm text-slate-200">
    <input type="checkbox" checked={checked} disabled={busy!==null}
     onChange={e=>{
      setChecked(e.target.checked);
      setReviewedAt(e.target.checked?new Date().toISOString():'');
     }}/>
    J’ai personnellement vérifié la dénomination juridique, l’ISIN,
    le ticker, la place de cotation, les devises et les deux sources officielles.
   </label>
   <label className="block space-y-1 text-sm text-slate-200">
    <span>Confirmer exactement : ONBOARD {candidate.isin}</span>
    <input value={confirmation} disabled={busy!==null}
     onChange={e=>setConfirmation(e.target.value)}
     placeholder={'ONBOARD '+candidate.isin}
     className="w-full rounded-md border border-amber-600 bg-slate-900 p-3 text-white"/>
   </label>
   <button type="button" onClick={()=>void commit()} disabled={busy!==null||
    !checked||confirmation!=='ONBOARD '+candidate.isin||
    (candidate.status==='EXISTING_ISSUER_NAME_REVIEW'&&!reuseIssuer)}
    className="rounded-md bg-amber-500 px-5 py-3 font-semibold text-slate-950 disabled:opacity-50">
    {busy==='commit'?'Transaction et relecture…':'Créer ou réutiliser dans SHADOW'}
   </button>
   {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>}
  {error&&(!candidate||blocked)&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  {commitResult&&<section className="space-y-3 rounded-lg border border-emerald-700 p-5">
   <h2 className="text-xl font-semibold text-white">
    Transaction SHADOW confirmée : {commitResult.outcome}
   </h2>
   <p className="text-sm text-slate-200">Les trois enregistrements ont
    été relus après la transaction, avec un reçu d’idempotence.
    Aucun run, score, stage ou snapshot n’a été créé.</p>
   <div className="space-y-1 break-all font-mono text-xs text-slate-300">
    <p>ISSUER_ID = {commitResult.issuerId}</p>
    <p>SECURITY_ID = {commitResult.securityId}</p>
    <p>DOSSIER_ID = {commitResult.dossierId}</p>
   </div>
   <a href={buildUntrustedRunPreflightHandoff({
     isin:values.isin,exchange:values.exchange,
    })}
    className="text-sm text-cyan-300 underline">
    Continuer vers P0-B : identité transmise, méthodologie à sélectionner
   </a>
  </section>}
 </div>;
}
