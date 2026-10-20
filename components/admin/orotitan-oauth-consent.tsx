"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { isAllowedChatGptRedirect, isAllowedChatGptReturnUrl } from "@/lib/orotitan-equity/v3/mcp-consent-policy-v1";

type Details = {
  client?:{id?:string;name?:string};
  redirect_uri?:string;
  scope?:string;
  authorization_id?:string;
  redirect_url?:string;
};

type Props = {
  authorizationId: string;
  supabaseUrl: string;
  publishableKey: string;
  siteOrigin: string;
};

/** OAuth credentials are separate from OroTitan's admin-cookie authentication. */
export function OroTitanOauthConsent({
  authorizationId,supabaseUrl,publishableKey,siteOrigin,
}:Props) {
  const supabase=useMemo(()=>createClient(supabaseUrl,publishableKey,{
    auth:{persistSession:true,autoRefreshToken:true,detectSessionInUrl:true},
  }),[supabaseUrl,publishableKey]);
  const [email,setEmail]=useState("");
  const [password,setPassword]=useState("");
  const [identity,setIdentity]=useState<string|null>(null);
  const [details,setDetails]=useState<Details|null>(null);
  const [error,setError]=useState("");
  const [busy,setBusy]=useState(true);
  const [acting,setActing]=useState(false);

  const refresh=useCallback(async()=>{
    setError("");
    setBusy(true);
    try{
      if(window.location.origin!==siteOrigin)throw new Error("ORIGIN_MISMATCH");
      const {data:{user},error:userError}=await supabase.auth.getUser();
      if(userError || !user){
        setIdentity(null);setDetails(null);return;
      }
      // Browser session object alone is never an admission decision. Server
      // independently calls Supabase Auth getUser(access_token) and allowlists it.
      const {data:{session}}=await supabase.auth.getSession();
      if(!session?.access_token)throw new Error("SESSION_MISSING");
      const response=await fetch("/api/orotitan/oauth/eligibility",{
        method:"POST",headers:{Authorization:"Bearer "+session.access_token},
        cache:"no-store",credentials:"same-origin",
      });
      if(!response.ok)throw new Error("ACCOUNT_NOT_ADMITTED");
      const {data,error:detailError}=await supabase.auth.oauth.getAuthorizationDetails(authorizationId);
      if(detailError||!data)throw new Error("AUTHORIZATION_NOT_FOUND");
      const next=data as Details;
      if("redirect_url" in next && !("authorization_id" in next)){
        if(!isAllowedChatGptReturnUrl(next.redirect_url)){
          throw new Error("REDIRECT_NOT_CHATGPT");
        }
        window.location.assign(next.redirect_url!);
        return;
      }
      if(!isAllowedChatGptRedirect(next.redirect_uri)){
        throw new Error("REDIRECT_NOT_CHATGPT");
      }
      if(!next.client?.name)throw new Error("CLIENT_IDENTITY_MISSING");
      setIdentity(user.email??user.id);
      setDetails(next);
    }catch{
      setDetails(null);
      setIdentity(null);
      setError("Demande ou identité non autorisée. Contactez l'administrateur OroTitan.");
    }finally{setBusy(false);}
  },[supabase,authorizationId,siteOrigin]);

  useEffect(()=>{void refresh();},[refresh]);

  async function login(event:React.FormEvent<HTMLFormElement>){
    event.preventDefault();setActing(true);setError("");
    try{
      const {error:loginError}=await supabase.auth.signInWithPassword({email,password});
      setPassword("");
      if(loginError)throw loginError;
      await refresh();
    }catch{
      setPassword("");
      setError("Connexion refusée. Vérifiez vos identifiants Supabase Auth.");
    }finally{setActing(false);}
  }

  async function decide(decision:"approve"|"deny"){
    if(!details||acting||busy)return;
    setActing(true);setError("");
    try{
      // Fail closed on changed authorization/client/redirect state.
      const {data:latest,error:checkError}=
        await supabase.auth.oauth.getAuthorizationDetails(authorizationId);
      if(checkError||!latest||!("authorization_id" in latest)||
        !isAllowedChatGptRedirect(latest.redirect_uri)||
        latest.redirect_uri!==details.redirect_uri||
        latest.client?.id!==details.client?.id){
        throw new Error("CONSENT_CONTEXT_CHANGED");
      }
      const result=decision==="approve"?
        await supabase.auth.oauth.approveAuthorization(authorizationId):
        await supabase.auth.oauth.denyAuthorization(authorizationId);
      if(result.error || !result.data?.redirect_url ||
        !isAllowedChatGptReturnUrl(result.data.redirect_url)){
        throw new Error("CONSENT_REDIRECT_REJECTED");
      }
      window.location.assign(result.data.redirect_url);
    }catch{
      setError("La demande d'autorisation a échoué. Aucune autorisation n'a été confirmée.");
      setActing(false);
    }
  }

  return <main className="mx-auto max-w-xl p-5 sm:p-8">
    <section className="rounded-2xl border border-slate-700 bg-slate-900 p-6 text-slate-100 shadow-xl">
      <p className="text-xs font-semibold uppercase tracking-widest text-cyan-300">
        OroTitan Equity Research · Connexion sécurisée
      </p>
      <h1 className="mt-3 text-2xl font-semibold">Autorisation d'accès ChatGPT</h1>
      <p className="mt-3 text-sm leading-6 text-slate-300">
        Cette autorisation permet uniquement de consulter les dossiers OroTitan
        préalablement autorisés. Elle ne lance aucune analyse et ne peut ni
        valider une étape ni publier un résultat.
      </p>
      {busy?<p className="mt-5 text-sm" role="status">Vérification de l'identité…</p>:null}
      {error?<p role="alert" className="mt-5 rounded-lg border border-rose-900 bg-rose-950/40 p-3 text-sm text-rose-200">{error}</p>:null}
      {!busy&&!identity?<form className="mt-6 space-y-4" onSubmit={event=>void login(event)}>
        <p className="text-sm text-slate-300">
          Connectez-vous avec un compte Supabase Auth créé et autorisé pour OroTitan.
          Le mot de passe de l'administration OroTitan ne fonctionne pas ici.
        </p>
        <label className="block text-sm">Adresse e-mail du compte autorisé
          <input required type="email" autoComplete="username" value={email}
            onChange={e=>setEmail(e.target.value)} className="mt-2 block w-full rounded-lg border border-slate-600 bg-slate-950 p-3"/>
        </label>
        <label className="block text-sm">Mot de passe Supabase Auth
          <input required type="password" autoComplete="current-password" value={password}
            onChange={e=>setPassword(e.target.value)} className="mt-2 block w-full rounded-lg border border-slate-600 bg-slate-950 p-3"/>
        </label>
        <button type="submit" disabled={acting}
          className="min-h-11 w-full rounded-lg bg-cyan-400 px-4 py-3 font-semibold text-slate-950 disabled:opacity-40">
          {acting?"Connexion…":"Se connecter pour examiner les droits"}
        </button>
      </form>:null}
      {!busy&&identity&&details?<div className="mt-6 space-y-4">
        <div className="rounded-xl border border-slate-700 bg-slate-950/60 p-4 text-sm">
          <p className="font-semibold">Application : {details.client?.name}</p>
          <p className="mt-2 text-slate-400">Compte : {identity}</p>
          <p className="mt-2 break-all text-xs text-slate-400">Retour : {details.redirect_uri}</p>
        </div>
        <div>
          <h2 className="text-sm font-semibold">Permissions demandées</h2>
          <ul className="mt-2 list-inside list-disc text-sm text-slate-300">
            {(details.scope??"").split(" ").filter(Boolean).map(scope=><li key={scope}>{scope}</li>)}
          </ul>
          <p className="mt-2 text-xs text-amber-200">
            Les droits effectifs restent limités côté serveur à la lecture des runs
            explicitement autorisés. Aucun droit d'écriture n'est activé.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button type="button" disabled={acting} onClick={()=>void decide("deny")}
            className="min-h-11 flex-1 rounded-lg border border-slate-600 px-4 py-3 text-sm disabled:opacity-40">
            Refuser
          </button>
          <button type="button" disabled={acting} onClick={()=>void decide("approve")}
            className="min-h-11 flex-1 rounded-lg bg-cyan-400 px-4 py-3 text-sm font-semibold text-slate-950 disabled:opacity-40">
            {acting?"Traitement…":"Autoriser la lecture"}
          </button>
        </div>
      </div>:null}
    </section>
  </main>;
}
