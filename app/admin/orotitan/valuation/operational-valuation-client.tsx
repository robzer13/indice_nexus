"use client";
import Link from "next/link";
import {useState,type ChangeEvent} from "react";
import {projectOperationalValuation,OperationalValuationInputError,SCENARIOS,
  upsideToFairValue,type OperationalView,type ScenarioName} from "@/lib/orotitan-equity/v3/operational-valuation-view";

const euro=(x:number)=>x.toLocaleString("fr-FR",{style:"currency",currency:"EUR",maximumFractionDigits:2});
const percent=(x:number)=>(x*100).toLocaleString("fr-FR",{maximumFractionDigits:2})+" %";
function Field({name,value}:{name:string;value:string}){
  return <div className="rounded-lg border border-slate-800 bg-slate-950/70 p-4">
    <div className="text-xs text-slate-400">{name}</div>
    <div className="mt-2 text-xl font-semibold text-white">{value}</div>
  </div>;
}
export function OperationalValuationClient(){
  const [data,setData]=useState<OperationalView|null>(null);
  const [filename,setFilename]=useState("");
  const [error,setError]=useState("");
  const [name,setName]=useState<ScenarioName>("BASE");
  const [ptc,setPtc]=useState(true);
  const [priceInput,setPriceInput]=useState("");

  async function openFile(event:ChangeEvent<HTMLInputElement>){
    const file=event.target.files?.[0];event.target.value="";
    if(!file)return;
    setError("");setData(null);setFilename("");
    if(file.size===0||file.size>3_000_000){
      setError("Fichier vide ou trop volumineux (3 Mo maximum).");return;
    }
    try{
      const parsed=projectOperationalValuation(JSON.parse(await file.text()) as unknown);
      setData(parsed);setFilename(file.name);setName("BASE");setPtc(true);
      setPriceInput(String(parsed.referencePrice));
    }catch(err){
      setError(err instanceof OperationalValuationInputError?
        "Modèle non conforme : "+err.code:"JSON illisible ou mal formé.");
    }
  }
  const selected=data?.scenarios[name];
  const fair=selected?(ptc?selected.combinedFairValue:selected.coreFairValue):null;
  const price=Number(priceInput);
  const validPrice=priceInput.trim()!==""&&Number.isFinite(price)&&price>0;
  const upside=fair===null||!validPrice?null:upsideToFairValue(fair,price);
  const waccRows=data?.waccSensitivity
    .filter(row=>Math.abs(row.terminalGrowth-.025)<1e-9)
    .sort((a,b)=>a.wacc-b.wacc)??[];

  return <main className="mx-auto max-w-6xl space-y-6 px-5 py-8">
    <header className="space-y-2">
      <Link href="/admin/orotitan" className="text-sm text-cyan-400">← Retour au Pilotage</Link>
      <p className="text-xs font-semibold uppercase tracking-[.2em] text-cyan-400">OroTitan · espace analyste</p>
      <h1 className="text-3xl font-semibold text-white">Valorisation de travail</h1>
      <p className="max-w-3xl text-sm text-slate-400">
        Comparaison de scénarios financiers importés localement. Lecture dans le navigateur :
        aucun transfert vers le serveur, aucune écriture Registry ou Supabase.
      </p>
    </header>
    <div role="note" className="rounded-xl border border-amber-700/60 bg-amber-950/25 p-4 text-sm text-amber-200">
      <strong>ANALYSE NON CERTIFIÉE.</strong> Cette vue ne crée ni VALUATION_LOCK, ni
      score OroTitan, ni permission de publication. Les paramètres peuvent être des hypothèses d'analyste.
    </div>
    <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
      <label htmlFor="operational-json" className="block text-sm font-semibold text-white">
        Ouvrir un résultat JSON de valorisation opérationnelle
      </label>
      <p className="my-2 text-xs text-slate-400">
        Import local temporaire. Exemple : operational-valuation.json.
        Le fichier financier n'est ni préchargé ni conservé.
      </p>
      <input id="operational-json" type="file" accept=".json,application/json" onChange={openFile}
        className="block max-w-full text-sm text-slate-200 file:mr-4 file:rounded-lg file:border-0 file:bg-cyan-400 file:px-3 file:py-2 file:font-semibold file:text-slate-950"/>
      {filename&&<p className="mt-2 text-xs text-slate-400">Fichier ouvert : {filename}</p>}
      {error&&<p className="mt-3 text-sm text-rose-300" role="alert">{error}</p>}
    </section>
    {data&&selected&&fair!==null?<>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-slate-400">
          Run : <code>{data.runId}</code><span className="ml-2">· Cutoff : {data.cutoff}</span>
        </div>
        <label className="flex items-center gap-2 text-sm text-slate-200">
          <input type="checkbox" checked={ptc} onChange={e=>setPtc(e.target.checked)} className="accent-cyan-400"/>
          Inclure l'effet PTC
        </label>
      </div>
      <nav aria-label="Choisir le scénario" className="grid grid-cols-3 gap-2">
        {SCENARIOS.map(s=><button key={s} type="button" aria-pressed={name===s}
          onClick={()=>setName(s)}
          className={"rounded-lg border px-3 py-3 text-sm font-semibold "+
            (name===s?"border-cyan-400 bg-cyan-950/50 text-cyan-200":"border-slate-700 bg-slate-900 text-slate-300")}>
          {s}
        </button>)}
      </nav>
      <section aria-label="Indicateurs" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Field name="Cours historique au cutoff" value={euro(data.referencePrice)}/>
        <Field name="Juste valeur indicative" value={euro(fair)}/>
        <Field name="Potentiel au cours de référence" value={percent(fair/data.referencePrice-1)}/>
        <Field name="TRI 5 ans · prix de référence" value={percent(ptc?selected.combinedIrr:selected.coreIrr)}/>
      </section>
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="mb-1 text-lg font-semibold text-white">Comparer les scénarios</h2>
          <p className="mb-4 text-xs text-slate-400">Valeur par action au cutoff, avec ou sans PTC.</p>
          <div className="space-y-4">{SCENARIOS.map(s=>{
            const v=ptc?data.scenarios[s].combinedFairValue:data.scenarios[s].coreFairValue;
            const max=Math.max(1,...SCENARIOS.map(key=>ptc?data.scenarios[key].combinedFairValue:data.scenarios[key].coreFairValue));
            return <button key={s} type="button" onClick={()=>setName(s)}
              className="block w-full text-left">
              <div className="mb-1 flex justify-between text-sm"><span>{s}</span><span className="font-semibold text-white">{euro(v)}</span></div>
              <div className="h-3 overflow-hidden rounded-full bg-slate-800">
                <div className={"h-full rounded-full "+(s===name?"bg-cyan-400":"bg-slate-500")}
                  style={{width:(100*v/max)+"%"}}/>
              </div>
            </button>;
          })}</div>
        </section>
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="mb-1 text-lg font-semibold text-white">Simuler un prix d'entrée</h2>
          <p className="mb-3 text-xs text-slate-400">Potentiel par rapport à la juste valeur du scénario, sans recalculer son TRI historique.</p>
          <label htmlFor="test-entry" className="block text-sm text-slate-300">Prix hypothétique (€/action)</label>
          <input id="test-entry" type="number" min="0.01" step="0.01" value={priceInput}
            onChange={e=>setPriceInput(e.target.value)}
            className="mt-1 w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-lg text-white"/>
          <p className="mt-4 text-sm text-slate-300">Potentiel : <strong className="text-xl text-white">
            {upside===null?"Prix invalide":percent(upside)}
          </strong></p>
          <p className="mt-2 text-xs text-amber-200">
            Le TRI affiché en haut reste calculé au cours de référence historique.
          </p>
        </section>
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="text-lg font-semibold text-white">Prix d'entrée conditionnels avec PTC</h2>
          <table className="mt-3 w-full text-sm">
            <thead className="text-slate-400"><tr><th className="py-2 text-left">TRI cible</th><th className="py-2 text-right">Prix maximal</th></tr></thead>
            <tbody className="divide-y divide-slate-800 text-slate-100">
              {([["10 %",selected.ladder.ten],["12,5 %",selected.ladder.twelveFive],["15 %",selected.ladder.fifteen]] as const)
                .map(([k,v])=><tr key={k}><td className="py-3">{k}</td><td className="py-3 text-right font-semibold">{euro(v)}</td></tr>)}
            </tbody>
          </table>
        </section>
        <section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
          <h2 className="text-lg font-semibold text-white">Hypothèses de travail</h2>
          <dl className="mt-3 space-y-3 text-sm text-slate-300">
            {([
              ["WACC Schneider",percent(selected.wacc)],
              ["Actions économiques (proxy)",data.externalSharesProxyM.toFixed(2)+" M"],
              ["Dette nette (proxy)",data.netDebtProxyB.toFixed(2)+" Md€"],
              ["NPV de PTC",selected.ptcNpvB.toFixed(2)+" Md€"],
              ["Poids terminal",percent(selected.terminalWeight)],
            ] as const).map(([k,v])=><div key={k} className="flex justify-between gap-4">
              <dt className="text-slate-400">{k}</dt><dd className="font-medium text-white">{v}</dd>
            </div>)}
          </dl>
        </section>
      </div>
      {waccRows.length>0&&<section className="rounded-xl border border-slate-800 bg-slate-900/60 p-5">
        <h2 className="text-lg font-semibold text-white">Sensibilité WACC · scénario Base</h2>
        <p className="mt-1 text-xs text-slate-400">PTC inclus, croissance terminale 2,5 % ; valeurs pré-calculées dans le JSON.</p>
        <div className="mt-4 flex flex-wrap gap-2">{waccRows.map((r,i)=><div key={i}
          className="rounded-lg border border-slate-700 bg-slate-950 px-4 py-3">
          <div className="text-xs text-slate-400">{percent(r.wacc)}</div>
          <div className="mt-1 font-semibold text-white">{euro(r.combinedFairValue)}</div>
        </div>)}</div>
      </section>}
      <p className="text-xs text-slate-500">Analyste, non autoritatif · calcul : {data.calculationDate} · lecture locale. Aucun score ni gate modifié.</p>
    </>:<p className="text-sm text-slate-500">Charge un fichier JSON de valorisation pour afficher ses résultats, sans import en base.</p>}
  </main>;
}
