'use client';
import {useEffect,useState} from 'react';

const endpoint='/admin/research-factory/control-tower/c8-gate/owner-session';
type AuthStatus={ok?:boolean;authenticated?:boolean;error?:string};

export function C8OwnerOtpPanel(){
  const [email,setEmail]=useState('');
  const [otp,setOtp]=useState('');
  const [authenticated,setAuthenticated]=useState(false);
  const [loading,setLoading]=useState(false);
  const [notice,setNotice]=useState('');
  const [error,setError]=useState('');

  useEffect(()=>{
    let active=true;
    void fetch(endpoint,{cache:'no-store',credentials:'same-origin'})
      .then(async response=>{
        const result=await response.json() as AuthStatus;
        if(active&&response.ok&&result.ok===true)setAuthenticated(result.authenticated===true);
      })
      .catch(()=>{if(active)setAuthenticated(false);});
    return ()=>{active=false;};
  },[]);

  async function action(kind:'request'|'verify'|'logout'){
    setLoading(true);setError('');setNotice('');
    try{
      const data=kind==='request'?{email:email.trim()}:
        kind==='verify'?{email:email.trim(),token:otp.trim()}:null;
      const response=await fetch(`${endpoint}/${kind}`,{
        method:'POST',credentials:'same-origin',cache:'no-store',
        ...(data?{headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{}),
      });
      const result=await response.json() as AuthStatus;
      if(!response.ok||result.ok!==true)throw Error(result.error??'OWNER_AUTH_UNAVAILABLE');
      if(kind==='request')setNotice('Demande de code envoyée. Consulte la messagerie du compte propriétaire.');
      if(kind==='verify'){
        setAuthenticated(true);setOtp('');setNotice('Compte propriétaire attesté pour cette session.');
      }
      if(kind==='logout'){
        setAuthenticated(false);setOtp('');setNotice('Session propriétaire déconnectée.');
      }
    }catch(reason){setError(reason instanceof Error?reason.message:'OWNER_AUTH_UNAVAILABLE');}
    finally{setLoading(false);}
  }

  return <section className="space-y-4 rounded-lg border border-cyan-800 bg-slate-950 p-4">
    <p className="text-sm">Session owner : <strong>{authenticated?'ATTESTÉE':'NON ATTESTÉE'}</strong></p>
    <label className="block text-sm">Email du compte propriétaire SHADOW
      <input type="email" required autoComplete="email" value={email}
        onChange={event=>setEmail(event.target.value)}
        className="mt-1 w-full rounded border border-slate-600 bg-slate-900 p-2"/>
    </label>
    <button type="button" disabled={loading||!email.trim()} onClick={()=>void action('request')}
      className="rounded border border-cyan-500 px-3 py-2 disabled:opacity-50">Envoyer l’OTP</button>
    <label className="block text-sm">Code reçu par email
      <input inputMode="numeric" autoComplete="one-time-code" value={otp}
        onChange={event=>setOtp(event.target.value)}
        className="mt-1 w-full rounded border border-slate-600 bg-slate-900 p-2"/>
    </label>
    <button type="button" disabled={loading||!/^\d{6,10}$/.test(otp)} onClick={()=>void action('verify')}
      className="rounded border border-emerald-500 px-3 py-2 disabled:opacity-50">Attester</button>
    {authenticated&&<button type="button" disabled={loading} onClick={()=>void action('logout')}
      className="ml-3 text-sm underline disabled:opacity-50">Déconnecter le compte owner</button>}
    {notice&&<p role="status" className="text-sm text-emerald-300">{notice}</p>}
    {error&&<p role="alert" className="text-sm text-rose-300">{error}</p>}
  </section>;
}
