/**
 * CI-only PostgREST + Storage subset for recovery E2E, backed by disposable
 * PostgreSQL 17 and an in-memory private object store. No hosted Supabase.
 * All routes are loopback and already protected by the fake CI API key.
 */
import {createHash} from 'node:crypto';
const RUN_ID='f4ae40f7-eb9b-4472-b124-9fb152b739ce';
const BUCKET='orotitan-source-files-v1';
const OBJECT=/^research-recovery-v1\/asml-observation\/q2-2026\/[0-9a-f]{64}\.pdf$/;
const JOURNAL='orotitan_product_recovery_attempts_shadow_v1';
const storage=new Map();
let uploads=0;
const digest=(value)=>createHash('sha256').update(value).digest('hex');
async function bytes(req,max=65536) {
  const out=[];let n=0;
  for await(const chunk of req){
    n+=chunk.length;
    if(n>max)throw Error('RECOVERY_FIXTURE_BODY_TOO_LARGE');
    out.push(chunk);
  }
  return Buffer.concat(out);
}
const json=(value)=>JSON.stringify(value);
export async function handleRecoveryFixture({req,res,url,query,emit}) {
  if(process.env.CI!=='true' || process.env.OROTITAN_PRODUCT_BROWSER_FIXTURE!=='1' ||
     process.env.OROTITAN_RECOVERY_E2E_FIXTURE!=='1' ||
     process.env.NODE_ENV!=='development' || process.env.VERCEL_ENV)
    return false;
  if(req.method==='GET' && url.pathname==='/rest/v1/orotitan_artifacts'){
    emit(res,200,[]);return true;
  }
  if(req.method==='GET' && url.pathname==='/storage/v1/bucket/'+BUCKET){
    emit(res,200,{id:BUCKET,public:false,allowed_mime_types:['application/pdf']});
    return true;
  }
  const prefix='/storage/v1/object/'+BUCKET+'/';
  if(url.pathname.startsWith(prefix)){
    const path=decodeURIComponent(url.pathname.slice(prefix.length));
    if(!OBJECT.test(path)){emit(res,403,{error:'INVALID_FIXTURE_PATH'});return true;}
    if(req.method==='POST'){
      const payload=await bytes(req,2_000_500);
      const objectHash=path.split('/').at(-1).slice(0,64);
      if(payload.length<10000 || payload.length>2_000_000 ||
         digest(payload)!==objectHash ||
         req.headers['content-type']?.split(';')[0]!=='application/pdf'){
        emit(res,400,{error:'RECOVERY_FIXTURE_CONTENT_INVALID'});return true;
      }
      if(storage.has(path)){
        emit(res,409,{code:'KeyAlreadyExists',message:'The resource already exists'});
        return true;
      }
      storage.set(path,payload);uploads++;
      emit(res,200,{Key:BUCKET+'/'+path});return true;
    }
    if(req.method==='GET'){
      const file=storage.get(path);
      if(!file){emit(res,404,{error:'ObjectNotFound'});return true;}
      res.writeHead(200,{'content-type':'application/pdf','cache-control':'no-store'});
      res.end(file);
      return true;
    }
    emit(res,405,{error:'READ_WRITE_CREATE_ONLY'});return true;
  }
  if(url.pathname==='/__fixture/recovery-state' && req.method==='GET'){
    emit(res,200,{objectCount:storage.size,uploads,
      objects:[...storage].map(([path,data])=>({path,size:data.length,sha256:digest(data)}))});
    return true;
  }
  if(url.pathname!=='/rest/v1/'+JOURNAL)return false;
  if(req.method==='GET'){
    const run=url.searchParams.get('run_id');
    if(run!=='eq.'+RUN_ID){emit(res,400,{code:'22P02',message:'RUN_FILTER_REQUIRED'});return true;}
    const text=query("SET ROLE service_role; "+
      "SELECT coalesce(json_agg(json_build_object("+
      "'blocker_code',blocker_code,'strategy',strategy,'outcome',outcome,"+
      "'detail',detail,'evidence_refs',evidence_refs) ORDER BY created_at),"+
      "'[]'::json)::text FROM public."+JOURNAL+
      " WHERE run_id='"+RUN_ID+"'::uuid AND stage_code='DEEP_DIVE'"+
      " AND run_state_version=6 AND stage_state_version=2; RESET ROLE;");
    emit(res,200,JSON.parse(text||'[]'));return true;
  }
  if(req.method!=='POST'){emit(res,405,{code:'405',message:'Append only'});return true;}
  const data=JSON.parse((await bytes(req,5000)).toString('utf8'));
  const fields=['run_id','stage_code','run_state_version','stage_state_version',
    'blocker_code','strategy','outcome','detail','evidence_refs'];
  if(!data || Array.isArray(data) || Object.keys(data).length!==fields.length ||
     !fields.every(key=>Object.hasOwn(data,key)) ||
     data.run_id!==RUN_ID || data.stage_code!=='DEEP_DIVE' ||
     data.run_state_version!==6 || data.stage_state_version!==2 ||
     typeof data.blocker_code!=='string' || data.blocker_code.length>100 ||
     typeof data.detail!=='string' || data.detail.length>1500 ||
     !Array.isArray(data.evidence_refs) || data.evidence_refs.length>50){
    emit(res,400,{code:'22P02',message:'INVALID_TEST_JOURNAL_INPUT'});return true;
  }
  const encoded=Buffer.from(json(data)).toString('base64');
  const decoded="convert_from(decode('"+encoded+"','base64'),'UTF8')::jsonb";
  const sql="SET ROLE service_role; WITH raw AS(SELECT "+decoded+" AS j) "+
    "INSERT INTO public."+JOURNAL+" ("+fields.join(',')+") "+
    "SELECT (j->>'run_id')::uuid,j->>'stage_code',"+
    "(j->>'run_state_version')::bigint,(j->>'stage_state_version')::bigint,"+
    "j->>'blocker_code',j->>'strategy',j->>'outcome',j->>'detail',"+
    "j->'evidence_refs' FROM raw; RESET ROLE;";
  try {
    query(sql);emit(res,201,{});
  } catch(error) {
    if(/duplicate key|unique constraint/.test(String(error?.stderr??'')))
      emit(res,409,{code:'23505',message:'Immutable journal key already present'});
    else emit(res,400,{code:'PGRST400',message:'Journal insert rejected'});
  }
  return true;
}
