import { createHash, timingSafeEqual } from 'node:crypto';
import { NextRequest, NextResponse } from 'next/server';
import { runPaperDailyAutomation } from '@/lib/automation/paper-daily';
export const dynamic='force-dynamic';
export const runtime='nodejs';
export const maxDuration=60;
function authorized(request:NextRequest):boolean {
  const secret=process.env.CRON_SECRET;
  if(!secret)return false;
  const received=request.headers.get('authorization');
  if(!received?.startsWith('Bearer '))return false;
  const a=createHash('sha256').update(received.slice(7)).digest();
  const b=createHash('sha256').update(secret).digest();
  return timingSafeEqual(a,b);
}
export async function GET(request:NextRequest){
  if(!authorized(request))return NextResponse.json({ok:false,error:'Unauthorized'},{status:401});
  try{
    const result=await runPaperDailyAutomation();
    return NextResponse.json({ok:result.status!=='PARTIAL',...result},
      {status:result.status==='PARTIAL'?207:200});
  }catch(e){return NextResponse.json({ok:false,error:e instanceof Error?e.message:'Paper automation failed'},{status:500});}
}
