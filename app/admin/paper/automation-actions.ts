'use server';
import { redirect } from 'next/navigation';
import { revalidatePath } from 'next/cache';
import { requireAdminSession } from '@/lib/auth/admin-session';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { loadPaperDashboard } from '@/lib/data/paper-portfolio';
export async function configurePaperAutopilotAction(form:FormData):Promise<void>{
  try{
    await requireAdminSession();
  }catch{redirect('/admin');}
  try{
    const view=await loadPaperDashboard();
    if(!view.portfolio)throw new Error('Portefeuille non initialisé.');
    const enabled=form.get('enabled')==='true';
    // Fixed operational cap. Increasing allocations requires a reviewed policy successor.
    const {error}=await createServerSupabaseClient().rpc('paper_configure_autopilot',{
      p_portfolio_id:view.portfolio.id,p_enabled:enabled,p_maximum_weight:0.05,p_fee_eur:1,
    });
    if(error)throw new Error(error.message);
  }catch(e){
    redirect('/admin/paper?error='+encodeURIComponent(e instanceof Error?e.message:'Configuration impossible.'));
  }
  revalidatePath('/admin/paper');
  redirect('/admin/paper?success='+encodeURIComponent('Mode fictif automatique mis à jour. Aucun ordre réel.'));
}
