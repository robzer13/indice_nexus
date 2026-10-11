import 'server-only';
import { createServerSupabaseClient } from '@/lib/supabase/server';
import { loadPaperDashboard, latestStoredQuote, eurNumber } from '@/lib/data/paper-portfolio';
import { evaluateLongTermPaperSignal } from '@/lib/domain/paper-signals';
import { planPaperAutoFill, trustedExchangeQuote } from '@/lib/domain/paper-autopilot';

export type PaperAutomationSettings = {
  portfolio_id: string; enabled: boolean;
  maximum_initial_capital_weight: number | string;
  simulated_fee_eur: number | string;
};
export type PaperDailyReport = {
  status: 'DISABLED' | 'COMPLETE' | 'PARTIAL' | 'MIGRATION_NOT_INSTALLED' | 'PORTFOLIO_NOT_INITIALIZED';
  evaluated: number; signals: number; fills: number;
  issues: Array<{ issuer: string; reason: string }>;
};

export async function readPaperAutomationSettings(portfolioId: string): Promise<PaperAutomationSettings | null | 'MIGRATION_NOT_INSTALLED'> {
  const { data, error } = await createServerSupabaseClient().from('paper_autopilot_settings')
    .select('*').eq('portfolio_id',portfolioId).maybeSingle();
  if (error && (error.code==='42P01' || error.code==='PGRST205')) return 'MIGRATION_NOT_INSTALLED';
  if (error) throw new Error('Cannot read paper automation settings: '+error.message);
  return data as PaperAutomationSettings | null;
}
function reason(error: unknown): string { return error instanceof Error ? error.message : 'UNKNOWN_ERROR'; }

export async function runPaperDailyAutomation(): Promise<PaperDailyReport> {
  const report: PaperDailyReport={status:'DISABLED',evaluated:0,signals:0,fills:0,issues:[]};
  const initial=await loadPaperDashboard();
  if(initial.migrationMissing){report.status='MIGRATION_NOT_INSTALLED';return report;}
  if(!initial.portfolio){report.status='PORTFOLIO_NOT_INITIALIZED';return report;}
  const settings=await readPaperAutomationSettings(initial.portfolio.id);
  if(settings==='MIGRATION_NOT_INSTALLED'){report.status='MIGRATION_NOT_INSTALLED';return report;}
  if(!settings?.enabled) return report;
  report.status='COMPLETE';
  const db=createServerSupabaseClient();
  const now=new Date().toISOString();

  // Reread the state per issuer: an earlier simulated fill may change cash.
  // Never use the visible UI cache as trading authority.
  for(const candidate of initial.policies){
    try {
      const view=await loadPaperDashboard();
      const currentPolicy=view.policies.find(p=>p.company_id===candidate.company_id);
      if(!currentPolicy||currentPolicy.id!==candidate.id) continue;
      const company=view.companies.find(c=>c.id===candidate.company_id);
      if(!company||company.currency!=='EUR'||company.quote_unit!=='MAJOR'){
        report.issues.push({issuer:candidate.company_id,reason:'UNSUPPORTED_CURRENCY_OR_UNIT'});continue;
      }
      const {data:dossier,error:dossierError}=await db.from('research_dossiers')
        .select('current_snapshot_id').eq('issuer_id',candidate.company_id)
        .eq('active',true).eq('current_snapshot_id',candidate.snapshot_id).limit(1).maybeSingle();
      if(dossierError||!dossier){
        report.issues.push({issuer:candidate.company_id,reason:'CANONICAL_SNAPSHOT_NOT_CURRENT'});continue;
      }
      const quote=await latestStoredQuote(candidate.company_id);
      if(!quote||!trustedExchangeQuote(quote,now)){
        report.issues.push({issuer:candidate.company_id,reason:'EXCHANGE_QUOTE_NOT_VERIFIABLE'});continue;
      }
      let pos=view.positions.find(p=>p.company_id===candidate.company_id);
      let held=pos?eurNumber(pos.quantity):0;
      const {data:prior,error:priorError}=await db.from('paper_signals').select('*')
        .eq('policy_id',candidate.id).order('created_at',{ascending:false}).limit(1).maybeSingle();
      if(priorError) throw new Error('LAST_SIGNAL_LOOKUP_FAILED: '+priorError.message);
      if(prior&&(prior.action==='BUY_CANDIDATE'||prior.action==='SELL_CANDIDATE')
        &&Date.parse(quote.as_of)>Date.parse(prior.created_at)
        &&Date.parse(quote.created_at)>Date.parse(prior.created_at)){
        const {data:existing,error:existingError}=await db.from('paper_fills')
          .select('id').eq('signal_id',prior.id).limit(1).maybeSingle();
        if(existingError) throw new Error('FILL_LOOKUP_FAILED: '+existingError.message);
        if(!existing){
          const plan=planPaperAutoFill(prior.action,eurNumber(quote.price),held,{
            enabled:true,initialCapitalEur:eurNumber(view.portfolio!.initial_cash_eur),
            availableCashEur:eurNumber(view.portfolio!.cash_eur),
            maxInitialCapitalWeight:eurNumber(settings.maximum_initial_capital_weight),
            simulatedFeeEur:eurNumber(settings.simulated_fee_eur),
          });
          if(plan.permitted){
            const {error:fillError}=await db.rpc('paper_execute_fill',{
              p_signal:prior.id,p_quote:quote.id,p_quantity:plan.quantity,
              p_fee_eur:eurNumber(settings.simulated_fee_eur),
            });
            if(fillError&&fillError.code!=='23505') throw new Error('VIRTUAL_FILL_REJECTED: '+fillError.message);
            if(!fillError) {report.fills++;held=prior.action==='BUY_CANDIDATE'?held+plan.quantity:held-plan.quantity;}
          } else report.issues.push({issuer:candidate.company_id,reason:plan.reason});
        }
      }

      const result=evaluateLongTermPaperSignal({
        strategy:{
          profile:candidate.profile,thesisState:candidate.thesis_state,
          milestoneState:candidate.milestone_state,
          entryPrice:candidate.entry_price_eur===null?null:eurNumber(candidate.entry_price_eur),
          exitTargetPrice:candidate.exit_target_eur===null?null:eurNumber(candidate.exit_target_eur),
        },
        snapshot:{
          id:candidate.snapshot_id,analysisDate:company.analysis_date,
          canonicalEntryPrice:company.price_o90,
          eligibleForInvestment:['OROTITAN','FINALIST','PRICE_WAIT'].includes(company.status),
          coreQualityConfirmed:company.quality_orotitan===true,
        },
        quote:{price:eurNumber(quote.price),asOf:quote.as_of},
        positionQuantity:held,now,
      });
      report.evaluated++;
      if(result.action==='DATA_BLOCKED') {report.issues.push({issuer:candidate.company_id,reason:result.reason});continue;}
      const {error:signalError}=await db.rpc('paper_record_signal',{
        p_policy:candidate.id,p_quote:quote.id,p_ruleset:result.ruleset,
        p_action:result.action,p_reason:result.reason,
      });
      if(signalError) throw new Error('SIGNAL_RECORD_REJECTED: '+signalError.message);
      report.signals++;
    }catch(e){report.issues.push({issuer:candidate.company_id,reason:reason(e)});}
  }
  if(report.issues.length)report.status='PARTIAL';
  return report;
}
