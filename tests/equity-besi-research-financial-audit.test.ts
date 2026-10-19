import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const data = JSON.parse(readFileSync(new URL(
  "../docs/orotitan-equity/research-audits/BESI_FY2021_FY2025_FORENSIC_INPUTS_V0_1.json",
  import.meta.url,
), "utf8")) as {
  dataset_id: string;
  status: string;
  data_cutoff: string;
  source_hierarchy: Array<{id:string;published:string;url:string}>;
  years: Array<{
    year:number;revenue:number;ebit:number;net_income_rounded:number;
    pretax_income:number;cfo:number;ppe_capex:number;capitalized_development_cash:number;
    sbc_expense:number;working_capital_movement:number;cash_tax_paid:number;
    da:number;net_financial_expense_adjustment:number;net_cash_interest_received_paid:number;
    cash_deposits:number;interest_bearing_debt:number;equity:number;filing:string;
    calculated:{
      cash_flow_statement_recomputed:number;
      cfo_reconciliation_residual:number;
      after_ppe_and_dev_cash:number;
      after_ppe_dev_and_sbc_sensitivity:number;
      reported_ebit_margin_pct:number;
      after_ppe_and_dev_margin_pct:number;
      net_cash_deposits_minus_debt:number;
      book_financing_proxy_not_roic:number;
    }
  }>;
  five_year_totals:Record<string,number>;
  admission: Record<string,boolean>;
};
const close = (a:number,b:number,reason:string) =>
  assert.ok(Math.abs(a-b)<0.00100001,reason+" "+a+" vs "+b);
const rnd=(x:number)=>Math.round((x+1e-9)*1000)/1000;

test("FY2021–FY2025 versioned audit has exactly five source-linked pre-cutoff records", () => {
  assert.equal(data.status,"DRAFT_RESEARCH_INPUTS_NOT_CERTIFIED");
  assert.equal(data.data_cutoff,"2026-10-12");
  assert.deepEqual(data.years.map(x=>x.year),[2021,2022,2023,2024,2025]);
  const sources = new Map(data.source_hierarchy.map(x=>[x.id,x]));
  for (const year of data.years) {
    const source=sources.get(year.filing);
    assert.ok(source, "Missing source authority: "+year.filing);
    assert.ok(source!.published <= data.data_cutoff,"Future source inadmissible: "+year.year);
    assert.ok(source!.url.startsWith("https://"));
  }
});

test("cash flow bridge matches PRETAX+D&A+SBC+FIN+WC-TAX+INTEREST for every audited year", () => {
  for(const x of data.years){
    const reconciled=rnd(
      x.pretax_income+x.da+x.sbc_expense+x.net_financial_expense_adjustment+
      x.working_capital_movement-x.cash_tax_paid+x.net_cash_interest_received_paid
    );
    close(reconciled,x.cfo,"CFO mismatch FY"+x.year);
    close(reconciled,x.calculated.cash_flow_statement_recomputed,"Stored bridge drift FY"+x.year);
    close(x.calculated.cfo_reconciliation_residual,0,"CFO residual FY"+x.year);
  }
});

test("cash after PPE and capitalized R&D and SBC sensitivity recompute with no hidden residual", () => {
  let sumCfo=0,sumPpe=0,sumDev=0,sumResidual=0,sumSbc=0,sumSensitivity=0;
  for(const x of data.years){
    const residual=rnd(x.cfo-x.ppe_capex-x.capitalized_development_cash);
    const sensitivity=rnd(residual-x.sbc_expense);
    close(residual,x.calculated.after_ppe_and_dev_cash,"Residual FY"+x.year);
    close(sensitivity,x.calculated.after_ppe_dev_and_sbc_sensitivity,"SBC FY"+x.year);
    close(100*x.ebit/x.revenue,x.calculated.reported_ebit_margin_pct,
      "EBIT margin FY"+x.year);
    close(100*residual/x.revenue,x.calculated.after_ppe_and_dev_margin_pct,
      "Cash margin FY"+x.year);
    sumCfo+=x.cfo;sumPpe+=x.ppe_capex;sumDev+=x.capitalized_development_cash;
    sumResidual+=residual;sumSbc+=x.sbc_expense;sumSensitivity+=sensitivity;
    close(x.cash_deposits-x.interest_bearing_debt,
      x.calculated.net_cash_deposits_minus_debt,"Net cash FY"+x.year);
    close(x.equity+x.interest_bearing_debt-x.cash_deposits,
      x.calculated.book_financing_proxy_not_roic,"Financing proxy FY"+x.year);
  }
  for(const [k,v] of Object.entries({
    cfo:sumCfo,ppe_capex:sumPpe,capitalized_development_cash:sumDev,
    after_ppe_and_dev_cash:sumResidual,sbc_expense:sumSbc,
    after_ppe_dev_and_sbc_sensitivity:sumSensitivity,
  }))close(rnd(v),data.five_year_totals[k],"five-year total "+k);
});

test("2024 unusually low tax and separate 2025 investment property must not be hidden", () => {
  const y24=data.years.find(x=>x.year===2024)!;
  const y25=data.years.find(x=>x.year===2025)!;
  close(188.520*(1-0.131),163.824,"2024 illustrative after-tax");
  close(y25.cfo-y25.ppe_capex-y25.capitalized_development_cash,136.352,
    "2025 mechanical cash residue");
  // Unusual 2025 property acquisition is not silently included in PPE capex.
  assert.equal((y25 as unknown as {investment_property_acquisition:number}).investment_property_acquisition,5.164);
  close(y24.cfo,201.083,"2024 audited CFO");
});

test("audit remains explicitly non-authoritative despite complete arithmetic reconciliation", () => {
  const blockers=[
    "full_FY2021_FY2022_standalone_audited_note_crosscheck",
    "C003_conflict_resolved","G012_source_coverage_resolved",
    "maintenance_vs_growth_development_split_proven",
    "working_capital_normalization_approved",
    "sbc_economic_cost_approved","normalized_sustainable_cash_taxes_approved",
    "roic_economic_certified","owner_earnings_certified",
    "research_final_admitted","handoff_deep_dive_authorized",
    "writer_v31_authorized","publication_authorized",
  ];
  for(const k of blockers)assert.equal(data.admission[k],false,k+" cannot be admitted by an exploratory spreadsheet");
  assert.equal(data.admission.financial_statement_columns_cross_checked,true);
});
