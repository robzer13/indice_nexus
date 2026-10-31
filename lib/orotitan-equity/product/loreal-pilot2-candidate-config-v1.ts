/**
 * L'Oreal pilot 2: explicit NON-AUTHORITATIVE fixture transcribed from issuer
 * HTML and Phase-0 handoff as of 2026-10-30. Not issuer PDF bytes.
 * Never pass this module directly to an Evidence Ledger or Registry writer.
 */
import type {ArithmeticCheck} from './research-financial-arithmetic-qa-v1';
import type {NumericProbe} from './research-pdf-numeric-core-v1';
export const LOREAL_PILOT2_CUTOFF='2026-10-30' as const;
export const LOREAL_PILOT2_RUN_ID:null=null;
export const LOREAL_PILOT2_SOURCES=[
 {id:'LR-S01',title:'2025 Universal Registration Document',
  date:'2026-03-18',type:'PDF_CANDIDATE',sha256:null},
 {id:'LR-S02',title:'FY2025 Annual Results',
  date:'2026-02-12',type:'ISSUER_HTML',sha256:null},
 {id:'LR-S03',title:'2026 Half-Year Financial Report',
  date:'2026-07-31',type:'PDF_CANDIDATE',sha256:null},
 {id:'LR-S04',title:'2026 Half-Year Results',
  date:'2026-07-29',type:'ISSUER_HTML',sha256:null},
 {id:'LR-S05',title:'Sales at 30 September 2026',
  date:'2026-10-22',type:'ISSUER_HTML',sha256:null},
] as const;
export const LOREAL_PILOT2_PDF_PROBES:Readonly<Record<string,readonly NumericProbe[]>>={
 'LR-S01':[
  {claimId:'LR-C001',numeric:'44052.0',contextHints:['sales','revenue']},
  {claimId:'LR-C002',numeric:'74.3',contextHints:['gross margin']},
  {claimId:'LR-C003',numeric:'20.2',contextHints:['operating margin']},
  {claimId:'LR-C004',numeric:'7161.6',contextHints:['net cash flow']},
  {claimId:'LR-C012',numeric:'533783028',contextHints:['shares']},
 ],
 'LR-S03':[
  {claimId:'LR-C005',numeric:'23776.1',contextHints:['sales']},
  {claimId:'LR-C006',numeric:'74.8',contextHints:['gross margin']},
  {claimId:'LR-C007',numeric:'21.3',contextHints:['operating margin']},
 ],
};
// All numbers below originate from the pre-admission Phase-0 fixture / issuer
// releases. They have NOT been reconciled to private PDF byte receipts.
const divisions9m=[
 {label:'Professional Products',value:'4389.8'},
 {label:'Consumer Products',value:'12687.1'},
 {label:'L’Oreal Luxe',value:'12173.3'},
 {label:'Dermatological Beauty',value:'6087.4'},
];
const regions9m=[
 {label:'Europe',value:'12177.0'},
 {label:'North America',value:'9218.2'},
 {label:'North Asia',value:'7743.5'},
 {label:'SAPMENA–SSA',value:'3470.0'},
 {label:'Latin America',value:'2728.9'},
];
const divisionsH1=[
 {label:'Professional Products',value:'2920.8'},
 {label:'Consumer Products',value:'8642.8'},
 {label:'L’Oreal Luxe',value:'7996.7'},
 {label:'Dermatological Beauty',value:'4215.8'},
];
const divisionsQ3=[
 {label:'Professional Products',value:'1469.1'},
 {label:'Consumer Products',value:'4044.3'},
 {label:'L’Oreal Luxe',value:'4176.6'},
 {label:'Dermatological Beauty',value:'1871.6'},
];
const regionsQ3=[
 {label:'Europe',value:'4043.2'},
 {label:'North America',value:'3151.1'},
 {label:'North Asia',value:'2229.0'},
 {label:'SAPMENA–SSA',value:'1222.7'},
 {label:'Latin America',value:'915.5'},
];
export const LOREAL_PILOT2_QA:readonly ArithmeticCheck[]=[
 {id:'QA-9M-DIV',description:'9M 2026 by division',unit:'EUR_MILLION',
  decimalPlaces:1,terms:divisions9m,reportedComparator:'35337.6',
  sourceIds:['LR-S05']},
 {id:'QA-9M-REG',description:'9M 2026 by region',unit:'EUR_MILLION',
  decimalPlaces:1,terms:regions9m,reportedComparator:'35337.6',
  sourceIds:['LR-S05']},
 {id:'QA-H1-DIV',description:'H1 2026 by division',unit:'EUR_MILLION',
  decimalPlaces:1,terms:divisionsH1,reportedComparator:'23776.1',
  sourceIds:['LR-S04']},
 {id:'QA-Q3-DIV',description:'Q3 2026 by division',unit:'EUR_MILLION',
  decimalPlaces:1,terms:divisionsQ3,reportedComparator:'11561.4',
  sourceIds:['LR-S05']},
 {id:'QA-Q3-REG',description:'Q3 2026 by region',unit:'EUR_MILLION',
  decimalPlaces:1,terms:regionsQ3,reportedComparator:'11561.4',
  sourceIds:['LR-S05']},
 {id:'QA-9M-TEMPORAL',description:'H1 + Q3 versus 9M 2026',
  unit:'EUR_MILLION',decimalPlaces:1,
  terms:[{label:'H1 2026',value:'23776.1'},
         {label:'Q3 2026',value:'11561.4'}],
  reportedComparator:'35337.6',sourceIds:['LR-S04','LR-S05']},
 {id:'QA-FY25-BRIDGE',description:'9M + Q4 versus FY2025',
  unit:'EUR_MILLION',decimalPlaces:1,
  terms:[{label:'9M 2025',value:'32807.0'},
         {label:'Q4 2025',value:'11245.0'}],
  reportedComparator:'44052.0',sourceIds:['LR-S02']},
];
export const LOREAL_PILOT2_STATUS={
 evidenceClaimsValidated:0 as const,registryAdmitted:false as const,
 pdfOriginalBytesCaptured:0 as const,readyForDeepDive:'NOT_EVALUATED' as const,
 researchStageAdmitted:false as const,
 dataProvenance:'PUBLIC_ISSUER_HTML_TRANSCRIPTION_PRE_ADMISSION' as const,
};
