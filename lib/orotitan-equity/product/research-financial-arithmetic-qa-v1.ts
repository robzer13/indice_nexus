/**
 * OroTitan - reusable, exact-decimal, issuer-neutral arithmetic QA.
 * This module reconciles REPORTED CANDIDATES; it cannot validate their source.
 */
export type ArithmeticTerm=Readonly<{label:string;value:string}>;
export type ArithmeticCheck=Readonly<{
 id:string;description:string;unit:string;decimalPlaces:number;
 terms:readonly ArithmeticTerm[];reportedComparator:string;
 sourceIds:readonly string[];
}>;
export type ArithmeticResult=Readonly<{
 id:string;description:string;unit:string;sourceIds:readonly string[];
 calculatedSum:string;reportedComparator:string;difference:string;
 roundingTolerance:string;
 verdict:'EXACT_MATCH'|'ROUNDING_COMPATIBLE'|'OUT_OF_TOLERANCE';
 passed:boolean;claimValidated:false;registryAdmitted:false;
}>;
const fail=():never=>{throw Error('RESEARCH_ARITHMETIC_INPUT_INVALID');};
function parse(x:string,places:number):bigint{
 if(typeof x!=='string'||x.length>32)fail();
 const match=/^(-?)(0|[1-9]\d*)(?:\.(\d+))?$/.exec(x);
 if(!match)fail();
 const fraction=match[3]??'';
 if(fraction.length>places)fail();
 const scale=10n**BigInt(places);
 const value=BigInt(match[2])*scale+
  BigInt((fraction.padEnd(places,'0'))||'0');
 return match[1]==='-'?-value:value;
}
function render(n:bigint,places:number):string{
 const neg=n<0n;const absolute=neg?-n:n;
 const scale=10n**BigInt(places);
 const whole=absolute/scale;
 const frac=places?'.'+(absolute%scale).toString().padStart(places,'0'):'';
 return (neg?'-':'')+whole.toString()+frac;
}
export function checkReportedArithmetic(check:ArithmeticCheck):ArithmeticResult{
 if(!check||!/^[A-Z0-9_-]{3,80}$/.test(check.id)||
    typeof check.description!=='string'||!check.description||
    typeof check.unit!=='string'||!check.unit||
    !Number.isSafeInteger(check.decimalPlaces)||check.decimalPlaces<0||
    check.decimalPlaces>6||!Array.isArray(check.terms)||
    check.terms.length<1||check.terms.length>50||
    !Array.isArray(check.sourceIds)||check.sourceIds.length<1||
    check.sourceIds.some(s=>typeof s!=='string'||!s.trim()))fail();
 const total=check.terms.reduce((sum,t)=>{
  if(!t||typeof t.label!=='string'||!t.label.trim())fail();
  return sum+parse(t.value,check.decimalPlaces);
 },0n);
 const comparator=parse(check.reportedComparator,check.decimalPlaces);
 const difference=total-comparator;
 // Rounded input and rounded comparator may each differ by up to half the
 // last quoted decimal place. Compare exact integers in HALF-units.
 const roundingToleranceHalfUnits=BigInt(check.terms.length+1);
 const absDiff=difference<0n?-difference:difference;
 const verdict=difference===0n?'EXACT_MATCH':
  2n*absDiff<=roundingToleranceHalfUnits?'ROUNDING_COMPATIBLE':
  'OUT_OF_TOLERANCE';
 // Tolerance in one extra decimal digit; half-unit = 5 of that digit.
 const tolRender=render(roundingToleranceHalfUnits*5n,check.decimalPlaces+1);
 return {
  id:check.id,description:check.description,unit:check.unit,
  sourceIds:check.sourceIds,calculatedSum:render(total,check.decimalPlaces),
  reportedComparator:render(comparator,check.decimalPlaces),
  difference:render(difference,check.decimalPlaces),
  roundingTolerance:tolRender,verdict,passed:verdict!=='OUT_OF_TOLERANCE',
  claimValidated:false,registryAdmitted:false,
 };
}
export function runArithmeticChecks(checks:readonly ArithmeticCheck[]){
 const ids=new Set<string>();
 return checks.map(check=>{
  if(ids.has(check.id))fail();
  ids.add(check.id);
  return checkReportedArithmetic(check);
 });
}
