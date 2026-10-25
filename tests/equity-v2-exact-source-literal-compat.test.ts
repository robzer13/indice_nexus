import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import Ajv2020 from "ajv/dist/2020";
import {
  buildV1FieldLocalLiteralCompatibleSchema,
  EXACT_SOURCE_SEMANTIC_COMPATIBILITY,
} from "../lib/orotitan-equity/v2/exact-source-semantic-compatibility";

const original = (): Record<string, unknown> =>
  JSON.parse(readFileSync(new URL("../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.10.json", import.meta.url), "utf8"));

type Rec = Record<string, unknown>;
function defLeaf(schema: Rec, node: "analyticalMetrics" | "operationalState", field: string): Rec {
  const defs=schema.$defs as Rec;
  const props=(defs[node] as Rec).properties as Rec;
  return props[field] as Rec;
}

test("exact two certified source enum literals added, frozen schema unchanged",()=>{
  const frozen=original();
  const hashBefore=JSON.stringify(frozen);
  const patched=buildV1FieldLocalLiteralCompatibleSchema(frozen);
  assert.equal(JSON.stringify(frozen),hashBefore);
  const ri=defLeaf(patched,"analyticalMetrics","return_interpretability").enum as string[];
  const rf=defLeaf(patched,"operationalState","refresh_class").enum as string[];
  assert.equal(ri.at(-1),"MEDIUM");
  assert.equal(rf.at(-1),"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED");
  assert.equal(ri.length,5);
  assert.equal(rf.length,5);

  // No collateral relaxation of any V1 analytical, numeric or terminal rule.
  const expected=original();
  const r=defLeaf(expected,"analyticalMetrics","return_interpretability");
  const o=defLeaf(expected,"operationalState","refresh_class");
  r.enum=[...(r.enum as string[]),"MEDIUM"];
  o.enum=[...(o.enum as string[]),"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED"];
  expected.$id=EXACT_SOURCE_SEMANTIC_COMPATIBILITY.compatibilitySchemaId;
  expected.title=patched.title;
  expected.description=patched.description;
  assert.deepEqual(patched,expected);
});

test("old validator rejects both source literals; opt-in overlay accepts exactly them",()=>{
  const frozen=original();
  const patched=buildV1FieldLocalLiteralCompatibleSchema(frozen);
  const ajv=new Ajv2020({strict:false,allErrors:true});
  const originalEnum=ajv.compile({
    type:"object",additionalProperties:false,required:["returnInterpretability","refreshClass"],
    properties:{
      returnInterpretability:defLeaf(frozen,"analyticalMetrics","return_interpretability"),
      refreshClass:defLeaf(frozen,"operationalState","refresh_class"),
    },
  });
  const newEnum=ajv.compile({
    type:"object",additionalProperties:false,required:["returnInterpretability","refreshClass"],
    properties:{
      returnInterpretability:defLeaf(patched,"analyticalMetrics","return_interpretability"),
      refreshClass:defLeaf(patched,"operationalState","refresh_class"),
    },
  });
  const literals={
    returnInterpretability:"MEDIUM",
    refreshClass:"ACTIVATION_CHECK_OR_TARGETED_DELTA_IF_PRICE_ZONE_REACHED",
  };
  assert.equal(originalEnum(literals),false);
  assert.equal(newEnum(literals),true,JSON.stringify(newEnum.errors));
  for(const bad of [
    {returnInterpretability:"LOW",refreshClass:literals.refreshClass},
    {returnInterpretability:literals.returnInterpretability,refreshClass:"TARGETED_DELTA"},
    {returnInterpretability:null,refreshClass:literals.refreshClass},
  ]) assert.equal(newEnum(bad),false);
  assert.equal(originalEnum({
    returnInterpretability:"INTERPRETABLE_WITH_LIMITATIONS",
    refreshClass:"NOT_APPLICABLE",
  }),true);
});

test("reject incorrect frozen enum source — never widen unknown future revisions",()=>{
  const schema=original();
  (defLeaf(schema,"analyticalMetrics","return_interpretability").enum as string[]).push("UNREVIEWED");
  assert.throws(()=>buildV1FieldLocalLiteralCompatibleSchema(schema),/FROZEN_RETURN_INTERPRETABILITY_ENUM_CHANGED/);
  const schema2=original();
  (defLeaf(schema2,"operationalState","refresh_class").enum as string[]).push("SOMETHING_NEW");
  assert.throws(()=>buildV1FieldLocalLiteralCompatibleSchema(schema2),/FROZEN_REFRESH_CLASS_ENUM_CHANGED/);
});
