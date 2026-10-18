import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

/** Preserve frozen V2 bytes. The Next.js projection can alter ONLY schema
 * loading and relative product import, never the analytical business rules. */
test("D9 Next.js V2 projection is a mechanically exact copy of frozen V2 logic", () => {
  const original = readFileSync(
    new URL("../lib/orotitan-equity/v2/research-snapshot-schema.ts", import.meta.url), "utf8");
  const projection = readFileSync(
    new URL("../lib/orotitan-equity/v3/pilotage-v2-bundled-validator.ts", import.meta.url), "utf8");
  const pref = "/** Bundler-compatible V2 validator projection for the D9 operator path.\n * Strictly preserve frozen V2 validator logic; only load JSON schema statically.\n * Original V2 implementation remains byte-pinned and unchanged.\n */\n";
  const transformed = original.replace('import { readFileSync } from "node:fs";',
    'import schemaV2 from "../../../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V2.json";\n' +
    'import compatV1 from "../../../contracts/orotitan-equity/v2/04_SCREENER_SCHEMA_V1_COMPAT_V2.0.10.json";')
    .replace('import { validateV2ProductSemantics, type V2Product } from "./product";',
      'import { validateV2ProductSemantics, type V2Product } from "../v2/product";')
    .replace(/const schema = JSON\.parse\(\s*readFileSync\(new URL\("..\/..\/..\/contracts\/orotitan-equity\/v2\/04_SCREENER_SCHEMA_V2\.json", import\.meta\.url\), "utf8"\),\s*\) as JsonObject;/,
      'const schema = schemaV2 as JsonObject;')
    .replace(/const v1CompatSchema = JSON\.parse\(\s*readFileSync\(new URL\("..\/..\/..\/contracts\/orotitan-equity\/v2\/04_SCREENER_SCHEMA_V1_COMPAT_V2\.0\.10\.json", import\.meta\.url\), "utf8"\),\s*\) as JsonObject;/,
      'const v1CompatSchema = compatV1 as JsonObject;');
  assert.equal(projection, pref + transformed,
    "V2 projection has changed a field, gate or analytical rule outside module loading");
});
