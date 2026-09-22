import assert from "node:assert/strict";
import test from "node:test";

import {
  VNEXT_SHADOW_SUPABASE_URL,
  assertVNextShadowSupabaseUrl,
} from "../runtime/vnext/environment";

test("VNext accepts only the dedicated shadow Supabase project", () => {
  assert.equal(
    assertVNextShadowSupabaseUrl(VNEXT_SHADOW_SUPABASE_URL),
    VNEXT_SHADOW_SUPABASE_URL,
  );
});

test("VNext rejects OroTitan production Supabase", () => {
  assert.throws(
    () =>
      assertVNextShadowSupabaseUrl(
        "https://cugpgtzygqqlxetyeven.supabase.co",
      ),
    /production Supabase is forbidden/,
  );
});

test("VNext rejects any other Supabase project", () => {
  assert.throws(
    () =>
      assertVNextShadowSupabaseUrl(
        "https://aaaaaaaaaaaaaaaaaaaa.supabase.co",
      ),
    /expected awgsurdyvsyolcgpnygh\.supabase\.co/,
  );
});
