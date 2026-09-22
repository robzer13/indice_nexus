export const VNEXT_SHADOW_PROJECT_REF = "awgsurdyvsyolcgpnygh" as const;
export const VNEXT_SHADOW_SUPABASE_URL =
  "https://awgsurdyvsyolcgpnygh.supabase.co" as const;

const PRODUCTION_PROJECT_REF = "cugpgtzygqqlxetyeven" as const;

export function assertVNextShadowSupabaseUrl(value: string): string {
  const url = new URL(value);

  if (url.protocol !== "https:") {
    throw new Error("VNEXT_WRONG_ENVIRONMENT: Supabase URL must use HTTPS");
  }

  if (url.hostname === `${PRODUCTION_PROJECT_REF}.supabase.co`) {
    throw new Error("VNEXT_WRONG_ENVIRONMENT: production Supabase is forbidden");
  }

  if (url.hostname !== `${VNEXT_SHADOW_PROJECT_REF}.supabase.co`) {
    throw new Error(
      `VNEXT_WRONG_ENVIRONMENT: expected ${VNEXT_SHADOW_PROJECT_REF}.supabase.co`,
    );
  }

  return url.origin;
}
