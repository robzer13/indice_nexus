"use server";

import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { requireAdminSession } from "@/lib/auth/admin-session";
import { createServerSupabaseClient } from "@/lib/supabase/server";
import {
  assertProductionUrl, loadServerPilotageDecision,
} from "@/lib/orotitan-equity/v3/pilotage-trusted-loader-server";
import { observeSupabaseEnvironment } from "@/lib/orotitan-equity/v3/pilotage-operator-cockpit-server";
import {
  prepareResearchToFundamentalsStart, ResearchStageStartError,
} from "@/lib/orotitan-equity/v3/pilotage-research-stage-start-v3-1";
import { safeVerifierReason } from "@/lib/orotitan-equity/v3/pilotage-operator-view-v3-1";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

/**
 * Explicit admin action, not a background automation or generic stage writer.
 * Freshly verifies the immutable Research FINAL and uses the Registry's own
 * optimistic concurrency RPC. No bypass on READ/TOKEN/FINAL/CAS failure.
 */
export async function startFundamentalsFromResearch(formData: FormData): Promise<never> {
  await requireAdminSession();
  const rawRunId = formData.get("runId");
  const runId = typeof rawRunId === "string" && UUID.test(rawRunId) ? rawRunId : null;
  if (!runId) redirect("/admin/orotitan?transition=RUN_ID_INVALID");

  const target = "/admin/orotitan?run=" + encodeURIComponent(runId);
  const runVersion = Number(formData.get("expectedRunStateVersion"));
  const researchVersion = Number(formData.get("expectedResearchStageStateVersion"));
  if (!Number.isSafeInteger(runVersion) || runVersion < 1 ||
    !Number.isSafeInteger(researchVersion) || researchVersion < 1) {
    redirect(target + "&transition=INVALID_EXPECTED_VERSIONS");
  }

  let transition = "started";
  try {
    assertProductionUrl();
    const environment = await observeSupabaseEnvironment();
    // This invokes D2's immutable-byte, pin, manifest and seven-output
    // verification before D1 issues an eligible Research FINAL decision.
    const verifiedDecision = await loadServerPilotageDecision({
      runId, expectedRunStateVersion: runVersion,
      expectedCurrentStageStateVersion: researchVersion,
      verifiedEnvironment: environment,
    });
    const plan = prepareResearchToFundamentalsStart({
      runId, expectedRunStateVersion: runVersion,
      expectedResearchStageStateVersion: researchVersion,
      verifiedDecision,
    });
    const { data, error } = await createServerSupabaseClient().rpc(plan.rpc, plan.args);
    if (error) {
      transition = error.code === "40001" ? "STALE_RUN_CAS" :
        "REGISTRY_STAGE_START_REJECTED";
    } else if (!data || typeof data !== "object" || Array.isArray(data) ||
      data.run_id !== runId || data.stage !== "DEEP_DIVE") {
      transition = "REGISTRY_STAGE_START_NOT_CONFIRMED";
    }
  } catch (error) {
    transition = error instanceof ResearchStageStartError ? error.code :
      safeVerifierReason(error);
  }

  if (transition === "started") revalidatePath("/admin/orotitan");
  redirect(target + "&transition=" + encodeURIComponent(transition));
}
