import {
  activeContractPinPack,
  activeContractSetComputedSha256,
  assertRuntimeBootstrapIntegrity,
  runtimeBootstrap,
  runtimeBootstrapCanonicalSha256,
} from "../../../../lib/orotitan-equity/runtime";

export const dynamic = "force-dynamic";

export function GET(): Response {
  try {
    assertRuntimeBootstrapIntegrity();
    return Response.json(
      {
        service: "OROTITAN_EQUITY_RESEARCH",
        runtime: "OROTITAN_RUNTIME_BOOTSTRAP_V3.0.1",
        runtime_bootstrap_sha256: runtimeBootstrapCanonicalSha256,
        production_status: runtimeBootstrap.production_status,
        active_for_new_runs: true,
        new_run_admission: "ADMISSIBLE",
        active_contract_set_sha256: activeContractPinPack.contract_set_sha256,
        computed_contract_set_sha256: activeContractSetComputedSha256,
        production_project_ref: runtimeBootstrap.production_environment.project_ref,
        v2_existing_run_compatibility: "PRESERVED",
        historical_run_rebinding: false,
        publication_authorized: false,
      },
      { headers: { "cache-control": "no-store" } },
    );
  } catch (error) {
    return Response.json(
      {
        service: "OROTITAN_EQUITY_RESEARCH",
        production_status: "BLOCKED",
        new_run_admission: "BLOCKED",
        error: error instanceof Error ? error.message : "UNKNOWN_RUNTIME_ERROR",
      },
      { status: 503, headers: { "cache-control": "no-store" } },
    );
  }
}
