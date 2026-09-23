import {
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_SPEC,
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V03_MODULE_ID,
  GATE18_PHASE_B_V03_MODULE_QUESTION,
  GATE18_PHASE_B_V03_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V03_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V03_SCOPE,
  GATE18_PHASE_B_V03_SYSTEM_PROMPT,
  assertGate18PhaseBV03Semantics,
  buildGate18PhaseBV03ModelInput,
  buildVerifiedGate18V03MoatPacket,
  gate18PhaseBV03GenerationSchemaSha256,
  gate18PhaseBV03OutputSchema,
  gate18PhaseBV03PromptTemplateSha256,
  type Gate18PhaseBV03Output,
} from "./model-calibration-pilot-v03";

export const GATE18_PHASE_B_PROTOCOL_VERSION = "0.4" as const;

/**
 * Common transport envelope for every Gate 18 candidate under protocol v0.4.
 *
 * Why 4096:
 * - protocol v0.3 used 1536;
 * - the first SOL/high company attempt consumed all 1536 tokens as reasoning,
 *   emitted zero visible text, and finished for length;
 * - the structured visible-output schema itself remains compact and unchanged.
 *
 * This is a transport/execution correction only. It does not change the
 * evidence packet, prompt, schema, analytical methodology, candidate reasoning
 * profile, or publication authority.
 *
 * Gate 18 section 7 requires the max output contract to remain identical across
 * candidates, so this value MUST NOT vary by model.
 */
export const GATE18_PHASE_B_V04_MAX_OUTPUT_TOKENS = 4096;

export {
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_ID,
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_SPEC,
  GATE18_PHASE_B_V03_GENERATION_SCHEMA_VERSION,
  GATE18_PHASE_B_V03_MODULE_ID,
  GATE18_PHASE_B_V03_MODULE_QUESTION,
  GATE18_PHASE_B_V03_PROMPT_TEMPLATE_ID,
  GATE18_PHASE_B_V03_PROMPT_TEMPLATE_VERSION,
  GATE18_PHASE_B_V03_SCOPE,
  GATE18_PHASE_B_V03_SYSTEM_PROMPT,
  assertGate18PhaseBV03Semantics,
  buildGate18PhaseBV03ModelInput,
  buildVerifiedGate18V03MoatPacket,
  gate18PhaseBV03GenerationSchemaSha256,
  gate18PhaseBV03OutputSchema,
  gate18PhaseBV03PromptTemplateSha256,
};

export type { Gate18PhaseBV03Output };
