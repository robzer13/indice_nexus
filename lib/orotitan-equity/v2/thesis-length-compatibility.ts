import { validateV2ProductSemantics } from "./product";

/**
 * V2.0.11 additive thesis-length compatibility.
 *
 * The frozen V2 contract enforces 240 in its original validator. This
 * *opt-in* validation extension only removes the max-length error for
 * non-empty exact certified thesis strings; everything else delegates
 * unchanged to the original V2 product validator.
 *
 * No truncation, rewriting, numeric mutation or analyst inference.
 */
const ALLOWED_FIELDS = ["quality_case", "valuation_case", "key_risk"] as const;

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validateV2ProductWithLongCertifiedThesis(value: unknown): string[] {
  const errors = validateV2ProductSemantics(value);
  if (!isObject(value) || !isObject(value.investment_thesis)) return errors;
  const thesis = value.investment_thesis;

  return errors.filter((error) => {
    for (const field of ALLOWED_FIELDS) {
      if (error === `investment_thesis.${field} must contain 1-240 characters`) {
        const original = thesis[field];
        // Only the old maximum may be waived. Required, non-blank and string
        // invariants must remain in place. Unicode is stored verbatim.
        return typeof original !== "string" || original.trim().length === 0;
      }
    }
    return true;
  });
}

/** UI preview only: always keep the certified source intact. */
export function compactThesisForDisplay(fullText: string, visibleChars = 650): {
  fullText: string;
  preview: string;
  truncated: boolean;
} {
  if (!Number.isSafeInteger(visibleChars) || visibleChars < 1) {
    throw new Error("visibleChars must be a positive safe integer");
  }
  const points = Array.from(fullText);
  if (points.length <= visibleChars) return { fullText, preview: fullText, truncated: false };
  return { fullText, preview: points.slice(0, visibleChars).join("") + "…", truncated: true };
}
