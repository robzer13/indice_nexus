# Historical buyback observation audit — public proposal

**Status: NON-AUTHORITATIVE, NON-ADMITTED.** No methodology or Registry authority.

The generic TypeScript helper checks self-reported historical buyback notice dates,
trade dates, quantities, duplicate keys and arithmetic against DATA_CUTOFF.
An HTTPS locator is only syntactic metadata: the helper does not retrieve,
hash or attest original source bytes. Identifier uniqueness does not prove
real-world transaction deduplication across separate source documents.

Trade-date buybacks do not establish settlement/economic-effect dates,
exhaustive share movement classes or the issuer's externally outstanding shares.
Therefore every successful result explicitly returns economicShareCount=UNKNOWN
and denominatorValuationAdmission=NO. It cannot create a Valuation Lock.

The governing V3 economic share-count methodology remains unchanged at
contracts/orotitan-equity/v3/01_ANALYSIS_STANDARD_V1_1_ECONOMIC_SHARE_COUNT_PATCH.md.
Any later V3.1 valuation-date alignment requires a separately pinned
successor contract set; do not rebind historical V3 runs.

No private run IDs, company-specific workpapers, issuer figures or private
primary source artifacts are included in this public repository change.
Tests use synthetic quantities and issuer.example.org only.

Run via the existing npm test script (tsx --test tests/*.test.ts).
