# Method-V2 runtime serialization profiles

These inactive profiles define physical persistence outside the frozen analytical
Authority Manifest and Contract Set. Historical artifacts are not rewritten.

`OROTITAN_METHOD_V2_CERTIFICATION_PERSISTENCE_PROFILE_V1.0.json` retains the mission's
property names. Its dedicated `method_v2_challenge_binding` envelope serializes
exact current Question Ledger, Challenge Report and lock refs and the concern
tuples returned by `admitMethodV2Certification`. It does not map those tuples into
V1 `certification.material_limitations[]` / `issueRef`. The inherited canonical
Certification projection remains separately governed.

`lib/orotitan-equity/method-v2/certification-persistence.ts` provides offline
parse, deterministic serialize and exact binding verification functions. The
serializer sorts object keys and concern IDs, retains other array order and string
values, and emits compact UTF-8 JSON without a BOM or final newline. The parser
accepts alternate valid whitespace/order; these remain distinct raw-byte hashes.
Duplicate decoded JSON member names are rejected in every object before
`JSON.parse`, including escaped spellings and inherited content. This is a
structural ambiguity check, not an inherited Certification judgment. UUID shape,
lowercase hashes, positive safe integers and real date checks follow existing
Challenge/Registry representation conventions; comparisons never normalize.

The verifier consumes a trusted context containing run ID, revision, cutoff, all
four exact refs, status and `challenge_limitations` taken unchanged from the
validated Challenge `limitations` output. Concern order has no semantic meaning.
The upstream consumer must establish current persisted lineage and valid Challenge
admission; neither parser nor verifier decides Certification or score permission.
Additional inherited content may change whole-artifact bytes, never the identity
of this Challenge binding. Runtime identity is format/version under a separately
pinned Method-V2 profile; it does not independently classify or activate a run.

The profile was independently reviewed and merged into `vnext` by PR #372 at
`cf83831d13b87bc892aef59e2951c7c4518119e8`. Resumed PR #371 pins the exact profile
raw-byte SHA-256 `a9b3eff1930a9a7e6164cfb1f0248c98c8041975ef9adc77f9d152062df0a53c`.
The recomputed runtime binding raw-byte SHA-256 is
`b078df1a792aa6dc46b913457406e9d48d6dd79cca55d8951d8f14218eb384d0`, replacing
`0832d3c90afab1e4e044d0b84af3992edcb2961298b379890cd7d978d391ca2c`.
The runtime consumer verifies authoritative persisted Certification bytes and exact
Challenge context before creating its owner-only proof. No publication or
activation is implemented.
