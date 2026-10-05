# Method-V2 analytical authority closure

MISSION = METHOD-V2-AUTHORITY-CLOSURE-001
BASELINE = vnext@bd6731767bf07d7c1acaf68dcb96880a7b17dccd
STATUS = FROZEN
PRODUCTION_ACTIVE = NO

## Analytical identity

`OROTITAN_METHOD_V2_AUTHORITY_MANIFEST.json` defines exactly:

V1 BASE + EXPLICIT SCOPED SUCCESSORS = METHOD_V2.

The seven inherited members preserve the Analysis Standard, Master Prompt,
Investment Policy, execution-policy patch, Research/Deep Dive analytical contracts
and I2 canonical computation rules. The five previously frozen successors are the
Economic Share Count authority and prompt patch, DCF Timing, and Valuation Date
Alignment authority and prompt patch. The final new analytical authority is
Pre-Certification Challenge; its two schemas are normative artifact dependencies.
Total: 15 manifest members, including two schemas. No new scoring authority exists.

Authority precedence is scoped, never general last-file-wins. Date Alignment
supersedes only the contradictory date clauses identified in its Section 10.
Economic Share Count still governs denominator definition and bounds; DCF Timing
still governs temporal mechanics. The Challenge adds admission before Certification,
not a replacement for Certification, its formulas or the V1 terminal conjunction.
All other V1 clauses remain controlling. An unresolved cross-scope conflict fails
closed and requires project authority; runtime labels cannot resolve it.

The two Date Alignment files were absent from the baseline and are brought forward
byte-for-byte from commits `7422937e76449bfe32be8fb16433a458194406d2` and
`6b229fdbe018ba67e65ad0abbfee0ca3065aa0d1` in this repository. They are not newly
designed or edited here. Existing V1/V2/V3 files are unchanged.

## Immutable bytes and recomputation

Every member includes its repository, source commit, path, Git blob SHA,
canonical-content SHA-256 and stored-file SHA-256. Multipart packaging is listed
with exact part hashes. The existing V1 resolver verifies decompressed canonical
bytes and all archive parts; compression never changes analytical identity.

The set digest is SHA-256 of canonical UTF-8 JSON of the complete manifest with
only `authority_set_sha256` removed: recursively sort object keys by Unicode code
point, retain array order, serialize with no whitespace or final newline, and
allow integer numbers only. All current keys are ASCII. Thus membership, roles,
scopes, precedence, exclusions and immutable locators are bound together. This
digest is a separate analytical authority identity, NOT `contract_set_sha256`.
The latter retains its historical execution meaning and computation unchanged.
The exact manifest file also has an independently reportable raw-byte SHA-256.

New frozen members were committed before the manifest to give them exact immutable
source locators without a self-referential commit hash. The PR's candidate HEAD
locates the manifest itself. Verification checks current files against the pinned
bytes; a remote resolver must honor the recorded commit and repository.

Run `npx tsx --test tests/method-v2-authority.test.ts` for the offline manifest and
admission matrix. The trusted expected set digest is pinned separately in
`lib/orotitan-equity/method-v2/authority.ts`; a modified manifest cannot validate
itself by supplying its own freshly recomputed digest.

## Execution support is a different identity domain

`OROTITAN_METHOD_V2_EXECUTION_SUPPORT.json` records the five successor overlays
and their immutable hashes plus the exact inherited V1 execution references.
It is outside the analytical authority set. Its overlays make execution-process,
Pilotage, Deep Dive, Master Prompt and Integration sequencing explicit without
silently adopting unrelated historical V2/V3 runtime changes.

Research methodology remains unchanged. Integration adds only verification of
Challenge/Certification lineage; it gains no analytical powers. Registry stage
codes stay RESEARCH, DEEP_DIVE, INTEGRATION. No active pin pack, bootstrap, schema,
Registry function, publication path or production selector is changed.

## Classification and admission boundary

Runs existing before a separately authorized Method-V2 activation resolve METHOD_V1
from their persisted origin, even with historical runtime V2/V3 or schema 2.0.0.
The classification helper rejects attempts to attach Method-V2 authority to such
a historical run. A new Method-V2 claim requires explicit METHOD_V2 and the exact
approved analytical set identity, with full member-byte verification before use.
An absent new-run generation, mixed Method-V1/new authority, unknown generation,
or missing/wrong set identity fails closed. Existing Method-V1 compatibility is
not broadened by this helper. Future METHOD_V3 requires separate approved authority.

These are offline reference validators, not an implemented database binding or
an activation service. Their inputs must come from trusted persisted provenance,
not arbitrary client declarations. The consuming resolver must establish that
the lock/report/ledger references are the current authorized ones, resolve exact
artifact bytes and validate upstream admission, including Valuation Artifact and
Evidence/Conflict/Calculation/Assumption lineage. A digest alone is not admission.

The Challenge validator then verifies run, company, cutoff, exact lock and ledger
references, aggregate/count reconciliation, coverage, independent saturation and
concern propagation. `material_concerns` and `company_specific_questions` contain
question IDs; limitations returned for Certification retain ID, decision impact
and mitigation. Final-pass `conclusion_scope_sha256` covers canonical JSON of
run_id, data_cutoff, fundamentals_lock_ref, valuation_lock_ref and question_ledger_ref.
Hashing the ledger prevents reuse of saturation evidence for a different question
set. Evidence dates must not exceed cutoff. Hypothetical future scenarios are
not represented as future factual evidence dates.

DELTA must consume verified prior passing report/ledger references, a documented
impact map and revalidated inherited coverage. Its effective ledger includes all
carried-forward questions/concerns; it cannot omit prior limitations to obtain PASS.
The current pure validator checks supplied provenance and complete current coverage;
the upstream trusted resolver owns proof of prior passing status and carry-forward
completeness. No production enforcement is claimed by these offline tests.

## Freeze versus activation

The approved Challenge correction is sufficient breadth plus material saturation,
with question count descriptive only. The old draft's numerical gate is not carried
forward. The draft branch/PR remains historical design provenance.

No historical snapshot payload, run authority, score, formula, cap, Elite threshold,
Weak Link treatment, policy or terminal conjunction changes. No database migration,
live run, publication, pointer update or production operation occurs. Production
use requires separately authorized Registry/runtime implementation, validation and
activation; this PR does not satisfy those steps. Do not restart V2-11 from this
freeze alone. Changes to frozen membership or semantics require an approved
successor version rather than silent rebinding.
