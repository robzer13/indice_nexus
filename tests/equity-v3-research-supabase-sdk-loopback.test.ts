import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer, type IncomingMessage } from "node:http";
import type { AddressInfo } from "node:net";
import test from "node:test";
import { createClient } from "@supabase/supabase-js";
import {
  createPrivateResearchIntakePort,
  createSupabasePrivateResearchBlobPort,
  RESEARCH_PRIVATE_BUCKET,
  type ResearchReceiptTransactionPort,
} from "../lib/orotitan-equity/v3/research-private-quarantine-adapter-v1";
import {
  ResearchIntakeError, submitResearchDraftCandidate,
  type IntakeActor, type ResearchDraftInput, type TrustedSubmissionGrant,
  type PreparedResearchDraft, type UnverifiedReceipt,
} from "../lib/orotitan-equity/v3/research-evidence-intake-candidate-v1";

/**
 * Offline SDK wire tests. Only a loopback HTTP fixture is contacted; this is
 * NOT an actual Supabase Storage deployment or a substitute for hosted RLS.
 * The production @supabase/supabase-js Storage client handles HTTP encoding.
 */
const RUN = "48b9426b-4d0d-4858-9d87-efcd13ac415d";
const ATTEMPT = "b78d8826-9f6b-4d9c-a1b8-ec5d13faac01";
const NOW = Date.parse("2026-10-23T09:00:00.000Z");
const SOURCE = Buffer.from('{"evidence_id":"E001","verified":false}');
const SHA = createHash("sha256").update(SOURCE).digest("hex");
const PATH = ["research-intake/v1", RUN, "RESEARCH", SHA, ATTEMPT + ".json"].join("/");
const actor: IntakeActor = {
  subject: "fixture-oauth-sub",
  clientId: "fixture-registered-client",
  resource: "https://orotitan-chatgpt-mcp-pilot.vercel.app/api/orotitan/mcp",
};
const grant: TrustedSubmissionGrant = {
  grantId: "fixture-server-authorized-grant", runId: RUN, issuerId: "fixture-issuer",
  stage: "RESEARCH", subject: actor.subject, clientId: actor.clientId,
  resource: actor.resource, runStateVersion: 6, stageStateVersion: 4,
  contractPinsSha256: "a".repeat(64), dataCutoff: "2026-10-12",
  issuedAtMs: NOW - 10_000, expiresAtMs: NOW + 60_000,
  maySubmitResearchDraft: true,
};
const input: ResearchDraftInput = {
  runId: RUN, expectedStage: "RESEARCH",
  expectedRunStateVersion: 6, expectedStageStateVersion: 4,
  fileName: "E001.json", mediaType: "application/json",
  sourceDate: "2026-07-23", bytes: SOURCE, claimedSha256: SHA,
};
type Wire = { method: string; path: string; upsert: string | undefined; mime: string | undefined };
type Fixture = {
  objects: Map<string, Buffer>;
  log: Wire[];
  tamperDownload: boolean;
  rejectUpload: boolean;
};

async function collect(req: IncomingMessage): Promise<Buffer> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const b = Buffer.from(chunk);
    size += b.length;
    if (size > 2_000_000) throw new Error("fixture max request exceeded");
    chunks.push(b);
  }
  return Buffer.concat(chunks);
}

async function withOfflineStorage<T>(
  fn: (args: { endpoint: string; fixture: Fixture }) => Promise<T>,
): Promise<T> {
  const fixture: Fixture = {
    objects: new Map(), log: [], tamperDownload: false, rejectUpload: false,
  };
  const server = createServer((req, res) => {
    void (async () => {
      const url = new URL(req.url ?? "/", "http://127.0.0.1");
      const bucket = RESEARCH_PRIVATE_BUCKET;
      const uploadPrefix = "/storage/v1/object/" + bucket + "/";
      const downloadPrefix = "/storage/v1/object/authenticated/" + bucket + "/";
      const path = url.pathname.startsWith(uploadPrefix)
        ? decodeURIComponent(url.pathname.slice(uploadPrefix.length))
        : url.pathname.startsWith(downloadPrefix)
          ? decodeURIComponent(url.pathname.slice(downloadPrefix.length))
          : null;
      const method = req.method ?? "";
      fixture.log.push({
        method, path: url.pathname,
        upsert: req.headers["x-upsert"] as string | undefined,
        mime: req.headers["content-type"] as string | undefined,
      });
      res.setHeader("Content-Type", "application/json");
      if (path === null || path !== PATH || !req.headers.apikey ||
        !req.headers.authorization) {
        res.writeHead(403);
        res.end(JSON.stringify({ statusCode: 403, error: "AccessDenied", message: "forbidden" }));
        return;
      }
      if (method === "POST" && url.pathname.startsWith(uploadPrefix)) {
        const body = await collect(req);
        if (body.indexOf(SOURCE) === -1) {
          res.writeHead(400);
          res.end(JSON.stringify({ statusCode: 400, error: "InvalidBody" }));
        } else if (fixture.rejectUpload) {
          res.writeHead(503);
          res.end(JSON.stringify({ statusCode: 503, error: "Unavailable" }));
        } else if (fixture.objects.has(path)) {
          res.writeHead(409);
          res.end(JSON.stringify({ statusCode: 409, error: "Duplicate", message: "exists" }));
        } else if (req.headers["x-upsert"] === "true") {
          res.writeHead(403);
          res.end(JSON.stringify({ statusCode: 403, error: "UpsertForbidden" }));
        } else {
          fixture.objects.set(path, Buffer.from(SOURCE));
          res.writeHead(200);
          res.end(JSON.stringify({ Key: bucket + "/" + path, Id: "fixture-object" }));
        }
        return;
      }
      if (method === "GET" && (url.pathname.startsWith(downloadPrefix) ||
        url.pathname.startsWith(uploadPrefix))) {
        const data = fixture.objects.get(path);
        if (!data) {
          res.writeHead(404);
          res.end(JSON.stringify({ statusCode: 404, error: "NotFound" }));
        } else {
          res.setHeader("Content-Type", "application/octet-stream");
          res.writeHead(200);
          res.end(fixture.tamperDownload ? Buffer.from("corrupted") : data);
        }
        return;
      }
      res.writeHead(405);
      res.end(JSON.stringify({ statusCode: 405, error: "MethodNotAllowed" }));
    })().catch(() => {
      if (!res.headersSent) res.writeHead(500);
      res.end(JSON.stringify({ statusCode: 500, error: "FixtureError" }));
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolve);
  });
  const addr = server.address() as AddressInfo;
  try {
    return await fn({ endpoint: "http://127.0.0.1:" + addr.port, fixture });
  } finally {
    server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
}

function makeClient(endpoint: string) {
  // Obviously invalid synthetic key: never load, share or generate real secrets.
  return createClient(endpoint, "local-d29b-test-only-nonsecret", {
    auth: { persistSession: false, autoRefreshToken: false, detectSessionInUrl: false },
  });
}
function expectedReceipt(d: PreparedResearchDraft): UnverifiedReceipt {
  return {
    receiptId: "fixture-unverified-receipt", runId: d.runId,
    issuerId: d.issuerId, stage: "RESEARCH",
    runStateVersion: d.runStateVersion,
    stageStateVersion: d.stageStateVersion,
    sha256: d.sha256, fileName: d.fileName, sizeBytes: d.sizeBytes,
    submittedBySubject: d.submittedBySubject,
    status: "RECEIVED_NOT_VERIFIED", authority: "QUARANTINE_ONLY",
    duplicate: false, admitted: false, mayAdvance: false, mayPublish: false,
  };
}
function transactions(): {
  receiptCount: number;
  port: ResearchReceiptTransactionPort;
} {
  const record = { receiptCount: 0 };
  const port: ResearchReceiptTransactionPort = {
    async checkBeforeUpload(d, g) {
      assert.equal(d.runId, g.runId);
      assert.equal(d.stageStateVersion, g.stageStateVersion);
      assert.equal(d.runStateVersion, g.runStateVersion);
    },
    async commitReceiptAndAudit({ draft, stored }) {
      assert.equal(stored.objectPath, PATH);
      assert.equal(stored.sha256, SHA);
      assert.equal(stored.sizeBytes, SOURCE.byteLength);
      record.receiptCount++;
      return { receipt: expectedReceipt(draft), authoritativeObjectPath: PATH };
    },
    async readVerifierMetadata() { return null; },
  };
  return { get receiptCount() { return record.receiptCount; }, port };
}

test("real supabase-js SDK over loopback: immutable POST + authenticated GET", async () => {
  await withOfflineStorage(async ({endpoint, fixture}) => {
    const blob = createSupabasePrivateResearchBlobPort(makeClient(endpoint));
    await blob.putImmutable(PATH, SOURCE, "application/json");
    assert.deepEqual(Buffer.from(await blob.readExact(PATH)), SOURCE);
    assert.equal(fixture.objects.size, 1);
    assert.equal(fixture.log.length, 2);
    assert.match(fixture.log[0].path, /\/storage\/v1\/object\/orotitan-research-evidence-quarantine-v1\//);
    assert.equal(fixture.log[0].method, "POST");
    assert.notEqual(fixture.log[0].upsert, "true");
    assert.equal(fixture.log[1].method, "GET");
    assert.match(fixture.log[1].path,
      /\/storage\/v1\/object\/(?:authenticated\/)?orotitan-research-evidence-quarantine-v1\//);
    assert.equal(fixture.log.some(x => /list|sign|public/.test(x.path)), false);
  });
});

test("loopback server rejects duplicate, missing bytes and bad storage routes", async () => {
  await withOfflineStorage(async ({endpoint, fixture}) => {
    const blob = createSupabasePrivateResearchBlobPort(makeClient(endpoint));
    await assert.rejects(() => blob.readExact(PATH),
      (e: unknown) => e instanceof ResearchIntakeError &&
        e.code === "INTAKE_PRIVATE_REREAD_FAILED");
    await blob.putImmutable(PATH, SOURCE, "application/json");
    await assert.rejects(() => blob.putImmutable(PATH, SOURCE, "application/json"),
      (e: unknown) => e instanceof ResearchIntakeError &&
        e.code === "INTAKE_PRIVATE_UPLOAD_FAILED");
    assert.deepEqual(fixture.objects.get(PATH), SOURCE);
    await assert.rejects(() => blob.putImmutable("../" + PATH, SOURCE, "application/json"),
      (e: unknown) => e instanceof ResearchIntakeError &&
        e.code === "INTAKE_PRIVATE_PATH_INVALID");
    assert.equal(fixture.log.length, 3);
  });
});

test("full candidate -> real SDK wire -> readback -> unverified receipt only", async () => {
  await withOfflineStorage(async ({endpoint, fixture}) => {
    const tx = transactions();
    const p = createPrivateResearchIntakePort({
      grants: {
        async resolveSubmissionGrant(a, runId) {
          return a.subject === actor.subject && a.clientId === actor.clientId &&
            a.resource === actor.resource && runId === RUN ? grant : null;
        },
        async resolveVerifierGrant() { return null; },
      },
      blobs: createSupabasePrivateResearchBlobPort(makeClient(endpoint)),
      receipts: tx.port,
      newAttemptId: () => ATTEMPT,
    });
    const receipt = await submitResearchDraftCandidate(input, actor, p, NOW);
    assert.equal(receipt.status, "RECEIVED_NOT_VERIFIED");
    assert.equal(receipt.authority, "QUARANTINE_ONLY");
    assert.equal(receipt.admitted, false);
    assert.equal(receipt.mayAdvance, false);
    assert.equal(tx.receiptCount, 1);
    assert.equal(fixture.objects.size, 1);
  });
});

test("corrupt downloaded bytes and HTTP upload failures cannot yield a receipt", async () => {
  for (const kind of ["corrupt", "503"] as const) {
    await withOfflineStorage(async ({endpoint, fixture}) => {
      fixture.tamperDownload = kind === "corrupt";
      fixture.rejectUpload = kind === "503";
      const tx = transactions();
      const p = createPrivateResearchIntakePort({
        grants: {
          async resolveSubmissionGrant() { return grant; },
          async resolveVerifierGrant() { return null; },
        },
        blobs: createSupabasePrivateResearchBlobPort(makeClient(endpoint)),
        receipts: tx.port,
        newAttemptId: () => ATTEMPT,
      });
      await assert.rejects(
        () => submitResearchDraftCandidate(input, actor, p, NOW),
        (e: unknown) => e instanceof ResearchIntakeError &&
          e.code === (kind === "corrupt"
            ? "INTAKE_PRIVATE_STORED_HASH_MISMATCH"
            : "INTAKE_PRIVATE_UPLOAD_FAILED"),
      );
      assert.equal(tx.receiptCount, 0);
      assert.equal(fixture.objects.size, kind === "corrupt" ? 1 : 0);
    });
  }
});
