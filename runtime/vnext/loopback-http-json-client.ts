import http from "node:http";

export interface LoopbackJsonRequestOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: string;
  timeoutMs: number;
  maxResponseBytes?: number;
}

export class LoopbackHttpStatusError extends Error {
  readonly statusCode: number;
  readonly responseBody: string;

  constructor(statusCode: number, responseBody: string) {
    super(
      [
        `LOOPBACK_HTTP_STATUS_${statusCode}`,
        responseBody.length > 0 ? responseBody : "NO_RESPONSE_BODY",
      ].join(":"),
    );
    this.name = "LoopbackHttpStatusError";
    this.statusCode = statusCode;
    this.responseBody = responseBody;
  }
}

export class LoopbackHttpExplicitTimeoutError extends Error {
  readonly timeoutMs: number;

  constructor(timeoutMs: number) {
    super(`LOOPBACK_HTTP_EXPLICIT_TIMEOUT_${timeoutMs}MS`);
    this.name = "LoopbackHttpExplicitTimeoutError";
    this.timeoutMs = timeoutMs;
  }
}

function assertLoopbackUrl(rawUrl: string): URL {
  const url = new URL(rawUrl);

  if (
    url.protocol !== "http:" ||
    (url.hostname !== "127.0.0.1" && url.hostname !== "localhost")
  ) {
    throw new Error("LOOPBACK_HTTP_NON_LOOPBACK_ENDPOINT_FORBIDDEN");
  }

  return url;
}

export async function requestLoopbackJson<T>(
  rawUrl: string,
  options: LoopbackJsonRequestOptions,
): Promise<T> {
  const url = assertLoopbackUrl(rawUrl);
  const maxResponseBytes = options.maxResponseBytes ?? 32 * 1024 * 1024;

  if (!Number.isInteger(options.timeoutMs) || options.timeoutMs <= 0) {
    throw new Error("LOOPBACK_HTTP_INVALID_TIMEOUT");
  }

  return await new Promise<T>((resolve, reject) => {
    const controller = new AbortController();
    let explicitlyTimedOut = false;
    let responseBytes = 0;
    const chunks: Buffer[] = [];

    const timer = setTimeout(() => {
      explicitlyTimedOut = true;
      controller.abort();
    }, options.timeoutMs);

    const finish = (fn: () => void): void => {
      clearTimeout(timer);
      fn();
    };

    const request = http.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        method: options.method ?? "GET",
        headers: options.headers,
        signal: controller.signal,
      },
      (response) => {
        response.on("data", (chunk: Buffer | string) => {
          const buffer = Buffer.isBuffer(chunk)
            ? chunk
            : Buffer.from(chunk);
          responseBytes += buffer.length;

          if (responseBytes > maxResponseBytes) {
            request.destroy(
              new Error("LOOPBACK_HTTP_RESPONSE_TOO_LARGE"),
            );
            return;
          }

          chunks.push(buffer);
        });

        response.on("error", (error) => {
          finish(() => reject(error));
        });

        response.on("end", () => {
          const responseBody = Buffer.concat(chunks).toString("utf8");
          const statusCode = response.statusCode ?? 0;

          if (statusCode < 200 || statusCode >= 300) {
            finish(() =>
              reject(
                new LoopbackHttpStatusError(
                  statusCode,
                  responseBody
                    .replace(/[\r\n\t]+/g, " ")
                    .trim()
                    .slice(0, 2000),
                ),
              ),
            );
            return;
          }

          try {
            const parsed = JSON.parse(responseBody) as T;
            finish(() => resolve(parsed));
          } catch (error) {
            finish(() =>
              reject(
                new Error(
                  `LOOPBACK_HTTP_JSON_PARSE_FAILED:${
                    error instanceof Error
                      ? error.message
                      : String(error)
                  }`,
                ),
              ),
            );
          }
        });
      },
    );

    request.on("error", (error) => {
      if (explicitlyTimedOut) {
        finish(() =>
          reject(
            new LoopbackHttpExplicitTimeoutError(
              options.timeoutMs,
            ),
          ),
        );
        return;
      }

      finish(() => reject(error));
    });

    if (options.body !== undefined) {
      request.write(options.body);
    }

    request.end();
  });
}
