/** Reject malformed or oversized Content-Length before processing MCP POST requests.
 * A missing header is valid for chunked/streaming HTTP transports.
 */
export function isAllowedMcpContentLength(value: string | null): boolean {
  return value === null || (/^\d+$/.test(value) && Number(value) <= 250_000);
}
