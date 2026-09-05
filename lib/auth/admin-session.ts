import 'server-only';
import { cookies } from 'next/headers';
import { createHash, createHmac, timingSafeEqual } from 'node:crypto';

const COOKIE_NAME = 'orotitan_admin_session';
const SESSION_SECONDS = 8 * 60 * 60;

type SessionPayload = { v: 1; iat: number; exp: number };

function required(name: 'ADMIN_PASSWORD' | 'ADMIN_SESSION_SECRET'): string {
  const value = process.env[name];
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

function safeEqual(a: string, b: string): boolean {
  const left = createHash('sha256').update(a).digest();
  const right = createHash('sha256').update(b).digest();
  return timingSafeEqual(left, right);
}

function sign(encodedPayload: string): string {
  const secret = required('ADMIN_SESSION_SECRET');
  if (secret.length < 32) throw new Error('ADMIN_SESSION_SECRET must be at least 32 characters');
  return createHmac('sha256', secret).update(encodedPayload).digest('base64url');
}

function makeToken(payload: SessionPayload): string {
  const encoded = Buffer.from(JSON.stringify(payload)).toString('base64url');
  return `${encoded}.${sign(encoded)}`;
}

function verifyToken(token: string): boolean {
  const [encoded, signature] = token.split('.');
  if (!encoded || !signature) return false;
  const expected = sign(encoded);
  if (!safeEqual(signature, expected)) return false;
  try {
    const payload = JSON.parse(Buffer.from(encoded, 'base64url').toString('utf8')) as SessionPayload;
    return payload.v === 1 && Number.isFinite(payload.exp) && payload.exp > Date.now();
  } catch {
    return false;
  }
}

export function verifyAdminPassword(candidate: string): boolean {
  return safeEqual(candidate, required('ADMIN_PASSWORD'));
}

export async function createAdminSession(): Promise<void> {
  const now = Date.now();
  const token = makeToken({ v: 1, iat: now, exp: now + SESSION_SECONDS * 1000 });
  const store = await cookies();
  store.set(COOKIE_NAME, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge: SESSION_SECONDS,
  });
}

export async function clearAdminSession(): Promise<void> {
  const store = await cookies();
  store.delete(COOKIE_NAME);
}

export async function isAdminAuthenticated(): Promise<boolean> {
  const store = await cookies();
  const token = store.get(COOKIE_NAME)?.value;
  return token ? verifyToken(token) : false;
}

export async function requireAdminSession(): Promise<void> {
  if (!(await isAdminAuthenticated())) throw new Error('UNAUTHORIZED');
}
