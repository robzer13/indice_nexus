import 'server-only';
import { createClient } from '@supabase/supabase-js';
import { pgrstClockSkewRetryFetch } from './pgrst-clock-retry';

function env(name: 'NEXT_PUBLIC_SUPABASE_URL' | 'SUPABASE_SERVICE_ROLE_KEY'): string {
  const value = process.env[name];
  if (!value) throw new Error(`Missing required environment variable: ${name}`);
  return value;
}

export function createServerSupabaseClient() {
  const url = env('NEXT_PUBLIC_SUPABASE_URL');
  return createClient(url, env('SUPABASE_SERVICE_ROLE_KEY'), {
    global: { fetch: pgrstClockSkewRetryFetch(fetch, url) },
    auth: {
      persistSession: false,
      autoRefreshToken: false,
      detectSessionInUrl: false,
    },
  });
}
