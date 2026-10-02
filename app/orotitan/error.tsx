'use client';

import { LoadErrorState } from '@/components/orotitan/ui';

export default function OroTitanError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return <LoadErrorState onRetry={reset} />;
}
