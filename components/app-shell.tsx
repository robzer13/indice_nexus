'use client';

import type { ReactNode } from 'react';
import { useState } from 'react';
import { Sidebar } from '@/components/sidebar';
import { TopBar } from '@/components/top-bar';

export function AppShell({ children }: { children: ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-cockpit-bg">
      <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="lg:pl-64">
        <TopBar onMenuClick={() => setSidebarOpen(true)} />
        <main className="mx-auto w-full max-w-[1600px] px-4 py-6 sm:px-6 sm:py-8 xl:px-8">{children}</main>
        <footer className="mx-auto w-full max-w-[1600px] px-4 pb-8 text-xs text-ink-muted sm:px-6 xl:px-8">
          OroTitan Screener · analyses versionnées, prix séparés, aucun silent overwrite.
        </footer>
      </div>
    </div>
  );
}
