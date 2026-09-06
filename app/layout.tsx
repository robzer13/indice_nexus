import type { Metadata } from 'next';
import { AppShell } from '@/components/app-shell';
import './globals.css';

export const metadata: Metadata = {
  title: { default: 'OroTitan Screener', template: '%s · OroTitan' },
  description: "Cockpit opérationnel d'une base fermée de sociétés déjà analysées.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="fr"><body><AppShell>{children}</AppShell></body></html>;
}
