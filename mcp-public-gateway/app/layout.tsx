import type { ReactNode } from "react";
export const metadata = { title: "OroTitan — MCP test gateway", robots: { index: false, follow: false } };
export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="fr"><body style={{fontFamily:"system-ui,sans-serif",background:"#081124",color:"#e8efff",padding:"3rem",margin:0}}>{children}</body></html>;
}
