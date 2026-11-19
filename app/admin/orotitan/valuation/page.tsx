import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { isAdminAuthenticated } from "@/lib/auth/admin-session";
import { OperationalValuationClient } from "./operational-valuation-client";

export const metadata: Metadata = { title: "OroTitan — valorisation de travail" };
export const dynamic = "force-dynamic";
export default async function OperationalValuationPage() {
  if (!(await isAdminAuthenticated())) redirect("/admin");
  return <OperationalValuationClient />;
}
