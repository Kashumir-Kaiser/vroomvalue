import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "VroomValue — Used Vehicle Price Estimate",
  description: "Estimate used-vehicle selling price with an 80% calibrated range and model support checks.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>
    <nav className="topnav" aria-label="Primary">
      <Link href="/">Estimate</Link><Link href="/feedback">Feedback</Link><Link href="/admin">Admin</Link>
    </nav>
    {children}
  </body></html>;
}