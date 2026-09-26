import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";

import ThemeToggle from "../components/ThemeToggle";

export const metadata: Metadata = {
  title: "VroomValue — Used Vehicle Price Estimate",
  description:
    "Estimate used-vehicle selling price with an 80% calibrated range and model support checks.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <nav className="topnav" aria-label="Primary">
          <Link href="/">Estimate</Link>
          <Link href="/feedback">Feedback</Link>
          <Link href="/admin">Admin</Link>
          <span className="nav-actions">
            <ThemeToggle />
            <a
              className="github-link"
              href="https://github.com/Kashumir-Kaiser"
              target="_blank"
              rel="noreferrer"
              aria-label="Kashumir-Kaiser on GitHub"
              title="Kashumir-Kaiser on GitHub"
            >
              <svg
                viewBox="0 0 24 24"
                width="20"
                height="20"
                aria-hidden="true"
                focusable="false"
              >
                <path
                  fill="currentColor"
                  d="M12 .7a11.5 11.5 0 0 0-3.64 22.4c.58.1.79-.25.79-.56v-2.2c-3.22.7-3.9-1.37-3.9-1.37-.53-1.34-1.29-1.7-1.29-1.7-1.05-.72.08-.71.08-.71 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.57-.29-5.27-1.29-5.27-5.73 0-1.27.45-2.3 1.19-3.11-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.16 1.19a10.9 10.9 0 0 1 5.76 0c2.2-1.5 3.16-1.19 3.16-1.19.63 1.6.23 2.78.11 3.07.74.81 1.19 1.84 1.19 3.11 0 4.45-2.71 5.43-5.29 5.72.42.36.79 1.07.79 2.16v3.2c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .7Z"
                />
              </svg>
            </a>
          </span>
        </nav>
        {children}
      </body>
    </html>
  );
}
