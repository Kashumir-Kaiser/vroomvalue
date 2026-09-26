"use client";

import { useState } from "react";

type Theme = "light" | "dark";

function systemTheme(): Theme {
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export default function ThemeToggle({
  initialTheme,
}: {
  initialTheme: Theme | null;
}) {
  const [theme, setTheme] = useState<Theme | null>(initialTheme);

  function toggle() {
    const current = theme ?? systemTheme();
    const next: Theme = current === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.dataset.theme = next;

    try {
      document.cookie =
        `vroomvalue_theme=${next}; Path=/; Max-Age=31536000; SameSite=Lax`;
    } catch {
      // The in-page theme still works if browser cookie storage is unavailable.
    }
  }

  const label =
    theme === "dark" ? "Light" : theme === "light" ? "Dark" : "Theme";

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      aria-label="Toggle color theme"
      title="Toggle color theme"
    >
      {label}
    </button>
  );
}
