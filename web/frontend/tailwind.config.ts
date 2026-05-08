import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: "#0b1220",
          800: "#0f1830",
          700: "#1a2547",
          600: "#243259",
          500: "#34416d",
        },
        brand: {
          50:  "#eef4ff",
          100: "#dbe6ff",
          200: "#b9cdff",
          300: "#8babff",
          400: "#5d83f6",
          500: "#3f60e6",
          600: "#2c44c4",
          700: "#26389e",
          800: "#1e3a5f",
          900: "#162a45",
        },
        accent: {
          cyan: "#22d3ee",
          violet: "#a78bfa",
          amber: "#f59e0b",
          rose: "#fb7185",
          emerald: "#34d399",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      backgroundImage: {
        "grid-soft":
          "linear-gradient(to right, rgba(255,255,255,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.06) 1px, transparent 1px)",
      },
      boxShadow: {
        glow: "0 10px 40px -10px rgba(63,96,230,0.4)",
        cardlift: "0 12px 30px -10px rgba(15,24,48,0.25)",
      },
      animation: {
        "fade-in": "fadeIn 600ms ease-out",
        "rise-in": "riseIn 700ms cubic-bezier(0.22,1,0.36,1)",
      },
      keyframes: {
        fadeIn: { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
        riseIn: {
          "0%": { opacity: "0", transform: "translateY(12px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
} satisfies Config;
