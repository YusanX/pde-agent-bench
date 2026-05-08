import { Github, Database, BookOpen, Mail } from "lucide-react";

export default function Footer() {
  return (
    <footer className="bg-ink-900 text-white/70 py-10 border-t border-white/5">
      <div className="container-page">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <span className="w-9 h-9 rounded-xl bg-gradient-to-br from-brand-600 to-accent-cyan grid place-items-center text-white font-serif italic text-xl">
              ∂
            </span>
            <span className="font-bold text-white">PDEAgent-Bench</span>
          </div>

          <nav className="flex flex-wrap items-center justify-center gap-1 text-sm">
            <FooterLink href="#leaderboard">Leaderboard</FooterLink>
            <FooterLink href="#explorer">Explorer</FooterLink>
            <FooterLink href="#findings">Findings</FooterLink>
            <FooterLink href="#getting-started">Get Started</FooterLink>
          </nav>

          <div className="flex items-center gap-2">
            <IconLink href="https://github.com/YusanX/pde-agent-bench" Icon={Github} label="GitHub" />
            <IconLink
              href="https://huggingface.co/datasets/eclipse00/PDEAgent-Bench"
              Icon={Database}
              label="HuggingFace"
            />
            <IconLink href="#citation" Icon={BookOpen} label="Paper" />
            <IconLink href="mailto:contact@pdeagent.dev" Icon={Mail} label="Contact" />
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-white/10 text-xs text-white/50 flex flex-wrap items-center justify-between gap-2">
          <p>© 2026 PDEAgent-Bench. All rights reserved.</p>
          <p>
            <a href="#home" className="hover:text-white">
              Back to top ↑
            </a>
          </p>
        </div>
      </div>
    </footer>
  );
}

function FooterLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      className="px-3 py-1.5 rounded-lg hover:text-white hover:bg-white/5 transition-colors"
    >
      {children}
    </a>
  );
}

function IconLink({
  href,
  Icon,
  label,
}: {
  href: string;
  Icon: React.ComponentType<{ className?: string }>;
  label: string;
}) {
  return (
    <a
      href={href}
      target={href.startsWith("http") ? "_blank" : undefined}
      rel={href.startsWith("http") ? "noopener noreferrer" : undefined}
      aria-label={label}
      title={label}
      className="w-9 h-9 grid place-items-center rounded-lg border border-white/10 hover:bg-white/10 hover:text-white transition-colors"
    >
      <Icon className="w-4 h-4" />
    </a>
  );
}
