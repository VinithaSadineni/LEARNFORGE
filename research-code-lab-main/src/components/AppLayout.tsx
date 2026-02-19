import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { LayoutDashboard, BookOpen, Code2, Headphones, Eye, Menu, X, Sparkles, Map, MessageSquare } from "lucide-react";
import ThemeToggle from "@/components/ThemeToggle";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Dashboard", path: "/", icon: LayoutDashboard },
  { title: "Text Learning", path: "/text-learning", icon: BookOpen },
  { title: "Code Generation", path: "/code-generation", icon: Code2 },
  { title: "Audio Lessons", path: "/audio-lessons", icon: Headphones },
  { title: "Visual Studio", path: "/visual-studio", icon: Eye },
  { title: "Roadmap", path: "/roadmap", icon: Map },
  { title: "Interview", path: "/interview", icon: MessageSquare },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="flex min-h-screen bg-background">
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-foreground/20 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-50 w-64 glass-card-elevated border-r border-border/50 flex flex-col transition-transform duration-300 lg:translate-x-0 lg:static lg:z-auto",
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex items-center gap-3 px-6 py-5 border-b border-border/50">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h1 className="text-base font-bold text-foreground tracking-tight">LearnForge AI</h1>
            <p className="text-[11px] text-muted-foreground font-medium">ML Education Platform</p>
          </div>
          <button
            onClick={() => setSidebarOpen(false)}
            className="ml-auto lg:hidden text-muted-foreground hover:text-foreground"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => setSidebarOpen(false)}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200",
                  isActive
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted/60"
                )}
              >
                <item.icon className={cn("h-[18px] w-[18px]", isActive && "text-primary")} />
                {item.title}
              </Link>
            );
          })}
        </nav>

        <div className="px-4 py-4 border-t border-border/50">
          <div className="glass-card p-3 text-center">
            <p className="text-xs text-muted-foreground">Powered by AI</p>
            <p className="text-[10px] text-muted-foreground/70 mt-0.5">confusion → clarity → mastery</p>
          </div>
        </div>
      </aside>

      {/* Footer component is rendered inside main below */}

      {/* Main */}
      <div className="flex-1 flex flex-col min-w-0">
        <header className="sticky top-0 z-30 glass-card border-b border-border/50 px-4 py-3 flex items-center gap-3 lg:px-6">
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden text-muted-foreground hover:text-foreground"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2 lg:hidden">
            <Sparkles className="h-4 w-4 text-primary" />
            <span className="font-semibold text-sm">LearnForge AI</span>
          </div>
          <div className="ml-auto">
            <ThemeToggle />
          </div>
        </header>

        <main className="flex-1 px-4 py-6 lg:px-8 lg:py-8 max-w-5xl mx-auto w-full">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-border/50 bg-card/50 backdrop-blur-sm mt-auto">
          <div className="max-w-5xl mx-auto px-4 py-10 lg:px-8">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-sm">
              {/* Product */}
              <div className="space-y-3">
                <h4 className="font-semibold text-foreground text-xs uppercase tracking-wider">Product</h4>
                <ul className="space-y-2">
                  {[
                    { label: "Text Learning", path: "/text-learning" },
                    { label: "Code Generation", path: "/code-generation" },
                    { label: "Audio Lessons", path: "/audio-lessons" },
                    { label: "Visual Studio", path: "/visual-studio" },
                    { label: "Roadmap", path: "/roadmap" },
                    { label: "Interview", path: "/interview" },
                  ].map((item) => (
                    <li key={item.path}>
                      <Link to={item.path} className="text-muted-foreground hover:text-foreground transition-colors text-xs">
                        {item.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Coming Soon */}
              <div className="space-y-3">
                <h4 className="font-semibold text-foreground text-xs uppercase tracking-wider">Coming Soon</h4>
                <ul className="space-y-2 text-xs text-muted-foreground">
                  <li>Adaptive Interviews</li>
                  <li>Personalized Paths</li>
                  <li>Skill Analytics</li>
                </ul>
              </div>

              {/* Company */}
              <div className="space-y-3">
                <h4 className="font-semibold text-foreground text-xs uppercase tracking-wider">Company</h4>
                <ul className="space-y-2 text-xs text-muted-foreground">
                  <li>Built for students</li>
                  <li>AI-powered learning</li>
                </ul>
              </div>

              {/* Legal */}
              <div className="space-y-3">
                <h4 className="font-semibold text-foreground text-xs uppercase tracking-wider">Legal</h4>
                <ul className="space-y-2 text-xs text-muted-foreground">
                  <li>Privacy</li>
                  <li>Terms</li>
                </ul>
              </div>
            </div>

            <div className="border-t border-border/50 mt-8 pt-6 flex flex-col sm:flex-row items-center justify-between gap-2">
              <p className="text-xs text-muted-foreground">© 2026 LearnForge AI.</p>
              <p className="text-xs text-muted-foreground">Built to turn learners into ML engineers.</p>
            </div>
          </div>
        </footer>
      </div>
    </div>
  );
}
