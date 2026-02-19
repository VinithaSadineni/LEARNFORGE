import { LayoutDashboard, FileText, Code2, Mic, Zap } from "lucide-react";
import { NavLink } from "@/components/NavLink";

const navItems = [
  { title: "Dashboard", url: "/", icon: LayoutDashboard },
  { title: "Research Papers", url: "/research", icon: FileText },
  { title: "ML Coding Arena", url: "/arena", icon: Code2 },
  { title: "Interview Engine", url: "/interview", icon: Mic },
];

export function AppSidebar() {
  return (
    <aside className="w-60 min-h-screen bg-sidebar border-r border-sidebar-border flex flex-col shrink-0">
      <div className="p-5 border-b border-sidebar-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
            <Zap className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-foreground tracking-tight">LearnForge AI</h1>
            <p className="text-[10px] text-muted-foreground tracking-wider uppercase">Research & Code</p>
          </div>
        </div>
      </div>

      <nav className="flex-1 p-3 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.url}
            to={item.url}
            end={item.url === "/"}
            className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground transition-colors"
            activeClassName="bg-sidebar-accent text-primary font-medium"
          >
            <item.icon className="w-4 h-4" />
            <span>{item.title}</span>
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-sidebar-border">
        <div className="px-3 py-2 rounded-lg bg-secondary/50">
          <p className="text-[11px] font-medium text-muted-foreground">v1.0 Beta</p>
          <p className="text-[10px] text-muted-foreground/60 mt-0.5">LearnForge AI Platform</p>
        </div>
      </div>
    </aside>
  );
}
