import { useState, useRef, useEffect } from "react";
import { Search, Bell, Sun, Moon } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useTheme } from "@/components/ThemeProvider";
import { codingProblems } from "@/data/codingProblems";
import { searchPapers } from "@/data/mockPapers";

interface SearchResult {
  type: "paper" | "problem";
  title: string;
  subtitle: string;
  action: () => void;
}

export function TopBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const handleSearch = (q: string) => {
    setQuery(q);
    if (q.trim().length < 2) { setResults([]); setOpen(false); return; }

    const items: SearchResult[] = [];

    // Search problems
    const lq = q.toLowerCase();
    codingProblems.filter(p =>
      p.title.toLowerCase().includes(lq) || p.category.includes(lq)
    ).slice(0, 4).forEach(p => {
      items.push({
        type: "problem",
        title: p.title,
        subtitle: `${p.difficulty} · ${p.categoryIcon} ${p.category}`,
        action: () => { navigate("/arena"); setOpen(false); setQuery(""); },
      });
    });

    // Search papers
    const papers = searchPapers(q).slice(0, 4);
    papers.forEach(p => {
      items.push({
        type: "paper",
        title: p.title,
        subtitle: `${p.year} · ${p.citations.toLocaleString()} citations`,
        action: () => { navigate("/research"); setOpen(false); setQuery(""); },
      });
    });

    setResults(items);
    setOpen(items.length > 0);
  };

  return (
    <header className="h-14 border-b border-border bg-background/80 backdrop-blur-sm flex items-center justify-between px-6 shrink-0">
      <div ref={ref} className="relative flex items-center gap-3 flex-1 max-w-md">
        <Search className="w-4 h-4 text-muted-foreground" />
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onFocus={() => { if (results.length > 0) setOpen(true); }}
          placeholder="Search papers, problems..."
          className="bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 outline-none w-full"
        />
        {open && results.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 rounded-xl bg-card border border-border shadow-lg overflow-hidden z-50">
            {results.map((r, i) => (
              <button
                key={i}
                onClick={r.action}
                className="w-full text-left px-4 py-3 hover:bg-secondary/60 transition-colors flex items-center gap-3 border-b border-border last:border-0"
              >
                <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-secondary text-muted-foreground uppercase">
                  {r.type === "paper" ? "Paper" : "Problem"}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-foreground truncate">{r.title}</p>
                  <p className="text-[11px] text-muted-foreground">{r.subtitle}</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={toggleTheme}
          className="w-8 h-8 rounded-lg hover:bg-secondary flex items-center justify-center transition-colors"
          title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
        >
          {theme === "dark" ? <Sun className="w-4 h-4 text-muted-foreground" /> : <Moon className="w-4 h-4 text-muted-foreground" />}
        </button>
        <button className="w-8 h-8 rounded-lg hover:bg-secondary flex items-center justify-center transition-colors">
          <Bell className="w-4 h-4 text-muted-foreground" />
        </button>
        <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-xs font-medium text-primary">
          U
        </div>
      </div>
    </header>
  );
}
