import { useState } from "react";
import { Search, BookOpen, Sparkles, Loader2 } from "lucide-react";
import { motion } from "framer-motion";
import { searchPapers, type Paper } from "@/data/mockPapers";
import { useProgress } from "@/hooks/useProgress";
import { PaperCard } from "@/components/research/PaperCard";
import { ConceptTracker } from "@/components/research/ConceptTracker";
import { usePaperIntelligence } from "@/hooks/usePaperIntelligence";

const sectionLabels: Record<string, { icon: string; label: string }> = {
  foundation: { icon: "⭐", label: "Foundations" },
  improvement: { icon: "🚀", label: "Improvements" },
  modern: { icon: "🧠", label: "Modern Extensions" },
};

const suggestedTopics = ["Random Forest", "Transformers", "Gradient Boosting", "Reinforcement Learning", "Neural Networks", "Clustering"];

const ResearchPapers = () => {
  const [query, setQuery] = useState("");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [searched, setSearched] = useState(false);
  const [mode, setMode] = useState<"summary" | "deep">("summary");
  const { markPaperViewed } = useProgress();
  const { fetchSection, loading, results, getKey } = usePaperIntelligence();

  const handleSearch = (q?: string) => {
    const term = q || query;
    if (!term.trim()) return;
    setQuery(term);
    setPapers(searchPapers(term));
    setSearched(true);
  };

  const grouped = papers.reduce<Record<string, Paper[]>>((acc, p) => {
    const s = p.section || "foundation";
    if (!acc[s]) acc[s] = [];
    acc[s].push(p);
    return acc;
  }, {});

  const sectionOrder = ["foundation", "improvement", "modern"];

  // Empty state AI suggestions
  const emptyKey = getKey("empty_suggestions", query, mode, query);
  const emptyData = results[emptyKey];
  const emptyLoading = loading[emptyKey];

  const handleEmptySuggestions = () => {
    if (!emptyData && !emptyLoading) {
      fetchSection("empty_suggestions", query, "", mode, query);
    }
  };

  return (
    <div className="p-8 max-w-3xl mx-auto">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-primary" />
            <h1 className="text-xl font-semibold text-foreground">Research Intelligence</h1>
          </div>

          {/* Mode Toggle */}
          <div className="flex items-center gap-1 p-0.5 rounded-lg bg-secondary border border-border">
            <button
              onClick={() => setMode("summary")}
              className={`text-[11px] font-medium px-3 py-1.5 rounded-md transition-colors ${
                mode === "summary"
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Summary
            </button>
            <button
              onClick={() => setMode("deep")}
              className={`text-[11px] font-medium px-3 py-1.5 rounded-md transition-colors ${
                mode === "deep"
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Deep Technical
            </button>
          </div>
        </div>
        <p className="text-sm text-muted-foreground mb-6">
          Enter a topic, algorithm, or concept. We'll find the most important papers and explain them clearly.
        </p>

        <div className="flex gap-2 mb-6">
          <div className="flex-1 flex items-center gap-2.5 px-4 py-2.5 rounded-lg bg-secondary border border-border focus-within:border-primary/40 transition-colors">
            <Search className="w-4 h-4 text-muted-foreground shrink-0" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="e.g. transformer, random forest, reinforcement learning..."
              className="bg-transparent text-sm text-foreground placeholder:text-muted-foreground/50 outline-none w-full"
            />
          </div>
          <button
            onClick={() => handleSearch()}
            className="px-5 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors shrink-0"
          >
            Search
          </button>
        </div>
      </motion.div>

      {/* Concept Tracker */}
      <ConceptTracker />

      {searched && papers.length > 0 && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex items-center gap-2 px-4 py-3 rounded-lg bg-primary/5 border border-primary/10 mb-6"
          >
            <Sparkles className="w-4 h-4 text-primary shrink-0" />
            <p className="text-sm text-foreground/80">
              <span className="font-medium text-foreground">Research Intelligence.</span>{" "}
              Found {papers.length} key papers on "{query}".{" "}
              <span className="text-muted-foreground">
                {mode === "deep" ? "Deep technical mode active." : "Click any paper to expand."}
              </span>
            </p>
          </motion.div>

          {sectionOrder.map((sKey) => {
            const sectionPapers = grouped[sKey];
            if (!sectionPapers?.length) return null;
            const { icon, label } = sectionLabels[sKey];
            return (
              <div key={sKey} className="mb-6">
                <h2 className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-2">
                  <span>{icon}</span> {label}
                </h2>
                <div className="space-y-3">
                  {sectionPapers.map((paper, i) => (
                    <PaperCard
                      key={paper.id}
                      paper={paper}
                      index={i}
                      mode={mode}
                      onExpand={markPaperViewed}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </>
      )}

      {searched && papers.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center py-16"
          onAnimationComplete={handleEmptySuggestions}
        >
          <Search className="w-10 h-10 text-muted-foreground/30 mx-auto mb-4" />
          <h3 className="text-base font-semibold text-foreground mb-2">No exact matches for "{query}"</h3>
          <p className="text-sm text-muted-foreground mb-6">Try broader terms or related algorithms.</p>

          <div className="flex flex-wrap justify-center gap-2 mb-6">
            {suggestedTopics.map((t) => (
              <button
                key={t}
                onClick={() => handleSearch(t)}
                className="text-xs font-medium px-3 py-1.5 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors"
              >
                {t}
              </button>
            ))}
          </div>

          {emptyLoading && (
            <div className="flex items-center justify-center gap-2 mt-4">
              <Loader2 className="w-4 h-4 animate-spin text-primary" />
              <span className="text-xs text-muted-foreground">Finding related topics...</span>
            </div>
          )}

          {emptyData && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-6 text-left max-w-md mx-auto space-y-4"
            >
              {emptyData.relatedTopics?.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground mb-2">AI-Suggested Related Topics</p>
                  <div className="flex flex-wrap gap-1.5">
                    {emptyData.relatedTopics.map((t: string) => (
                      <button
                        key={t}
                        onClick={() => handleSearch(t)}
                        className="text-[11px] font-medium px-2.5 py-1 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                      >
                        {t}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {emptyData.foundationalPapers?.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-muted-foreground mb-2">Foundational Papers to Explore</p>
                  <ul className="space-y-1">
                    {emptyData.foundationalPapers.map((p: string, i: number) => (
                      <li key={i} className="text-xs text-foreground/80 leading-relaxed">📄 {p}</li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </motion.div>
      )}

      {!searched && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="text-center py-16">
          <BookOpen className="w-10 h-10 text-muted-foreground/30 mx-auto mb-3" />
          <p className="text-sm text-muted-foreground">
            Try searching for{" "}
            <button onClick={() => handleSearch("transformer")} className="text-primary hover:underline">transformer</button>,{" "}
            <button onClick={() => handleSearch("random forest")} className="text-primary hover:underline">random forest</button>, or{" "}
            <button onClick={() => handleSearch("reinforcement learning")} className="text-primary hover:underline">reinforcement learning</button>
          </p>
        </motion.div>
      )}
    </div>
  );
};

export default ResearchPapers;
