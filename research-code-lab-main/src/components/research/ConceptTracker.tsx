import { useMemo } from "react";
import { motion } from "framer-motion";
import { useProgress } from "@/hooks/useProgress";
import { searchPapers, type Paper } from "@/data/mockPapers";

export function ConceptTracker() {
  const { progress } = useProgress();

  const conceptCounts = useMemo(() => {
    if (progress.viewedPapers.length < 3) return null;

    // Aggregate concepts from all viewed papers
    // We need to find papers by searching for topics that would return them
    const counts: Record<string, number> = {};
    const allSearchTerms = [
      "transformer", "neural network", "random forest",
      "reinforcement learning", "clustering", "gradient boosting",
    ];

    for (const term of allSearchTerms) {
      const papers = searchPapers(term);
      for (const p of papers) {
        if (progress.viewedPapers.includes(p.id)) {
          for (const c of p.concepts) {
            counts[c] = (counts[c] || 0) + 1;
          }
        }
      }
    }

    // Sort by count descending, take top 8
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([name, count]) => ({ name, count }));
  }, [progress.viewedPapers]);

  if (!conceptCounts || conceptCounts.length === 0) return null;

  const maxCount = Math.max(...conceptCounts.map((c) => c.count));

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-card p-5 mb-6"
    >
      <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-widest mb-4">
        Concepts You've Explored
      </h3>
      <div className="space-y-2.5">
        {conceptCounts.map(({ name, count }) => {
          const pct = Math.round((count / maxCount) * 100);
          return (
            <div key={name} className="flex items-center gap-3">
              <span className="text-xs text-foreground/80 w-40 truncate shrink-0">
                {name}
              </span>
              <div className="flex-1 h-2 rounded-full bg-secondary overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.6, ease: "easeOut" }}
                  className="h-full rounded-full bg-primary/70"
                />
              </div>
              <span className="text-[10px] text-muted-foreground w-6 text-right">
                {count}
              </span>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
