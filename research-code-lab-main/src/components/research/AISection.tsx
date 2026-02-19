import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { usePaperIntelligence, type SectionType } from "@/hooks/usePaperIntelligence";

interface AISectionProps {
  title: string;
  icon: string;
  section: SectionType;
  paperTitle: string;
  paperContext: string;
  mode: "summary" | "deep";
  renderContent: (data: any) => React.ReactNode;
}

export function AISection({
  title,
  icon,
  section,
  paperTitle,
  paperContext,
  mode,
  renderContent,
}: AISectionProps) {
  const [open, setOpen] = useState(false);
  const { fetchSection, loading, results, errors, getKey } = usePaperIntelligence();

  const key = getKey(section, paperTitle, mode);
  const isLoading = loading[key];
  const data = results[key];
  const error = errors[key];

  const handleToggle = async () => {
    const next = !open;
    setOpen(next);
    if (next && !data && !isLoading) {
      await fetchSection(section, paperTitle, paperContext, mode);
    }
  };

  return (
    <div className="rounded-lg border border-border/60 overflow-hidden">
      <button
        onClick={handleToggle}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-secondary/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm">{icon}</span>
          <span className="text-xs font-semibold text-foreground">{title}</span>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary font-medium">
            AI
          </span>
        </div>
        <div className="flex items-center gap-2">
          {isLoading && <Loader2 className="w-3 h-3 animate-spin text-primary" />}
          {open ? (
            <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />
          )}
        </div>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 pt-1">
              {isLoading && !data && (
                <div className="flex items-center gap-2 py-4">
                  <Loader2 className="w-4 h-4 animate-spin text-primary" />
                  <span className="text-xs text-muted-foreground">
                    Generating analysis...
                  </span>
                </div>
              )}
              {error && (
                <p className="text-xs text-destructive py-2">{error}</p>
              )}
              {data && renderContent(data)}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
