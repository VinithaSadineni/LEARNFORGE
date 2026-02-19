import { useState } from "react";
import { ChevronDown, ChevronUp, ExternalLink, Star, Code, BookOpen, Zap, ArrowRight } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { type Paper, type ImpactClass } from "@/data/mockPapers";
import { AISection } from "./AISection";

const difficultyColor: Record<string, string> = {
  Beginner: "bg-success/15 text-success",
  Intermediate: "bg-warning/15 text-warning",
  Advanced: "bg-destructive/15 text-destructive",
};

const impactStyles: Record<ImpactClass, string> = {
  Foundational: "bg-primary/10 text-primary",
  Breakthrough: "bg-warning/15 text-warning",
  Optimization: "bg-accent/15 text-accent",
  "Production Innovation": "bg-success/15 text-success",
};

const whoLabels: Record<string, string> = {
  beginner: "🟢 Beginners",
  researcher: "🔬 Researchers",
  "interview prep": "💼 Interview Prep",
};

function Section({ title, children, icon }: { title: string; children: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        {icon}
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">{title}</h4>
      </div>
      <p className="text-sm text-foreground/85 leading-relaxed">{children}</p>
    </div>
  );
}

interface PaperCardProps {
  paper: Paper;
  index: number;
  mode: "summary" | "deep";
  onExpand: (id: string) => void;
}

export function PaperCard({ paper, index, mode, onExpand }: PaperCardProps) {
  const [expanded, setExpanded] = useState(false);
  const navigate = useNavigate();

  const handleToggle = () => {
    const next = !expanded;
    setExpanded(next);
    if (next) onExpand(paper.id);
  };

  const paperContext = `${paper.coreContribution}. ${paper.methodIntuition}. ${paper.problem}`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.35 }}
      className="rounded-xl border border-border bg-card card-hover overflow-hidden"
    >
      <button
        onClick={handleToggle}
        className="w-full text-left p-5 flex items-start justify-between gap-4"
      >
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full uppercase tracking-wide ${difficultyColor[paper.difficulty]}`}>
              {paper.difficulty}
            </span>
            <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${impactStyles[paper.impact]}`}>
              {paper.impact}
            </span>
            <span className="text-xs text-muted-foreground">{paper.year}</span>
            <span className="text-[10px] text-muted-foreground/60">•</span>
            <span className="text-[10px] text-muted-foreground">{paper.citations.toLocaleString()} citations</span>
          </div>
          <h3 className="text-sm font-semibold text-foreground leading-snug">{paper.title}</h3>
          <p className="text-xs text-muted-foreground mt-0.5">{paper.authors} · {paper.institution}</p>
        </div>
        <div className="mt-1 text-muted-foreground shrink-0">
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </div>
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
            <div className="px-5 pb-5 space-y-4 border-t border-border pt-4">
              <Section title="Why This Paper Matters" icon={<Star className="w-3.5 h-3.5 text-primary" />}>{paper.whyItMatters}</Section>
              <Section title="Problem It Addresses">{paper.problem}</Section>
              <Section title="Core Contribution">{paper.coreContribution}</Section>

              {mode === "deep" && (
                <Section title="Method Intuition (Simplified)">{paper.methodIntuition}</Section>
              )}

              <Section title="Results & Impact">{paper.results}</Section>

              <div className="grid grid-cols-2 gap-3">
                <Section title="Strengths">{paper.strengths}</Section>
                <Section title="Limitations">{paper.limitations}</Section>
              </div>

              <div className="p-3.5 rounded-lg bg-primary/5 border border-primary/10">
                <p className="text-xs font-semibold text-primary mb-1">TL;DR</p>
                <p className="text-sm text-foreground/90 leading-relaxed">{paper.tldr}</p>
              </div>

              <div className="p-3.5 rounded-lg bg-secondary">
                <p className="text-xs font-semibold text-muted-foreground mb-1">🔑 Key Insight</p>
                <p className="text-sm text-foreground leading-relaxed">{paper.keyInsight}</p>
              </div>

              {/* AI-Powered Sections */}
              <div className="space-y-2 pt-2">
                <AISection
                  title="How This Appears in Interviews"
                  icon="🧠"
                  section="interview"
                  paperTitle={paper.title}
                  paperContext={paperContext}
                  mode={mode}
                  renderContent={(data) => (
                    <div className="space-y-3 text-sm">
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1.5">Conceptual Questions</p>
                        <ul className="space-y-1">
                          {data.conceptual?.map((q: string, i: number) => (
                            <li key={i} className="text-foreground/85 text-xs leading-relaxed">• {q}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">System Design Angle</p>
                        <p className="text-xs text-foreground/85 leading-relaxed">{data.systemDesign}</p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Tradeoff Question</p>
                        <p className="text-xs text-foreground/85 leading-relaxed">{data.tradeoff}</p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Strong Answer Includes</p>
                        <ul className="space-y-1">
                          {data.strongAnswer?.map((a: string, i: number) => (
                            <li key={i} className="text-foreground/85 text-xs leading-relaxed">✓ {a}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                />

                <AISection
                  title="How You Would Implement This"
                  icon="⚙️"
                  section="implementation"
                  paperTitle={paper.title}
                  paperContext={paperContext}
                  mode={mode}
                  renderContent={(data) => (
                    <div className="space-y-3 text-sm">
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1.5">Algorithm Steps</p>
                        <ol className="space-y-1">
                          {data.steps?.map((s: string, i: number) => (
                            <li key={i} className="text-foreground/85 text-xs leading-relaxed">{i + 1}. {s}</li>
                          ))}
                        </ol>
                      </div>
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <p className="text-xs font-semibold text-muted-foreground mb-1">Time Complexity</p>
                          <p className="text-xs text-foreground/85 font-mono">{data.timeComplexity}</p>
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-muted-foreground mb-1">Space Complexity</p>
                          <p className="text-xs text-foreground/85 font-mono">{data.spaceComplexity}</p>
                        </div>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Data Structures</p>
                        <div className="flex flex-wrap gap-1.5">
                          {data.dataStructures?.map((ds: string, i: number) => (
                            <span key={i} className="text-[10px] px-2 py-0.5 rounded-full bg-secondary text-secondary-foreground">{ds}</span>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Optimizations</p>
                        <ul className="space-y-1">
                          {data.optimizations?.map((o: string, i: number) => (
                            <li key={i} className="text-foreground/85 text-xs leading-relaxed">→ {o}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                />

                <AISection
                  title="When This Breaks in Production"
                  icon="📊"
                  section="production"
                  paperTitle={paper.title}
                  paperContext={paperContext}
                  mode={mode}
                  renderContent={(data) => (
                    <div className="space-y-3 text-sm">
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Scaling Challenges</p>
                        <p className="text-xs text-foreground/85 leading-relaxed">{data.scaling}</p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Data Skew Issues</p>
                        <p className="text-xs text-foreground/85 leading-relaxed">{data.dataSkew}</p>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Monitoring Metrics</p>
                        <ul className="space-y-1">
                          {data.monitoring?.map((m: string, i: number) => (
                            <li key={i} className="text-foreground/85 text-xs leading-relaxed">📈 {m}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Failure Modes</p>
                        <ul className="space-y-1">
                          {data.failureModes?.map((f: string, i: number) => (
                            <li key={i} className="text-foreground/85 text-xs leading-relaxed">⚠️ {f}</li>
                          ))}
                        </ul>
                      </div>
                      <div>
                        <p className="text-xs font-semibold text-muted-foreground mb-1">Operational Tradeoff</p>
                        <p className="text-xs text-foreground/85 leading-relaxed">{data.tradeoffs}</p>
                      </div>
                    </div>
                  )}
                />
              </div>

              {/* Teach Me Button */}
              <TeachMeSection
                paperTitle={paper.title}
                paperContext={paperContext}
                mode={mode}
              />

              {/* Who Should Read */}
              <div className="flex items-center gap-3 flex-wrap">
                <span className="text-xs font-semibold text-muted-foreground">Who should read this:</span>
                {paper.whoShouldRead.map((w) => (
                  <span key={w} className="text-[10px] font-medium px-2 py-1 rounded-full bg-secondary text-secondary-foreground">
                    {whoLabels[w] || w}
                  </span>
                ))}
              </div>

              {/* Concepts */}
              {paper.concepts.length > 0 && (
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-semibold text-muted-foreground">Concepts:</span>
                  {paper.concepts.map((c) => (
                    <span key={c} className="text-[10px] font-medium px-2 py-0.5 rounded-full border border-border text-muted-foreground">
                      {c}
                    </span>
                  ))}
                </div>
              )}

              {/* Cross-Module Links */}
              <div className="flex flex-wrap gap-2 pt-2 border-t border-border">
                <button
                  onClick={() => navigate(`/arena?topic=${encodeURIComponent(paper.title)}`)}
                  className="inline-flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
                >
                  <Code className="w-3 h-3" /> Solve Related Problem
                </button>
                <button
                  onClick={() => navigate(`/research?prereq=${encodeURIComponent(paper.id)}`)}
                  className="inline-flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors"
                >
                  <BookOpen className="w-3 h-3" /> View Prerequisites
                </button>
                <a
                  href={paper.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-[11px] font-medium px-3 py-1.5 rounded-lg bg-secondary text-secondary-foreground hover:bg-secondary/80 transition-colors"
                >
                  <ExternalLink className="w-3 h-3" /> Original Paper
                </a>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function TeachMeSection({ paperTitle, paperContext, mode }: { paperTitle: string; paperContext: string; mode: "summary" | "deep" }) {
  const [show, setShow] = useState(false);

  return (
    <>
      {!show ? (
        <button
          onClick={() => setShow(true)}
          className="w-full flex items-center justify-center gap-2 py-3 rounded-lg border border-primary/20 bg-primary/5 text-primary text-xs font-semibold hover:bg-primary/10 transition-colors"
        >
          <Zap className="w-3.5 h-3.5" />
          Teach Me Why This Changed the Field
        </button>
      ) : (
        <AISection
          title="Why This Changed the Field"
          icon="⚡"
          section="history"
          paperTitle={paperTitle}
          paperContext={paperContext}
          mode={mode}
          renderContent={(data) => (
            <div className="space-y-3 text-sm">
              <div>
                <p className="text-xs font-semibold text-muted-foreground mb-1">What Existed Before</p>
                <p className="text-xs text-foreground/85 leading-relaxed">{data.before}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground mb-1">The Problem It Solved</p>
                <p className="text-xs text-foreground/85 leading-relaxed">{data.solved}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground mb-1">What It Unlocked</p>
                <p className="text-xs text-foreground/85 leading-relaxed">{data.unlocked}</p>
              </div>
              <div>
                <p className="text-xs font-semibold text-muted-foreground mb-1">Modern ML Legacy</p>
                <p className="text-xs text-foreground/85 leading-relaxed">{data.legacy}</p>
              </div>
            </div>
          )}
        />
      )}
    </>
  );
}
