import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Mic, ArrowRight, Send, Loader2, CheckCircle2, XCircle, AlertTriangle, Trophy, Target, TrendingUp, TrendingDown, BarChart3, Clock, RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import { useInterviewEngine, type InterviewRound } from "@/hooks/useInterviewEngine";

const TOPICS = ["Random Forest", "XGBoost", "Logistic Regression", "K-Means", "Transformers"];

const ROUND_LABELS = [
  "Foundational Understanding",
  "Mechanism Deep-Dive",
  "Edge Cases & Failures",
  "Production Trade-offs",
  "System Design Depth",
];

const depthColors: Record<string, string> = {
  Surface: "text-destructive",
  Intermediate: "text-warning",
  Strong: "text-success",
  Expert: "text-primary",
};

const scoreColor = (s: number) => s >= 8 ? "text-success" : s >= 5 ? "text-warning" : "text-destructive";
const scoreBg = (s: number) => s >= 8 ? "bg-success/10 border-success/20" : s >= 5 ? "bg-warning/10 border-warning/20" : "bg-destructive/10 border-destructive/20";

function TopicSelect({ onStart }: { onStart: (t: string) => void }) {
  const [custom, setCustom] = useState("");

  return (
    <div className="max-w-2xl mx-auto p-8">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2.5 mb-1">
          <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center">
            <Mic className="w-4 h-4 text-primary" />
          </div>
          <h1 className="text-xl font-semibold text-foreground">ML Interview Engine</h1>
        </div>
        <p className="text-sm text-muted-foreground mb-6 ml-[42px]">
          5-round adaptive interview simulation calibrated to FAANG hiring bars.
        </p>

        <div className="space-y-3 mb-6">
          {TOPICS.map((t) => (
            <button
              key={t}
              onClick={() => onStart(t)}
              className="w-full text-left p-4 rounded-xl bg-card border border-border card-hover flex items-center justify-between group"
            >
              <span className="text-sm font-medium text-foreground">{t}</span>
              <ArrowRight className="w-4 h-4 text-muted-foreground group-hover:text-primary transition-colors" />
            </button>
          ))}
        </div>

        <div className="flex gap-2">
          <input
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && custom.trim() && onStart(custom.trim())}
            placeholder="Or enter a custom topic..."
            className="flex-1 px-4 py-2.5 rounded-xl bg-card border border-border text-sm text-foreground placeholder:text-muted-foreground/50 outline-none focus:border-primary/50 transition-colors"
          />
          <button
            onClick={() => custom.trim() && onStart(custom.trim())}
            disabled={!custom.trim()}
            className="px-4 py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-40 transition-colors"
          >
            Start
          </button>
        </div>
      </motion.div>
    </div>
  );
}

function RoundProgress({ current, total, rounds }: { current: number; total: number; rounds: InterviewRound[] }) {
  return (
    <div className="flex items-center gap-1.5 mb-4">
      {Array.from({ length: total }, (_, i) => {
        const r = rounds[i];
        const isActive = i + 1 === current;
        const isDone = r?.evaluation != null;
        const score = r?.evaluation?.score;

        return (
          <div key={i} className="flex items-center gap-1.5">
            <div className={`w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold border transition-all ${
              isActive ? "border-primary bg-primary/15 text-primary" :
              isDone ? `${scoreBg(score || 0)} ${scoreColor(score || 0)}` :
              "border-border bg-card text-muted-foreground"
            }`}>
              {isDone ? score : i + 1}
            </div>
            {i < total - 1 && <div className={`w-6 h-px ${isDone ? "bg-primary/40" : "bg-border"}`} />}
          </div>
        );
      })}
    </div>
  );
}

function EvaluationPanel({ evaluation }: { evaluation: NonNullable<InterviewRound["evaluation"]> }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-4 rounded-xl bg-card border border-border space-y-3"
    >
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">🧠 Evaluation</h4>
        <div className={`text-lg font-bold ${scoreColor(evaluation.score)}`}>
          {evaluation.score}/10
        </div>
      </div>

      <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wide ${
        depthColors[evaluation.depthLevel]
      } bg-current/10`}>
        <span className={depthColors[evaluation.depthLevel]}>{evaluation.depthLevel}</span>
      </div>

      {evaluation.correctConcepts.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-success uppercase tracking-wide mb-1 flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" /> Correct Concepts
          </p>
          <ul className="space-y-0.5">
            {evaluation.correctConcepts.map((c, i) => (
              <li key={i} className="text-xs text-foreground/80 flex items-start gap-1.5">
                <span className="text-success mt-0.5">•</span> {c}
              </li>
            ))}
          </ul>
        </div>
      )}

      {evaluation.missingConcepts.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-warning uppercase tracking-wide mb-1 flex items-center gap-1">
            <AlertTriangle className="w-3 h-3" /> Missing Critical Concepts
          </p>
          <ul className="space-y-0.5">
            {evaluation.missingConcepts.map((c, i) => (
              <li key={i} className="text-xs text-foreground/80 flex items-start gap-1.5">
                <span className="text-warning mt-0.5">•</span> {c}
              </li>
            ))}
          </ul>
        </div>
      )}

      {evaluation.misconceptions.length > 0 && (
        <div>
          <p className="text-[10px] font-semibold text-destructive uppercase tracking-wide mb-1 flex items-center gap-1">
            <XCircle className="w-3 h-3" /> Misconceptions
          </p>
          <ul className="space-y-0.5">
            {evaluation.misconceptions.map((c, i) => (
              <li key={i} className="text-xs text-foreground/80 flex items-start gap-1.5">
                <span className="text-destructive mt-0.5">•</span> {c}
              </li>
            ))}
          </ul>
        </div>
      )}
    </motion.div>
  );
}

function FinalReportView({ report, topic, rounds, onRestart }: {
  report: NonNullable<import("@/hooks/useInterviewEngine").FinalReport>;
  topic: string;
  rounds: InterviewRound[];
  onRestart: () => void;
}) {
  const [expanded, setExpanded] = useState<number | null>(null);
  const decisionColor = report.decision === "Hire" ? "text-success" : report.decision === "Lean Hire" ? "text-warning" : "text-destructive";
  const decisionBg = report.decision === "Hire" ? "bg-success/10 border-success/20" : report.decision === "Lean Hire" ? "bg-warning/10 border-warning/20" : "bg-destructive/10 border-destructive/20";

  return (
    <div className="max-w-2xl mx-auto p-8 space-y-5">
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2.5 mb-4">
          <div className="w-8 h-8 rounded-lg bg-primary/15 flex items-center justify-center">
            <Trophy className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-foreground">Interview Complete</h2>
            <p className="text-xs text-muted-foreground">{topic} • 5 Rounds</p>
          </div>
        </div>

        {/* Decision badge */}
        <div className={`p-4 rounded-xl border ${decisionBg} mb-4`}>
          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide mb-1">Final Decision</p>
          <p className={`text-2xl font-bold ${decisionColor}`}>{report.decision}</p>
          <p className="text-sm text-muted-foreground mt-1">Average Score: {report.averageScore}/10</p>
        </div>

        {/* Metrics grid */}
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="p-3 rounded-xl bg-card border border-border">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingUp className="w-3 h-3 text-success" />
              <p className="text-[10px] font-semibold text-muted-foreground uppercase">Strongest</p>
            </div>
            <p className="text-sm font-medium text-foreground">{report.strongestDimension}</p>
          </div>
          <div className="p-3 rounded-xl bg-card border border-border">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingDown className="w-3 h-3 text-destructive" />
              <p className="text-[10px] font-semibold text-muted-foreground uppercase">Weakest</p>
            </div>
            <p className="text-sm font-medium text-foreground">{report.weakestDimension}</p>
          </div>
          <div className="p-3 rounded-xl bg-card border border-border">
            <div className="flex items-center gap-1.5 mb-1">
              <BarChart3 className="w-3 h-3 text-primary" />
              <p className="text-[10px] font-semibold text-muted-foreground uppercase">Conceptual Depth</p>
            </div>
            <p className="text-sm font-medium text-foreground">{report.conceptualDepth}</p>
          </div>
          <div className="p-3 rounded-xl bg-card border border-border">
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="w-3 h-3 text-primary" />
              <p className="text-[10px] font-semibold text-muted-foreground uppercase">System Thinking</p>
            </div>
            <p className="text-sm font-medium text-foreground">{report.systemThinking}</p>
          </div>
        </div>

        {/* Round breakdown */}
        <div className="space-y-2 mb-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Round Breakdown</h4>
          {rounds.map((r) => (
            <div key={r.round} className="rounded-xl bg-card border border-border overflow-hidden">
              <button
                onClick={() => setExpanded(expanded === r.round ? null : r.round)}
                className="w-full flex items-center justify-between p-3 text-left hover:bg-secondary/30 transition-colors"
              >
                <div className="flex items-center gap-2.5">
                  <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold border ${scoreBg(r.evaluation?.score || 0)} ${scoreColor(r.evaluation?.score || 0)}`}>
                    {r.evaluation?.score}
                  </div>
                  <span className="text-xs font-medium text-foreground">{ROUND_LABELS[r.round - 1]}</span>
                </div>
                {expanded === r.round ? <ChevronUp className="w-3.5 h-3.5 text-muted-foreground" /> : <ChevronDown className="w-3.5 h-3.5 text-muted-foreground" />}
              </button>
              <AnimatePresence>
                {expanded === r.round && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="border-t border-border"
                  >
                    <div className="p-3 space-y-2">
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase mb-0.5">Question</p>
                        <p className="text-xs text-foreground/80">{r.question}</p>
                      </div>
                      <div>
                        <p className="text-[10px] font-semibold text-muted-foreground uppercase mb-0.5">Your Answer</p>
                        <p className="text-xs text-foreground/80">{r.userAnswer}</p>
                      </div>
                      {r.evaluation && <EvaluationPanel evaluation={r.evaluation} />}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ))}
        </div>

        {/* Improvement Roadmap */}
        <div className="p-4 rounded-xl bg-card border border-border">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">📋 Improvement Roadmap</h4>
          <ul className="space-y-1.5">
            {report.improvementRoadmap.map((item, i) => (
              <li key={i} className="text-xs text-foreground/80 flex items-start gap-2">
                <span className="text-primary mt-0.5 font-bold">{i + 1}.</span> {item}
              </li>
            ))}
          </ul>
        </div>

        <button
          onClick={onRestart}
          className="w-full mt-4 py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors flex items-center justify-center gap-2"
        >
          <RotateCcw className="w-3.5 h-3.5" /> Start New Interview
        </button>
      </motion.div>
    </div>
  );
}

function InterviewActive({ session, loading, onSubmit }: {
  session: import("@/hooks/useInterviewEngine").InterviewSession;
  loading: boolean;
  onSubmit: (answer: string) => void;
}) {
  const [answer, setAnswer] = useState("");
  const currentRound = session.rounds[session.currentRound - 1];

  const handleSubmit = () => {
    if (!answer.trim() || loading) return;
    onSubmit(answer.trim());
    setAnswer("");
  };

  return (
    <div className="max-w-2xl mx-auto p-8 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-sm font-semibold text-foreground">{session.topic} Interview</h2>
          <p className="text-xs text-muted-foreground">Round {session.currentRound}/5 · {ROUND_LABELS[session.currentRound - 1]}</p>
        </div>
      </div>

      <RoundProgress current={session.currentRound} total={5} rounds={session.rounds} />

      {/* Show previous evaluations */}
      {session.rounds.filter((r) => r.evaluation && r.round < session.currentRound).map((r) => (
        <div key={r.round} className="space-y-2">
          <div className="p-3 rounded-xl bg-secondary/50 border border-border">
            <p className="text-[10px] font-semibold text-muted-foreground uppercase mb-1">Round {r.round} Question</p>
            <p className="text-xs text-foreground/80 mb-2">{r.question}</p>
            <p className="text-[10px] font-semibold text-muted-foreground uppercase mb-1">Your Answer</p>
            <p className="text-xs text-foreground/60">{r.userAnswer}</p>
          </div>
          {r.evaluation && <EvaluationPanel evaluation={r.evaluation} />}
        </div>
      ))}

      {/* Current question */}
      {currentRound && (
        <motion.div
          key={currentRound.round}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl bg-card border border-primary/20 card-glow"
        >
          <p className="text-[10px] font-semibold text-primary uppercase tracking-wide mb-2">
            Round {currentRound.round} — {ROUND_LABELS[currentRound.round - 1]}
          </p>
          <p className="text-sm text-foreground leading-relaxed">{currentRound.question}</p>
        </motion.div>
      )}

      {/* Show current round's evaluation if evaluating just completed */}
      {currentRound?.evaluation && session.status === "evaluating" && (
        <EvaluationPanel evaluation={currentRound.evaluation} />
      )}

      {/* Answer input */}
      {session.status === "answering" && (
        <div className="space-y-2">
          <textarea
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && e.metaKey) handleSubmit(); }}
            placeholder="Type your answer... (⌘+Enter to submit)"
            rows={6}
            className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm text-foreground placeholder:text-muted-foreground/50 outline-none focus:border-primary/50 transition-colors resize-none font-mono leading-relaxed"
          />
          <div className="flex justify-between items-center">
            <p className="text-[10px] text-muted-foreground">
              {answer.length > 0 ? `${answer.split(/\s+/).filter(Boolean).length} words` : "Be precise and technical"}
            </p>
            <button
              onClick={handleSubmit}
              disabled={!answer.trim() || loading}
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 disabled:opacity-40 transition-colors"
            >
              {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
              Submit Answer
            </button>
          </div>
        </div>
      )}

      {(session.status === "evaluating" || session.status === "waiting_question") && loading && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground p-4">
          <Loader2 className="w-4 h-4 animate-spin text-primary" />
          {session.status === "evaluating" ? "Evaluating your response..." : "Preparing question..."}
        </div>
      )}
    </div>
  );
}

function HistorySidebar({ history, onNew }: {
  history: import("@/hooks/useInterviewEngine").InterviewHistory[];
  onNew: () => void;
}) {
  if (history.length === 0) return null;

  return (
    <div className="mb-6 p-4 rounded-xl bg-card border border-border">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wide flex items-center gap-1.5">
          <Clock className="w-3 h-3" /> Past Interviews
        </h3>
      </div>
      <div className="space-y-2">
        {history.slice(0, 5).map((h, i) => {
          const decisionColor = h.finalDecision === "Hire" ? "text-success" : h.finalDecision === "Lean Hire" ? "text-warning" : "text-destructive";
          return (
            <div key={i} className="flex items-center justify-between py-1.5 border-b border-border last:border-0">
              <div>
                <p className="text-xs font-medium text-foreground">{h.topic}</p>
                <p className="text-[10px] text-muted-foreground">
                  {new Date(h.timestamp).toLocaleDateString()}
                </p>
              </div>
              <div className="text-right">
                <p className={`text-xs font-bold ${decisionColor}`}>{h.finalDecision}</p>
                <p className="text-[10px] text-muted-foreground">{h.averageScore}/10</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const InterviewEngine = () => {
  const { session, loading, history, startInterview, submitAnswer, resetInterview } = useInterviewEngine();

  return (
    <div className="overflow-y-auto h-[calc(100vh-3.5rem)]">
      {session.status === "topic_select" && (
        <>
          <div className="max-w-2xl mx-auto px-8 pt-8">
            <HistorySidebar history={history} onNew={resetInterview} />
          </div>
          <TopicSelect onStart={startInterview} />
        </>
      )}

      {session.status === "complete" && session.finalReport ? (
        <FinalReportView
          report={session.finalReport}
          topic={session.topic}
          rounds={session.rounds}
          onRestart={resetInterview}
        />
      ) : (
        (session.status === "answering" || session.status === "evaluating" || session.status === "waiting_question") && (
          <InterviewActive session={session} loading={loading} onSubmit={submitAnswer} />
        )
      )}
    </div>
  );
};

export default InterviewEngine;
