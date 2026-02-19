import { useState, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";

export interface InterviewRound {
  round: number;
  question: string;
  userAnswer: string;
  evaluation: {
    correctConcepts: string[];
    missingConcepts: string[];
    misconceptions: string[];
    depthLevel: "Surface" | "Intermediate" | "Strong" | "Expert";
    score: number;
  } | null;
}

export interface InterviewSession {
  topic: string;
  rounds: InterviewRound[];
  currentRound: number;
  status: "topic_select" | "waiting_question" | "answering" | "evaluating" | "complete";
  finalReport: FinalReport | null;
}

export interface FinalReport {
  averageScore: number;
  strongestDimension: string;
  weakestDimension: string;
  conceptualDepth: string;
  systemThinking: string;
  decision: "Hire" | "Lean Hire" | "No Hire";
  improvementRoadmap: string[];
}

export interface InterviewHistory {
  topic: string;
  averageScore: number;
  finalDecision: string;
  timestamp: number;
  roundScores: number[];
  depthLevels: string[];
}

const HISTORY_KEY = "learnforge_interview_history";

function loadHistory(): InterviewHistory[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveHistory(h: InterviewHistory[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
}

const ROUND_LABELS = [
  "Foundational conceptual understanding",
  "Mechanism-level explanation",
  "Edge cases and failure modes",
  "Production trade-offs and scaling",
  "System design or optimization depth",
];

export function useInterviewEngine() {
  const [session, setSession] = useState<InterviewSession>({
    topic: "",
    rounds: [],
    currentRound: 0,
    status: "topic_select",
    finalReport: null,
  });
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<InterviewHistory[]>(loadHistory);

  const startInterview = useCallback(async (topic: string) => {
    setSession({ topic, rounds: [], currentRound: 1, status: "waiting_question", finalReport: null });
    setLoading(true);
    try {
      const { data, error } = await supabase.functions.invoke("interview-engine", {
        body: { action: "ask", topic, round: 1, roundLabel: ROUND_LABELS[0], previousRounds: [] },
      });
      if (error) throw error;
      const question = data?.question || "Explain the core idea behind " + topic + ".";
      setSession((s) => ({
        ...s,
        rounds: [{ round: 1, question, userAnswer: "", evaluation: null }],
        status: "answering",
      }));
    } catch (e) {
      console.error("Failed to get question:", e);
      // Fallback question
      const fallbacks: Record<string, string[]> = {
        "Random Forest": [
          "Explain how Random Forest reduces variance compared to a single decision tree. What role does bagging play?",
          "Walk me through how feature subsampling at each split improves generalization. What's the mathematical intuition?",
          "When would Random Forest fail catastrophically? Describe scenarios where it underperforms linear models.",
          "You're deploying a Random Forest with 500 trees to serve 10K QPS. What are the latency/memory tradeoffs and how do you optimize?",
          "Design a real-time feature importance monitoring system for a Random Forest in production. How do you detect feature drift?"
        ],
        "XGBoost": [
          "Explain gradient boosting from first principles. How does XGBoost differ from vanilla gradient boosting?",
          "Walk through XGBoost's objective function including the regularization terms. Why does the second-order Taylor expansion matter?",
          "Describe scenarios where XGBoost overfits despite regularization. How do you diagnose and fix it?",
          "You need to train XGBoost on 1TB of data with 10K features. What distributed training strategies would you use?",
          "Design a system that automatically tunes XGBoost hyperparameters for 100 different prediction tasks daily."
        ],
        "Transformers": [
          "Explain the self-attention mechanism. Why does it scale quadratically and what does each matrix (Q, K, V) represent?",
          "Walk through multi-head attention mathematically. Why multiple heads instead of a single large attention?",
          "When does self-attention fail? Describe failure modes with very long sequences and positional encoding limitations.",
          "You're serving a 7B parameter transformer at 100ms latency SLA. Walk through your optimization strategy.",
          "Design an efficient inference system for a transformer model that handles variable-length inputs with batching."
        ],
      };
      const defaultQs = [
        `Explain the fundamental concept behind ${topic}. What problem does it solve?`,
        `Walk through the mathematical mechanism of ${topic}. What are the key equations?`,
        `Describe edge cases where ${topic} fails. What assumptions does it violate?`,
        `You're deploying ${topic} in production at scale. What are the critical tradeoffs?`,
        `Design a system that uses ${topic} as a core component. How do you handle monitoring and failure modes?`,
      ];
      const questions = fallbacks[topic] || defaultQs;
      setSession((s) => ({
        ...s,
        rounds: [{ round: 1, question: questions[0], userAnswer: "", evaluation: null }],
        status: "answering",
      }));
    } finally {
      setLoading(false);
    }
  }, []);

  const submitAnswer = useCallback(async (answer: string) => {
    const round = session.currentRound;
    setSession((s) => {
      const rounds = [...s.rounds];
      rounds[round - 1] = { ...rounds[round - 1], userAnswer: answer };
      return { ...s, rounds, status: "evaluating" };
    });
    setLoading(true);

    try {
      const { data, error } = await supabase.functions.invoke("interview-engine", {
        body: {
          action: "evaluate",
          topic: session.topic,
          round,
          roundLabel: ROUND_LABELS[round - 1],
          question: session.rounds[round - 1].question,
          answer,
          previousRounds: session.rounds.slice(0, round - 1).map((r) => ({
            question: r.question,
            answer: r.userAnswer,
            score: r.evaluation?.score || 0,
          })),
        },
      });
      if (error) throw error;

      const evaluation = data?.evaluation || {
        correctConcepts: ["Mentioned core concept"],
        missingConcepts: ["Deeper mathematical formulation"],
        misconceptions: [],
        depthLevel: "Intermediate" as const,
        score: 6,
      };

      const nextQuestion = data?.nextQuestion;

      setSession((s) => {
        const rounds = [...s.rounds];
        rounds[round - 1] = { ...rounds[round - 1], userAnswer: answer, evaluation };

        if (round >= 5) {
          // Generate final report
          const scores = rounds.map((r) => r.evaluation?.score || 0);
          const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
          const depths = rounds.map((r) => r.evaluation?.depthLevel || "Surface");
          
          const dimensions = ["Conceptual", "Mechanism", "Edge Cases", "Production", "System Design"];
          const strongest = dimensions[scores.indexOf(Math.max(...scores))];
          const weakest = dimensions[scores.indexOf(Math.min(...scores))];

          const decision: FinalReport["decision"] = avg >= 7.5 ? "Hire" : avg >= 5.5 ? "Lean Hire" : "No Hire";

          const roadmap: string[] = [];
          if (scores[0] < 7) roadmap.push("Review fundamental concepts and mathematical foundations");
          if (scores[1] < 7) roadmap.push("Practice explaining mechanisms step-by-step with equations");
          if (scores[2] < 7) roadmap.push("Study failure modes and edge cases in real datasets");
          if (scores[3] < 7) roadmap.push("Read production ML case studies from FAANG engineering blogs");
          if (scores[4] < 7) roadmap.push("Practice system design interviews with ML components");
          if (roadmap.length === 0) roadmap.push("Continue deepening expertise across all dimensions");

          const finalReport: FinalReport = {
            averageScore: Math.round(avg * 10) / 10,
            strongestDimension: strongest,
            weakestDimension: weakest,
            conceptualDepth: depths.filter((d) => d === "Strong" || d === "Expert").length >= 3 ? "Strong" : "Developing",
            systemThinking: scores[4] >= 7 ? "Strong" : "Needs Development",
            decision,
            improvementRoadmap: roadmap,
          };

          // Save to history
          const entry: InterviewHistory = {
            topic: s.topic,
            averageScore: finalReport.averageScore,
            finalDecision: decision,
            timestamp: Date.now(),
            roundScores: scores,
            depthLevels: depths,
          };
          const newHistory = [entry, ...loadHistory()].slice(0, 20);
          saveHistory(newHistory);
          setHistory(newHistory);

          return { ...s, rounds, currentRound: round, status: "complete", finalReport };
        }

        // Add next round
        const nextRound = round + 1;
        const newRound: InterviewRound = {
          round: nextRound,
          question: nextQuestion || `[Round ${nextRound}] Follow-up on ${s.topic}...`,
          userAnswer: "",
          evaluation: null,
        };
        rounds.push(newRound);

        return { ...s, rounds, currentRound: nextRound, status: "answering" };
      });
    } catch (e) {
      console.error("Evaluation failed:", e);
      // Fallback evaluation
      const score = Math.min(10, Math.max(2, Math.floor(answer.length / 40) + 3));
      const evaluation = {
        correctConcepts: answer.length > 50 ? ["Shows understanding of core concept"] : ["Attempted response"],
        missingConcepts: ["Deeper mathematical analysis", "Practical implementation details"],
        misconceptions: [] as string[],
        depthLevel: (score >= 7 ? "Strong" : score >= 5 ? "Intermediate" : "Surface") as InterviewRound["evaluation"] extends null ? never : NonNullable<InterviewRound["evaluation"]>["depthLevel"],
        score,
      };

      setSession((s) => {
        const rounds = [...s.rounds];
        rounds[round - 1] = { ...rounds[round - 1], userAnswer: answer, evaluation };

        if (round >= 5) {
          const scores = rounds.map((r) => r.evaluation?.score || 0);
          const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
          const depths = rounds.map((r) => r.evaluation?.depthLevel || "Surface");
          const dimensions = ["Conceptual", "Mechanism", "Edge Cases", "Production", "System Design"];
          const strongest = dimensions[scores.indexOf(Math.max(...scores))];
          const weakest = dimensions[scores.indexOf(Math.min(...scores))];
          const decision: FinalReport["decision"] = avg >= 7.5 ? "Hire" : avg >= 5.5 ? "Lean Hire" : "No Hire";
          const roadmap = ["Strengthen theoretical foundations", "Practice production ML scenarios"];

          const finalReport: FinalReport = {
            averageScore: Math.round(avg * 10) / 10,
            strongestDimension: strongest,
            weakestDimension: weakest,
            conceptualDepth: "Developing",
            systemThinking: "Needs Development",
            decision,
            improvementRoadmap: roadmap,
          };

          const entry: InterviewHistory = {
            topic: s.topic, averageScore: finalReport.averageScore, finalDecision: decision,
            timestamp: Date.now(), roundScores: scores, depthLevels: depths,
          };
          const newHistory = [entry, ...loadHistory()].slice(0, 20);
          saveHistory(newHistory);
          setHistory(newHistory);

          return { ...s, rounds, currentRound: round, status: "complete", finalReport };
        }

        const fallbackQs: Record<string, string[]> = {
          "Random Forest": [
            "", "Walk through how feature subsampling at each split improves generalization.",
            "When would Random Forest fail? Describe scenarios where linear models win.",
            "Deploying 500 trees at 10K QPS — what are the latency/memory tradeoffs?",
            "Design a real-time feature importance monitoring system for production RF."
          ],
          "XGBoost": [
            "", "Walk through XGBoost's objective function with regularization terms.",
            "Describe scenarios where XGBoost overfits despite regularization.",
            "Training XGBoost on 1TB data with 10K features — distributed strategies?",
            "Design auto-tuning system for 100 daily XGBoost prediction tasks."
          ],
          "Transformers": [
            "", "Walk through multi-head attention mathematically. Why multiple heads?",
            "When does self-attention fail? Long sequences, positional encoding limits?",
            "Serving a 7B transformer at 100ms latency SLA. Optimization strategy?",
            "Design efficient inference with variable-length inputs and batching."
          ],
        };
        const nextRound = round + 1;
        const topicQs = fallbackQs[s.topic];
        const nextQ = topicQs?.[nextRound - 1] || `As a follow-up on ${s.topic}: explain the ${ROUND_LABELS[nextRound - 1]}.`;

        rounds.push({ round: nextRound, question: nextQ, userAnswer: "", evaluation: null });
        return { ...s, rounds, currentRound: nextRound, status: "answering" };
      });
    } finally {
      setLoading(false);
    }
  }, [session]);

  const resetInterview = useCallback(() => {
    setSession({ topic: "", rounds: [], currentRound: 0, status: "topic_select", finalReport: null });
  }, []);

  return { session, loading, history, startInterview, submitAnswer, resetInterview };
}
