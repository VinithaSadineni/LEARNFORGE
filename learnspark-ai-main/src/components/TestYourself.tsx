import { useState, useCallback } from "react";
import { Brain, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { streamFromEdgeFunction } from "@/lib/streaming";
import { saveQuizResult } from "@/lib/progression";

interface Question {
  question: string;
  options: string[];
  correct: number; // 0-indexed
}

interface TestYourselfProps {
  topic: string;
  lessonContent: string;
}

export default function TestYourself({ topic, lessonContent }: TestYourselfProps) {
  const [questions, setQuestions] = useState<Question[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [answers, setAnswers] = useState<Record<number, number>>({});
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const generate = useCallback(() => {
    setIsLoading(true);
    setError("");
    setAnswers({});
    setSubmitted(false);
    let raw = "";

    streamFromEdgeFunction({
      functionName: "generate-quiz",
      topic,
      extraBody: { lessonSummary: lessonContent.slice(0, 2000) },
      onDelta: (text) => { raw += text; },
      onDone: () => {
        try {
          // Extract JSON from the response
          const jsonMatch = raw.match(/\[[\s\S]*\]/);
          if (!jsonMatch) throw new Error("No JSON");
          const parsed = JSON.parse(jsonMatch[0]) as Question[];
          if (!Array.isArray(parsed) || parsed.length === 0) throw new Error("Empty");
          setQuestions(parsed.slice(0, 5));
        } catch {
          setError("Could not generate quiz. Please try again.");
        }
        setIsLoading(false);
      },
      onError: () => {
        setError("AI temporarily unavailable. Please try again.");
        setIsLoading(false);
      },
    });
  }, [topic, lessonContent]);

  const handleSubmit = () => {
    setSubmitted(true);
    const correct = questions.reduce((acc, q, i) => acc + (answers[i] === q.correct ? 1 : 0), 0);
    saveQuizResult(topic, correct, questions.length);
  };

  const allAnswered = Object.keys(answers).length === questions.length && questions.length > 0;
  const score = submitted ? questions.reduce((acc, q, i) => acc + (answers[i] === q.correct ? 1 : 0), 0) : 0;

  if (questions.length === 0 && !isLoading && !error) {
    return (
      <button
        onClick={generate}
        className="w-full glass-card-elevated p-4 flex items-center justify-center gap-3 text-sm font-semibold text-primary hover:bg-primary/5 transition-all"
      >
        <Brain className="h-4 w-4" />
        Test Yourself — Generate Quiz
      </button>
    );
  }

  return (
    <div className="glass-card-elevated p-6 space-y-5 animate-fade-in">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Brain className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-bold text-foreground">Test Yourself</h3>
        </div>
        {submitted && (
          <span className="text-xs font-bold px-3 py-1 rounded-full bg-primary/10 text-primary">
            {score}/{questions.length} correct
          </span>
        )}
      </div>

      {isLoading && (
        <div className="flex items-center justify-center py-8 gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" />
          Generating questions...
        </div>
      )}

      {error && (
        <div className="text-center py-6 space-y-3">
          <p className="text-sm text-muted-foreground">{error}</p>
          <button onClick={generate} className="text-xs font-medium text-primary hover:underline">Try Again</button>
        </div>
      )}

      {questions.map((q, qi) => (
        <div key={qi} className="space-y-2">
          <p className="text-sm font-medium text-foreground">{qi + 1}. {q.question}</p>
          <div className="space-y-1.5">
            {q.options.map((opt, oi) => {
              const selected = answers[qi] === oi;
              const isCorrect = q.correct === oi;
              let stateClass = "hover:bg-muted/60";
              if (submitted) {
                if (isCorrect) stateClass = "bg-emerald-500/10 border-emerald-500/30";
                else if (selected && !isCorrect) stateClass = "bg-destructive/10 border-destructive/30";
              } else if (selected) {
                stateClass = "bg-primary/10 border-primary/30";
              }

              return (
                <button
                  key={oi}
                  disabled={submitted}
                  onClick={() => setAnswers((prev) => ({ ...prev, [qi]: oi }))}
                  className={`w-full text-left px-3 py-2 rounded-lg border border-border/50 text-sm transition-all flex items-center gap-2 ${stateClass} disabled:cursor-default`}
                >
                  {submitted && isCorrect && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />}
                  {submitted && selected && !isCorrect && <XCircle className="h-3.5 w-3.5 text-destructive shrink-0" />}
                  <span className={submitted && isCorrect ? "text-foreground font-medium" : "text-foreground/80"}>
                    {opt}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {questions.length > 0 && !submitted && (
        <button
          onClick={handleSubmit}
          disabled={!allAnswered}
          className="w-full py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-semibold disabled:opacity-40 hover:opacity-90 transition-all"
        >
          Submit Answers
        </button>
      )}

      {submitted && (
        <button
          onClick={generate}
          className="w-full py-2 rounded-lg bg-muted/60 text-foreground text-xs font-medium hover:bg-muted transition-all"
        >
          Generate New Quiz
        </button>
      )}
    </div>
  );
}
