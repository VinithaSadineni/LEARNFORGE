import { useState, useCallback, useEffect } from "react";
import { BookOpen } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import TopicInput from "@/components/TopicInput";
import LoadingAnimation from "@/components/LoadingAnimation";
import ActionButtons from "@/components/ActionButtons";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import TestYourself from "@/components/TestYourself";
import CrossModuleLinks from "@/components/CrossModuleLinks";
import { streamFromEdgeFunction } from "@/lib/streaming";
import { trackActivity } from "@/lib/progression";
import { toast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const SUGGESTIONS = ["Linear Regression", "Neural Networks", "Random Forest", "Gradient Descent", "Transformers"];

const DEPTH_OPTIONS = [
  { value: "quick", label: "Quick Summary" },
  { value: "interview", label: "Interview Ready" },
  { value: "mastery", label: "Comprehensive Mastery" },
] as const;

const DEPTH_LABELS: Record<string, string> = {
  quick: "Quick Summary",
  interview: "Interview Ready",
  mastery: "Comprehensive Mastery",
};

export default function TextLearning() {
  const [content, setContent] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentTopic, setCurrentTopic] = useState("");
  const [depth, setDepth] = useState("interview");
  const [searchParams, setSearchParams] = useSearchParams();

  const generate = useCallback((topic: string) => {
    setIsLoading(true);
    setContent("");
    setCurrentTopic(topic);

    streamFromEdgeFunction({
      functionName: "generate-text-lesson",
      topic,
      extraBody: { depth },
      onDelta: (text) => setContent((prev) => prev + text),
      onDone: () => {
        setIsLoading(false);
        trackActivity(topic, "text", { depth });
      },
      onError: (error) => {
        console.error(error);
        toast({ title: "AI temporarily unavailable", description: "Please try again in a moment." });
        setIsLoading(false);
      },
    });
  }, [depth]);

  // Auto-generate from query param
  useEffect(() => {
    const topic = searchParams.get("topic");
    if (topic && !content && !isLoading) {
      setSearchParams({}, { replace: true });
      generate(topic);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-bold text-foreground">Text Learning</h1>
        </div>
        <p className="text-sm text-muted-foreground">Enter any ML topic and get a comprehensive, structured lesson.</p>
      </div>

      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-muted-foreground whitespace-nowrap">Learning Depth</label>
        <Select value={depth} onValueChange={setDepth}>
          <SelectTrigger className="w-[220px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DEPTH_OPTIONS.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <TopicInput
        placeholder="e.g., Convolutional Neural Networks, SVM, Backpropagation..."
        onSubmit={generate}
        isLoading={isLoading}
        suggestions={SUGGESTIONS}
      />

      {isLoading && !content && <LoadingAnimation message="Generating your lesson..." />}

      {content && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-medium tracking-wide px-1">
            <span>Generated for: <span className="text-foreground">{currentTopic}</span></span>
            <span className="text-muted-foreground/40">·</span>
            <span>You are in: <span className="text-primary font-semibold">{DEPTH_LABELS[depth]}</span> mode</span>
          </div>
          <ActionButtons
            content={content}
            filename={`${currentTopic.replace(/\s+/g, "-").toLowerCase()}-lesson.md`}
            onRegenerate={() => generate(currentTopic)}
            isLoading={isLoading}
          />
          <MarkdownRenderer content={content} />

          {/* Cross-module links */}
          {!isLoading && (
            <div className="space-y-3">
              <p className="text-[11px] text-muted-foreground font-medium tracking-wide px-1">Continue with this topic →</p>
              <CrossModuleLinks topic={currentTopic} currentModule="text" />
            </div>
          )}

          {/* Test Yourself */}
          {!isLoading && <TestYourself topic={currentTopic} lessonContent={content} />}
        </div>
      )}
    </div>
  );
}
