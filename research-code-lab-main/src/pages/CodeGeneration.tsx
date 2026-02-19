import { useState, useCallback, useEffect } from "react";
import { Code2 } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import TopicInput from "@/components/TopicInput";
import LoadingAnimation from "@/components/LoadingAnimation";
import ActionButtons from "@/components/ActionButtons";
import MarkdownRenderer from "@/components/MarkdownRenderer";
import CrossModuleLinks from "@/components/CrossModuleLinks";
import { streamFromEdgeFunction } from "@/lib/streaming";
import { trackActivity } from "@/lib/progression";
import { toast } from "@/hooks/use-toast";

const SUGGESTIONS = ["Linear Regression", "K-Means Clustering", "Decision Tree", "PCA", "Logistic Regression"];

export default function CodeGeneration() {
  const [content, setContent] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentTopic, setCurrentTopic] = useState("");
  const [searchParams, setSearchParams] = useSearchParams();

  const generate = useCallback((topic: string) => {
    setIsLoading(true);
    setContent("");
    setCurrentTopic(topic);

    streamFromEdgeFunction({
      functionName: "generate-code",
      topic,
      onDelta: (text) => setContent((prev) => prev + text),
      onDone: () => {
        setIsLoading(false);
        trackActivity(topic, "code");
      },
      onError: (error) => {
        console.error(error);
        toast({ title: "AI temporarily unavailable", description: "Please try again in a moment." });
        setIsLoading(false);
      },
    });
  }, []);

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
          <Code2 className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-bold text-foreground">Code Generation</h1>
        </div>
        <p className="text-sm text-muted-foreground">Generate production-quality Python implementations for any ML algorithm.</p>
      </div>

      <TopicInput
        placeholder="e.g., K-Means Clustering, Neural Network from scratch..."
        onSubmit={generate}
        isLoading={isLoading}
        suggestions={SUGGESTIONS}
      />

      {isLoading && !content && <LoadingAnimation message="Writing Python code..." />}

      {content && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-medium tracking-wide px-1">
            <span>Generated for: <span className="text-foreground">{currentTopic}</span></span>
            <span className="text-muted-foreground/40">·</span>
            <span>AI Mentor Mode</span>
          </div>
          <ActionButtons
            content={content}
            filename={`${currentTopic.replace(/\s+/g, "-").toLowerCase()}.py`}
            onRegenerate={() => generate(currentTopic)}
            isLoading={isLoading}
          />
          <MarkdownRenderer content={content} />

          {!isLoading && (
            <div className="space-y-3">
              <p className="text-[11px] text-muted-foreground font-medium tracking-wide px-1">Continue with this topic →</p>
              <CrossModuleLinks topic={currentTopic} currentModule="code" />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
