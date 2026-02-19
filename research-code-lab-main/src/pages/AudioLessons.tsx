import { useState, useCallback, useRef, useEffect } from "react";
import { Headphones, Play, Pause, Volume2 } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import TopicInput from "@/components/TopicInput";
import LoadingAnimation from "@/components/LoadingAnimation";
import ActionButtons from "@/components/ActionButtons";
import CrossModuleLinks from "@/components/CrossModuleLinks";
import { streamFromEdgeFunction } from "@/lib/streaming";
import { trackActivity } from "@/lib/progression";
import { toast } from "@/hooks/use-toast";

const SUGGESTIONS = ["Overfitting", "Bias-Variance Tradeoff", "Feature Engineering", "Cross Validation", "Ensemble Methods"];

export default function AudioLessons() {
  const [content, setContent] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [currentTopic, setCurrentTopic] = useState("");
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();

  const generate = useCallback((topic: string) => {
    setIsLoading(true);
    setContent("");
    setCurrentTopic(topic);
    setIsPlaying(false);
    setProgress(0);
    window.speechSynthesis.cancel();

    streamFromEdgeFunction({
      functionName: "generate-audio-lesson",
      topic,
      onDelta: (text) => setContent((prev) => prev + text),
      onDone: () => {
        setIsLoading(false);
        trackActivity(topic, "audio");
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

  const togglePlayback = () => {
    if (isPlaying) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
      return;
    }

    const cleanText = content.replace(/\*([^*]+)\*/g, "$1").replace(/\.\.\./g, ", ");
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.95;
    utterance.pitch = 1;
    
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.name.includes("Google") || v.name.includes("Samantha") || v.lang === "en-US");
    if (preferred) utterance.voice = preferred;

    utterance.onboundary = (e) => {
      if (e.name === "word") {
        setProgress(Math.min((e.charIndex / cleanText.length) * 100, 100));
      }
    };
    utterance.onend = () => { setIsPlaying(false); setProgress(100); };
    utterance.onerror = () => { setIsPlaying(false); };

    utteranceRef.current = utterance;
    window.speechSynthesis.speak(utterance);
    setIsPlaying(true);
  };

  const formatScript = (text: string) => {
    return text.split("\n\n").map((paragraph, i) => {
      const formatted = paragraph
        .replace(/\*([^*]+)\*/g, '<em class="text-primary font-medium not-italic">$1</em>')
        .replace(/\.\.\./g, '<span class="text-muted-foreground/40">...</span>');
      return (
        <p
          key={i}
          className="text-sm text-foreground/80 leading-relaxed mb-4"
          dangerouslySetInnerHTML={{ __html: formatted }}
        />
      );
    });
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Headphones className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-bold text-foreground">Audio Lessons</h1>
        </div>
        <p className="text-sm text-muted-foreground">Generate podcast-style lessons and listen while you learn.</p>
      </div>

      <TopicInput
        placeholder="e.g., Overfitting, Backpropagation, Attention Mechanism..."
        onSubmit={generate}
        isLoading={isLoading}
        suggestions={SUGGESTIONS}
      />

      {isLoading && !content && <LoadingAnimation message="Writing your podcast script..." />}

      {content && !isLoading && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-medium tracking-wide px-1">
            <span>Generated for: <span className="text-foreground">{currentTopic}</span></span>
            <span className="text-muted-foreground/40">·</span>
            <span>AI Tutor Mode</span>
          </div>

          {/* Audio Player */}
          <div className="glass-card-elevated p-5 space-y-4">
            <div className="flex items-center gap-4">
              <button
                onClick={togglePlayback}
                className="h-12 w-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center hover:opacity-90 transition-all shadow-lg flex-shrink-0"
              >
                {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5 ml-0.5" />}
              </button>
              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-foreground">{currentTopic}</span>
                  <Volume2 className="h-4 w-4 text-muted-foreground" />
                </div>
                <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <p className="text-[11px] text-muted-foreground">
                  {isPlaying ? "Playing..." : progress > 0 ? "Paused" : "Ready to play"} · Browser Text-to-Speech
                </p>
              </div>
            </div>
          </div>

          <ActionButtons
            content={content}
            filename={`${currentTopic.replace(/\s+/g, "-").toLowerCase()}-audio-script.md`}
            onRegenerate={() => generate(currentTopic)}
            isLoading={isLoading}
          />

          {/* Script */}
          <div className="glass-card p-6 lg:p-8">
            <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-4">📝 Lesson Script</h3>
            {formatScript(content.replace(/KEY TAKEAWAYS:[\s\S]*$/, "").trim())}
          </div>

          {/* Key Takeaways */}
          {content.includes("KEY TAKEAWAYS:") && (
            <div className="glass-card p-6">
              <h3 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">🔑 Key Takeaways</h3>
              <ul className="space-y-1.5">
                {content
                  .split("KEY TAKEAWAYS:")[1]
                  ?.split("\n")
                  .filter((l) => l.trim().startsWith("-"))
                  .map((l, i) => (
                    <li key={i} className="text-sm text-foreground/80 flex items-start gap-2">
                      <span className="text-primary mt-0.5">•</span>
                      {l.replace(/^-\s*/, "")}
                    </li>
                  ))}
              </ul>
            </div>
          )}

          {/* Cross-module links */}
          <div className="space-y-3">
            <p className="text-[11px] text-muted-foreground font-medium tracking-wide px-1">Continue with this topic →</p>
            <CrossModuleLinks topic={currentTopic} currentModule="audio" />
          </div>
        </div>
      )}

      {content && isLoading && (
        <div className="glass-card p-6 animate-fade-in">
          <div className="prose prose-sm">
            {formatScript(content)}
          </div>
          <div className="shimmer h-4 w-3/4 rounded mt-2" />
        </div>
      )}
    </div>
  );
}
