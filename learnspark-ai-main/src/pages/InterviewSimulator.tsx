import { useState, useCallback, useRef, useEffect } from "react";
import { MessageSquare, Send, Loader2, RotateCcw } from "lucide-react";
import TopicInput from "@/components/TopicInput";
import { streamFromEdgeFunction } from "@/lib/streaming";
import { trackActivity } from "@/lib/progression";
import { toast } from "@/hooks/use-toast";

const SUGGESTIONS = ["Neural Networks", "Random Forest", "Gradient Descent", "Transformers", "Backpropagation"];

interface Message {
  role: "interviewer" | "candidate";
  content: string;
}

export default function InterviewSimulator() {
  const [topic, setTopic] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [started, setStarted] = useState(false);
  const [input, setInput] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const startInterview = useCallback((t: string) => {
    setTopic(t);
    setStarted(true);
    setMessages([]);
    setIsLoading(true);
    trackActivity(t, "text", { depth: "interview-sim" });

    let response = "";
    streamFromEdgeFunction({
      functionName: "generate-interview",
      topic: t,
      extraBody: { action: "start" },
      onDelta: (text) => {
        response += text;
        setMessages([{ role: "interviewer", content: response }]);
      },
      onDone: () => setIsLoading(false),
      onError: () => {
        toast({ title: "AI temporarily unavailable", description: "Please try again." });
        setIsLoading(false);
        setStarted(false);
      },
    });
  }, []);

  const sendAnswer = useCallback(() => {
    if (!input.trim() || isLoading) return;
    const userMsg = input.trim();
    setInput("");

    const newMessages: Message[] = [...messages, { role: "candidate", content: userMsg }];
    setMessages(newMessages);
    setIsLoading(true);

    // Build conversation context for the AI
    const conversationContext = newMessages
      .map((m) => `${m.role === "interviewer" ? "Interviewer" : "Candidate"}: ${m.content}`)
      .join("\n\n");

    let response = "";
    streamFromEdgeFunction({
      functionName: "generate-interview",
      topic,
      extraBody: { action: "continue", conversation: conversationContext },
      onDelta: (text) => {
        response += text;
        setMessages([...newMessages, { role: "interviewer", content: response }]);
      },
      onDone: () => setIsLoading(false),
      onError: () => {
        toast({ title: "AI temporarily unavailable", description: "Please try again." });
        setIsLoading(false);
      },
    });
  }, [input, messages, topic, isLoading]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendAnswer();
    }
  };

  if (!started) {
    return (
      <div className="space-y-6">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5 text-primary" />
            <h1 className="text-xl font-bold text-foreground">Interview Simulator</h1>
          </div>
          <p className="text-sm text-muted-foreground">Practice ML interview questions with an adaptive AI interviewer.</p>
        </div>

        <TopicInput
          placeholder="e.g., Neural Networks, SVM, Attention Mechanism..."
          onSubmit={startInterview}
          isLoading={isLoading}
          suggestions={SUGGESTIONS}
        />

        <div className="glass-card p-8 text-center animate-fade-in">
          <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 mb-4">
            <MessageSquare className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-base font-bold text-foreground mb-2">How It Works</h3>
          <div className="text-sm text-muted-foreground space-y-1.5 max-w-md mx-auto">
            <p>1. Choose an ML topic to be interviewed on</p>
            <p>2. The AI interviewer asks progressively deeper questions</p>
            <p>3. Answer in your own words — get real-time feedback</p>
            <p>4. Build confidence for real ML interviews</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-bold text-foreground">Interview: {topic}</h1>
        </div>
        <button
          onClick={() => { setStarted(false); setMessages([]); }}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-muted/60 text-muted-foreground hover:text-foreground hover:bg-muted transition-all"
        >
          <RotateCcw className="h-3 w-3" />
          New Topic
        </button>
      </div>

      {/* Chat messages */}
      <div className="glass-card p-4 lg:p-6 space-y-4 min-h-[400px] max-h-[600px] overflow-y-auto">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "candidate" ? "justify-end" : "justify-start"}`}>
            <div
              className={`max-w-[85%] rounded-xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === "candidate"
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted/60 text-foreground"
              }`}
            >
              <p className="text-[10px] font-bold uppercase tracking-wider mb-1 opacity-60">
                {msg.role === "interviewer" ? "🎯 Interviewer" : "You"}
              </p>
              {msg.content}
            </div>
          </div>
        ))}
        {isLoading && messages.length > 0 && messages[messages.length - 1].role === "candidate" && (
          <div className="flex justify-start">
            <div className="bg-muted/60 rounded-xl px-4 py-3 flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Thinking...
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your answer..."
          disabled={isLoading}
          rows={2}
          className="flex-1 glass-card-elevated px-4 py-3 text-sm rounded-xl border border-border/50 focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all placeholder:text-muted-foreground/50 disabled:opacity-50 resize-none"
        />
        <button
          onClick={sendAnswer}
          disabled={!input.trim() || isLoading}
          className="h-auto px-4 rounded-xl bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-30 hover:opacity-90 transition-all"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
