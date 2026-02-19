import { useState } from "react";
import { Send } from "lucide-react";

interface TopicInputProps {
  placeholder?: string;
  onSubmit: (topic: string) => void;
  isLoading: boolean;
  suggestions?: string[];
}

export default function TopicInput({ placeholder, onSubmit, isLoading, suggestions = [] }: TopicInputProps) {
  const [topic, setTopic] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (topic.trim() && !isLoading) {
      onSubmit(topic.trim());
    }
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="relative">
        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          placeholder={placeholder || "Enter an ML topic..."}
          disabled={isLoading}
          className="w-full glass-card-elevated px-5 py-4 pr-14 text-sm rounded-xl border border-border/50 focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/50 transition-all placeholder:text-muted-foreground/50 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!topic.trim() || isLoading}
          className="absolute right-2 top-1/2 -translate-y-1/2 h-9 w-9 rounded-lg bg-primary text-primary-foreground flex items-center justify-center disabled:opacity-30 hover:opacity-90 transition-all"
        >
          <Send className="h-4 w-4" />
        </button>
      </form>

      {suggestions.length > 0 && !isLoading && (
        <div className="flex flex-wrap gap-2">
          {suggestions.map((s) => (
            <button
              key={s}
              onClick={() => { setTopic(s); onSubmit(s); }}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-muted/60 text-muted-foreground hover:bg-primary/10 hover:text-primary transition-all"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
