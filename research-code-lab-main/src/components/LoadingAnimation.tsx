import { Sparkles } from "lucide-react";

export default function LoadingAnimation({ message = "Generating..." }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
      <div className="relative">
        <div className="h-16 w-16 rounded-2xl bg-primary/10 flex items-center justify-center animate-pulse-glow">
          <Sparkles className="h-7 w-7 text-primary animate-float" />
        </div>
      </div>
      <p className="mt-5 text-sm font-medium text-muted-foreground">{message}</p>
      <div className="mt-3 flex gap-1">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-primary/40"
            style={{ animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite` }}
          />
        ))}
      </div>
    </div>
  );
}
