import { getRecentTopics, getStats } from "@/lib/progression";
import { useNavigate } from "react-router-dom";
import { Clock, Flame, BookOpen, Code2, Headphones, Eye } from "lucide-react";

const moduleIcons = {
  text: BookOpen,
  code: Code2,
  audio: Headphones,
  visual: Eye,
};

const modulePaths = {
  text: "/text-learning",
  code: "/code-generation",
  audio: "/audio-lessons",
  visual: "/visual-studio",
};

export default function RecentTopics() {
  const recent = getRecentTopics(6);
  const stats = getStats();
  const navigate = useNavigate();

  if (recent.length === 0) return null;

  return (
    <div className="glass-card-elevated p-5 space-y-4 animate-fade-in">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-bold text-foreground">Recent Activity</h3>
        </div>
        {stats.streak > 0 && (
          <span className="inline-flex items-center gap-1 text-xs font-bold text-primary">
            <Flame className="h-3 w-3" />
            {stats.streak} day streak
          </span>
        )}
      </div>

      <div className="flex flex-wrap gap-2">
        <span className="px-2.5 py-1 rounded-full bg-muted/60 text-[11px] font-medium text-muted-foreground">
          {stats.uniqueTopics} topics explored
        </span>
        <span className="px-2.5 py-1 rounded-full bg-muted/60 text-[11px] font-medium text-muted-foreground">
          {stats.codingProblems} code generations
        </span>
        {stats.quizzesTaken > 0 && (
          <span className="px-2.5 py-1 rounded-full bg-muted/60 text-[11px] font-medium text-muted-foreground">
            {stats.quizzesTaken} quizzes taken
          </span>
        )}
      </div>

      <div className="space-y-1.5">
        {recent.map((entry, i) => {
          const Icon = moduleIcons[entry.module];
          return (
            <button
              key={i}
              onClick={() => navigate(`${modulePaths[entry.module]}?topic=${encodeURIComponent(entry.topic)}`)}
              className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-muted-foreground hover:bg-muted/60 hover:text-foreground transition-all text-left"
            >
              <Icon className="h-3.5 w-3.5 shrink-0" />
              <span className="truncate flex-1">{entry.topic}</span>
              <span className="text-[10px] text-muted-foreground/60 shrink-0">
                {formatTimeAgo(entry.timestamp)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}
