import { useNavigate } from "react-router-dom";
import { Code2, Headphones, Eye, BookOpen } from "lucide-react";

interface CrossModuleLinksProps {
  topic: string;
  currentModule: "text" | "code" | "audio" | "visual";
}

const modules = [
  { key: "text" as const, label: "Learn Concept", icon: BookOpen, path: "/text-learning" },
  { key: "code" as const, label: "Generate Code", icon: Code2, path: "/code-generation" },
  { key: "audio" as const, label: "Listen as Audio", icon: Headphones, path: "/audio-lessons" },
  { key: "visual" as const, label: "Visualize It", icon: Eye, path: "/visual-studio" },
];

export default function CrossModuleLinks({ topic, currentModule }: CrossModuleLinksProps) {
  const navigate = useNavigate();

  const available = modules.filter((m) => m.key !== currentModule);

  return (
    <div className="flex flex-wrap gap-2">
      {available.map((m) => (
        <button
          key={m.key}
          onClick={() => navigate(`${m.path}?topic=${encodeURIComponent(topic)}`)}
          className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg bg-muted/60 text-muted-foreground hover:bg-primary/10 hover:text-primary transition-all"
        >
          <m.icon className="h-3 w-3" />
          {m.label}
        </button>
      ))}
    </div>
  );
}
