import { Link } from "react-router-dom";
import { BookOpen, Code2, Headphones, Eye, ArrowRight, Sparkles, MessageSquare } from "lucide-react";
import RecentTopics from "@/components/RecentTopics";

const features = [
  {
    icon: BookOpen,
    title: "Text Learning",
    description: "Get comprehensive, structured explanations of any ML concept with intuition, math, and practical applications.",
    path: "/text-learning",
    gradient: "from-blue-500/10 to-indigo-500/10",
    iconColor: "text-blue-500",
  },
  {
    icon: Code2,
    title: "Code Generation",
    description: "Generate production-quality Python implementations with comments, dependencies, and example usage.",
    path: "/code-generation",
    gradient: "from-emerald-500/10 to-teal-500/10",
    iconColor: "text-emerald-500",
  },
  {
    icon: Headphones,
    title: "Audio Lessons",
    description: "Listen to engaging, podcast-style explanations that make complex ML concepts feel like a conversation.",
    path: "/audio-lessons",
    gradient: "from-violet-500/10 to-purple-500/10",
    iconColor: "text-violet-500",
  },
  {
    icon: Eye,
    title: "Visual Studio",
    description: "Generate clear diagrams and visual representations of ML concepts powered by AI.",
    path: "/visual-studio",
    gradient: "from-rose-500/10 to-pink-500/10",
    iconColor: "text-rose-500",
  },
  {
    icon: MessageSquare,
    title: "Interview Simulator",
    description: "Practice ML interview questions with an AI interviewer that adapts to your answers in real-time.",
    path: "/interview",
    gradient: "from-amber-500/10 to-orange-500/10",
    iconColor: "text-amber-500",
  },
];

export default function Dashboard() {
  return (
    <div className="space-y-12">
      {/* Hero */}
      <section className="text-center space-y-5 pt-8 lg:pt-12">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium mb-2">
          <Sparkles className="h-3 w-3" />
          AI-Powered Learning
        </div>
        <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight text-balance">
          <span className="gradient-text">Master Machine Learning</span>
          <br />
          <span className="text-foreground">Faster</span>
        </h1>
        <p className="text-muted-foreground text-base lg:text-lg max-w-xl mx-auto leading-relaxed">
          Understand concepts, generate working Python code, and listen to AI-powered lessons.
        </p>
        <Link
          to="/text-learning"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-primary text-primary-foreground font-semibold text-sm hover:opacity-90 transition-all shadow-lg hover:shadow-xl"
        >
          Start Learning
          <ArrowRight className="h-4 w-4" />
        </Link>
      </section>

      {/* Recent Activity */}
      <section className="max-w-2xl mx-auto">
        <RecentTopics />
      </section>

      {/* Feature Cards */}
      <section className="grid gap-5 md:grid-cols-2 lg:grid-cols-3">
        {features.map((feature, i) => (
          <Link
            key={feature.path}
            to={feature.path}
            className="group glass-card-elevated p-6 hover:scale-[1.02] transition-all duration-300 animate-fade-in"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            <div className={`inline-flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br ${feature.gradient} mb-4`}>
              <feature.icon className={`h-5 w-5 ${feature.iconColor}`} />
            </div>
            <h3 className="text-base font-bold text-foreground mb-2">{feature.title}</h3>
            <p className="text-sm text-muted-foreground leading-relaxed mb-4">{feature.description}</p>
            <span className="inline-flex items-center gap-1 text-xs font-medium text-primary group-hover:gap-2 transition-all">
              Try it <ArrowRight className="h-3 w-3" />
            </span>
          </Link>
        ))}
      </section>

      {/* Tagline */}
      <section className="text-center pb-8">
        <p className="text-sm text-muted-foreground/60 font-medium tracking-wide">
          From confusion → clarity → implementation → mastery
        </p>
      </section>
    </div>
  );
}
