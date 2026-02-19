import { CheckCircle2, Rocket, Brain } from "lucide-react";
import mlRoadmapImg from "@/assets/ml-roadmap.png";
import dlRoadmapImg from "@/assets/dl-roadmap.png";

const mlFoundations = ["Python for data work", "Linear algebra essentials", "Probability & statistics", "Data handling"];
const mlCore = ["Supervised learning", "Unsupervised learning", "Bias vs variance", "Model evaluation", "Feature engineering"];
const mlAdvanced = ["Ensemble methods", "Hyperparameter tuning", "Optimization thinking", "Interpretability"];
const mlRealWorld = ["End-to-end projects", "Competitions / Kaggle", "Deployment basics", "Business impact thinking"];

const dlPrereqs = ["ML fundamentals", "Calculus intuition", "Numpy / tensors", "Data pipelines"];
const dlCore = ["Neural networks", "Backpropagation", "CNN", "Sequence models"];
const dlModern = ["Attention", "Transformers", "LLM basics", "Fine-tuning"];
const dlEngineering = ["PyTorch / TensorFlow", "Training workflows", "Experiment tracking", "Inference optimization"];
const dlCareer = ["Reading research", "Production ML mindset", "System design awareness", "Tradeoff reasoning"];

const available = [
  "Text Learning with guided explanations",
  "Interview-ready Code Generation",
  "Podcast-style Audio Lessons",
  "Visual Studio for concept visualization",
];

export default function Roadmap() {
  return (
    <div className="space-y-10">
      {/* Header */}
      <section className="text-center space-y-4 pt-6 lg:pt-10">
        <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium">
          <Rocket className="h-3 w-3" />
          Product Roadmap
        </div>
        <h1 className="text-3xl lg:text-4xl font-extrabold tracking-tight">
          <span className="gradient-text">LearnForge AI</span>{" "}
          <span className="text-foreground">Roadmap</span>
        </h1>
        <p className="text-muted-foreground text-base max-w-lg mx-auto leading-relaxed">
          Building the smartest way to master Machine Learning.
        </p>
      </section>

      {/* Available Now */}
      <div className="max-w-2xl mx-auto">
        <div className="glass-card-elevated p-6 space-y-4">
          <div className="flex items-center gap-3">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-600">
              <CheckCircle2 className="h-5 w-5 text-emerald-500" />
              Available Now
            </div>
          </div>
          <ul className="space-y-2.5">
            {available.map((item) => (
              <li key={item} className="flex items-start gap-2.5 text-sm text-muted-foreground">
                <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary/40 shrink-0" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ML Mastery Path */}
      <section className="max-w-3xl mx-auto space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium">
            <Rocket className="h-3 w-3" />
            Learning Path
          </div>
          <h2 className="text-2xl lg:text-3xl font-extrabold tracking-tight text-foreground">
            🚀 Machine Learning Mastery Path
          </h2>
          <p className="text-muted-foreground text-sm max-w-md mx-auto">
            Step-by-step journey from fundamentals to job-ready ML engineer.
          </p>
        </div>
        <img src={mlRoadmapImg} alt="Machine Learning Mastery Roadmap" className="w-full rounded-xl border border-border shadow-lg" />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <PathBlock title="Foundations" items={mlFoundations} step={1} />
          <PathBlock title="Core Machine Learning" items={mlCore} step={2} />
          <PathBlock title="Advanced ML Skills" items={mlAdvanced} step={3} />
          <PathBlock title="Real World Readiness" items={mlRealWorld} step={4} />
        </div>
      </section>

      {/* DL & AI Engineer Path */}
      <section className="max-w-3xl mx-auto space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-primary/10 text-primary text-xs font-medium">
            <Brain className="h-3 w-3" />
            Learning Path
          </div>
          <h2 className="text-2xl lg:text-3xl font-extrabold tracking-tight text-foreground">
            🧠 Deep Learning & AI Engineer Path
          </h2>
          <p className="text-muted-foreground text-sm max-w-md mx-auto">
            From neural network foundations to modern AI systems.
          </p>
        </div>
        <img src={dlRoadmapImg} alt="Deep Learning AI Engineer Roadmap" className="w-full rounded-xl border border-border shadow-lg" />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <PathBlock title="Prerequisites" items={dlPrereqs} step={1} />
          <PathBlock title="Core Deep Learning" items={dlCore} step={2} />
          <PathBlock title="Modern AI Stack" items={dlModern} step={3} />
          <PathBlock title="Engineering & Scale" items={dlEngineering} step={4} />
          <PathBlock title="Career Maturity" items={dlCareer} step={5} className="sm:col-span-2 sm:max-w-[calc(50%-0.5rem)]" />
        </div>
      </section>

      {/* Closing */}
      <section className="text-center pb-8 pt-4">
        <div className="glass-card-elevated inline-block px-8 py-5 max-w-md">
          <p className="text-sm font-semibold text-foreground">
            We are building the operating system for ML education.
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            From confusion → clarity → implementation → mastery
          </p>
        </div>
      </section>
    </div>
  );
}

function PathBlock({ title, items, step, className }: { title: string; items: string[]; step: number; className?: string }) {
  return (
    <div className={`glass-card-elevated p-5 space-y-3 ${className ?? ""}`}>
      <div className="flex items-center gap-2.5">
        <span className="flex items-center justify-center h-6 w-6 rounded-full bg-primary/15 text-primary text-xs font-bold">{step}</span>
        <h3 className="text-sm font-bold text-foreground tracking-tight uppercase">{title}</h3>
      </div>
      <ul className="space-y-1.5">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-2 text-sm text-muted-foreground">
            <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary/40 shrink-0" />
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
