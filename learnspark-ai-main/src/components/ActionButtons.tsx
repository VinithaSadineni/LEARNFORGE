import { Copy, Download, RefreshCw, Check } from "lucide-react";
import { useState } from "react";
import { toast } from "@/hooks/use-toast";

interface ActionButtonsProps {
  content: string;
  filename?: string;
  onRegenerate: () => void;
  isLoading: boolean;
}

export default function ActionButtons({ content, filename = "lesson.md", onRegenerate, isLoading }: ActionButtonsProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    toast({ title: "Copied to clipboard!" });
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([content], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Downloaded!" });
  };

  return (
    <div className="flex gap-2 flex-wrap">
      <button
        onClick={handleCopy}
        className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-muted/60 text-foreground hover:bg-muted transition-all"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-primary" /> : <Copy className="h-3.5 w-3.5" />}
        {copied ? "Copied" : "Copy"}
      </button>
      <button
        onClick={handleDownload}
        className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-muted/60 text-foreground hover:bg-muted transition-all"
      >
        <Download className="h-3.5 w-3.5" />
        Download
      </button>
      <button
        onClick={onRegenerate}
        disabled={isLoading}
        className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-all disabled:opacity-50"
      >
        <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? "animate-spin" : ""}`} />
        Regenerate
      </button>
    </div>
  );
}
