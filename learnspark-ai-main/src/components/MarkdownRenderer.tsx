import { cn } from "@/lib/utils";

export default function MarkdownRenderer({ content, className }: { content: string; className?: string }) {
  // Simple markdown rendering with formatted sections
  const renderMarkdown = (text: string) => {
    const lines = text.split("\n");
    const elements: React.ReactNode[] = [];
    let inCodeBlock = false;
    let codeContent = "";
    let codeLang = "";
    let key = 0;

    for (const line of lines) {
      if (line.startsWith("```")) {
        if (inCodeBlock) {
          elements.push(
            <div key={key++} className="my-4 rounded-xl overflow-hidden border border-border/50">
              <div className="bg-muted/80 px-4 py-2 text-[11px] font-mono font-medium text-muted-foreground uppercase tracking-wider">
                {codeLang || "code"}
              </div>
              <pre className="bg-muted/30 p-4 overflow-x-auto">
                <code className="text-sm font-mono text-foreground leading-relaxed">{codeContent}</code>
              </pre>
            </div>
          );
          codeContent = "";
          codeLang = "";
          inCodeBlock = false;
        } else {
          inCodeBlock = true;
          codeLang = line.slice(3).trim();
        }
        continue;
      }

      if (inCodeBlock) {
        codeContent += (codeContent ? "\n" : "") + line;
        continue;
      }

      if (line.startsWith("## ")) {
        elements.push(
          <h2 key={key++} className="text-lg font-bold text-foreground mt-8 mb-3 first:mt-0">
            {line.slice(3)}
          </h2>
        );
      } else if (line.startsWith("### ")) {
        elements.push(
          <h3 key={key++} className="text-base font-semibold text-foreground mt-6 mb-2">{line.slice(4)}</h3>
        );
      } else if (line.startsWith("- ") || line.startsWith("* ")) {
        elements.push(
          <li key={key++} className="text-sm text-foreground/85 ml-4 mb-1.5 list-disc leading-relaxed">
            {renderInlineFormatting(line.slice(2))}
          </li>
        );
      } else if (/^\d+\.\s/.test(line)) {
        const text = line.replace(/^\d+\.\s/, "");
        elements.push(
          <li key={key++} className="text-sm text-foreground/85 ml-4 mb-1.5 list-decimal leading-relaxed">
            {renderInlineFormatting(text)}
          </li>
        );
      } else if (line.trim() === "") {
        elements.push(<div key={key++} className="h-2" />);
      } else {
        elements.push(
          <p key={key++} className="text-sm text-foreground/85 leading-relaxed mb-2">
            {renderInlineFormatting(line)}
          </p>
        );
      }
    }

    return elements;
  };

  const renderInlineFormatting = (text: string) => {
    // Bold
    const parts = text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
    return parts.map((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        return <strong key={i} className="font-semibold text-foreground">{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith("*") && part.endsWith("*")) {
        return <em key={i} className="italic">{part.slice(1, -1)}</em>;
      }
      if (part.startsWith("`") && part.endsWith("`")) {
        return <code key={i} className="px-1.5 py-0.5 rounded bg-muted text-xs font-mono text-primary">{part.slice(1, -1)}</code>;
      }
      return part;
    });
  };

  return (
    <div className={cn("glass-card p-6 lg:p-8 animate-fade-in", className)}>
      {renderMarkdown(content)}
    </div>
  );
}
