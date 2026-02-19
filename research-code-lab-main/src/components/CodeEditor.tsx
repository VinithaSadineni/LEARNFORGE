import { useRef, useEffect, useState, useCallback } from "react";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
}

export function CodeEditor({ value, onChange }: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const lineNumberRef = useRef<HTMLDivElement>(null);
  const [lineCount, setLineCount] = useState(1);

  const updateLines = useCallback(() => {
    const lines = value.split("\n").length;
    setLineCount(lines);
  }, [value]);

  useEffect(() => {
    updateLines();
  }, [updateLines]);

  const handleScroll = () => {
    if (textareaRef.current && lineNumberRef.current) {
      lineNumberRef.current.scrollTop = textareaRef.current.scrollTop;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const ta = textareaRef.current;
      if (!ta) return;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const newVal = value.substring(0, start) + "    " + value.substring(end);
      onChange(newVal);
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 4;
      });
    }
  };

  return (
    <div className="flex w-full h-full" style={{ background: "hsl(220 22% 5%)" }}>
      {/* Line numbers */}
      <div
        ref={lineNumberRef}
        className="select-none overflow-hidden shrink-0 py-4 text-right pr-3 pl-3"
        style={{
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: "13px",
          lineHeight: "1.7",
          color: "hsl(220 10% 30%)",
          borderRight: "1px solid hsl(220 14% 12%)",
          minWidth: "3.5rem",
        }}
      >
        {Array.from({ length: lineCount }, (_, i) => (
          <div key={i + 1}>{i + 1}</div>
        ))}
      </div>
      {/* Code area */}
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={handleScroll}
        onKeyDown={handleKeyDown}
        spellCheck={false}
        className="flex-1 h-full py-4 px-4 resize-none outline-none"
        style={{
          background: "transparent",
          color: "hsl(175 60% 70%)",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: "13px",
          lineHeight: "1.7",
          tabSize: 4,
          caretColor: "hsl(175 70% 50%)",
        }}
      />
    </div>
  );
}
