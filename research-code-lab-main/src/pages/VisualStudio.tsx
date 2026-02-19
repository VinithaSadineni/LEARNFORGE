import { useState, useCallback, useEffect } from "react";
import { Eye, Download, RefreshCw, ImageIcon } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import TopicInput from "@/components/TopicInput";
import CrossModuleLinks from "@/components/CrossModuleLinks";
import { trackActivity } from "@/lib/progression";
import { toast } from "@/hooks/use-toast";
import { Skeleton } from "@/components/ui/skeleton";

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

const SUGGESTIONS = ["Neural Network", "Backpropagation", "K-Means", "Decision Tree", "CNN Architecture"];

interface GeneratedImage {
  id: string;
  topic: string;
  imageUrl: string;
  description: string;
}

export default function VisualStudio() {
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTopic, setCurrentTopic] = useState("");
  const [searchParams, setSearchParams] = useSearchParams();

  const generate = useCallback(async (topic: string) => {
    setIsLoading(true);
    setCurrentTopic(topic);

    try {
      const resp = await fetch(`${SUPABASE_URL}/functions/v1/generate-visual`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${SUPABASE_KEY}`,
        },
        body: JSON.stringify({ topic }),
      });

      if (!resp.ok) {
        const errorData = await resp.json().catch(() => ({ error: "Request failed" }));
        throw new Error(errorData.error || `Error ${resp.status}`);
      }

      const data = await resp.json();

      if (data.imageUrl) {
        setImages((prev) => [
          {
            id: crypto.randomUUID(),
            topic,
            imageUrl: data.imageUrl,
            description: data.description || "",
          },
          ...prev,
        ]);
        trackActivity(topic, "visual");
        toast({ title: "Visual generated!" });
      } else {
        throw new Error("No image returned");
      }
    } catch (error) {
      console.error(error);
      toast({
        title: "AI temporarily unavailable",
        description: "Please try again in a moment.",
      });
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const topic = searchParams.get("topic");
    if (topic && images.length === 0 && !isLoading) {
      setSearchParams({}, { replace: true });
      generate(topic);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleDownload = (image: GeneratedImage) => {
    const a = document.createElement("a");
    a.href = image.imageUrl;
    a.download = `${image.topic.replace(/\s+/g, "-").toLowerCase()}-visual.png`;
    a.click();
    toast({ title: "Downloaded!" });
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <Eye className="h-5 w-5 text-primary" />
          <h1 className="text-xl font-bold text-foreground">Visualize Machine Learning</h1>
        </div>
        <p className="text-sm text-muted-foreground">
          Turn abstract ideas into clear visual understanding.
        </p>
      </div>

      <TopicInput
        placeholder="e.g., Neural Network, Backpropagation, K-Means..."
        onSubmit={generate}
        isLoading={isLoading}
        suggestions={SUGGESTIONS}
      />

      {isLoading && (
        <div className="glass-card-elevated p-4 animate-fade-in space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="h-5 w-5 rounded-full bg-primary/20 flex items-center justify-center">
              <ImageIcon className="h-3 w-3 text-primary animate-pulse" />
            </div>
            <span className="text-sm font-medium text-muted-foreground">Generating visual for "{currentTopic}"...</span>
          </div>
          <Skeleton className="w-full aspect-video rounded-lg" />
          <div className="flex gap-2">
            <Skeleton className="h-8 w-24 rounded-lg" />
            <Skeleton className="h-8 w-24 rounded-lg" />
          </div>
        </div>
      )}

      {images.length > 0 && (
        <div className="space-y-5">
          {images.map((image, i) => (
            <div
              key={image.id}
              className="glass-card-elevated overflow-hidden animate-fade-in"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <div className="relative group">
                <img
                  src={image.imageUrl}
                  alt={`${image.topic} visualization`}
                  className="w-full object-contain bg-muted/30 max-h-[600px]"
                />
                <div className="absolute inset-0 bg-foreground/0 group-hover:bg-foreground/5 transition-all" />
              </div>

              <div className="p-4 space-y-3">
                <div className="flex items-center gap-3 text-[11px] text-muted-foreground font-medium tracking-wide">
                  <span>Generated for: <span className="text-foreground">{image.topic}</span></span>
                  <span className="text-muted-foreground/40">·</span>
                  <span>AI Visual Engine</span>
                </div>
                <div className="flex items-start justify-between gap-3">
                  <div className="space-y-1 min-w-0">
                    <h3 className="text-sm font-bold text-foreground">{image.topic}</h3>
                    {image.description && (
                      <p className="text-xs text-muted-foreground line-clamp-2">{image.description}</p>
                    )}
                  </div>
                </div>

                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={() => handleDownload(image)}
                    className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-muted/60 text-foreground hover:bg-muted transition-all"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download
                  </button>
                  <button
                    onClick={() => generate(image.topic)}
                    disabled={isLoading}
                    className="inline-flex items-center gap-2 px-4 py-2 text-xs font-medium rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-all disabled:opacity-50"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? "animate-spin" : ""}`} />
                    Regenerate
                  </button>
                </div>

                {/* Cross-module links */}
                <div className="pt-2 border-t border-border/50">
                  <CrossModuleLinks topic={image.topic} currentModule="visual" />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {!isLoading && images.length === 0 && (
        <div className="glass-card p-12 text-center animate-fade-in">
          <div className="inline-flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10 mb-4">
            <ImageIcon className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-base font-bold text-foreground mb-1">No visuals yet</h3>
          <p className="text-sm text-muted-foreground">
            Enter an ML concept above to generate a diagram or visual representation.
          </p>
        </div>
      )}
    </div>
  );
}
