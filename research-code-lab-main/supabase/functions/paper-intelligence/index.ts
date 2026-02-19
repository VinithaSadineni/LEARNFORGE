import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

type SectionType =
  | "interview"
  | "implementation"
  | "production"
  | "history"
  | "empty_suggestions";

const systemPrompts: Record<SectionType, string> = {
  interview: `You are a FAANG ML interviewer. Given a research paper title and its core contribution, generate interview preparation content. Return JSON:
{
  "conceptual": ["3 precise conceptual questions"],
  "systemDesign": "1 system design angle (1 sentence)",
  "tradeoff": "1 practical tradeoff question",
  "strongAnswer": ["3-4 bullet points of what a strong answer includes"]
}
Be extremely concise. No fluff. FAANG-level precision. Each question should test deep understanding, not recall.`,

  implementation: `You are a Staff ML engineer. Given a paper title and method description, generate implementation guidance. Return JSON:
{
  "steps": ["5-7 high-level algorithm steps"],
  "dataStructures": ["key data structures involved"],
  "timeComplexity": "time complexity with brief explanation",
  "spaceComplexity": "space complexity with brief explanation",
  "optimizations": ["2-3 optimization notes"]
}
Engineer clarity only. No full code. No pseudocode longer than 1 line per step.`,

  production: `You are a Staff ML engineer who has deployed models at scale. Given a paper title and method, explain production reality. Return JSON:
{
  "scaling": "scaling challenge (2 sentences max)",
  "dataSkew": "data skew issue (2 sentences max)",
  "monitoring": ["2-3 monitoring metrics to track"],
  "failureModes": ["2-3 failure modes"],
  "tradeoffs": "key operational tradeoff (2 sentences max)"
}
Sound like someone who has been paged at 3am because of this system.`,

  history: `You are an ML historian and senior researcher. Given a paper title and its contribution, explain its historical significance. Return JSON:
{
  "before": "What existed before this paper (2-3 sentences)",
  "solved": "What problem it solved (2 sentences)",
  "unlocked": "What it unlocked for the field (2 sentences)",
  "legacy": "How modern ML depends on it (2 sentences)"
}
Story-driven. Insightful. High credibility. No marketing language.`,

  empty_suggestions: `You are an ML research advisor. A student searched for a topic but no papers matched. Given their query, suggest related topics they should explore. Return JSON:
{
  "relatedTopics": ["4-5 related ML topics"],
  "broaderTerms": ["2-3 broader search terms"],
  "foundationalPapers": ["2-3 foundational paper titles to explore"]
}
Be helpful and specific to ML/AI domain.`,
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { section, paperTitle, paperContext, mode, query } = await req.json();

    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const sectionType = section as SectionType;
    const systemPrompt = systemPrompts[sectionType];
    if (!systemPrompt) throw new Error(`Unknown section: ${section}`);

    const modePrefix =
      mode === "deep"
        ? "Provide deep technical detail. Include complexity analysis, edge cases, and nuanced tradeoffs. "
        : "Keep it concise and accessible. Focus on key insights. ";

    const userContent =
      sectionType === "empty_suggestions"
        ? `Search query: "${query}"`
        : `Paper: "${paperTitle}"\nContext: ${paperContext}`;

    const response = await fetch(
      "https://ai.gateway.lovable.dev/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${LOVABLE_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "google/gemini-3-flash-preview",
          messages: [
            { role: "system", content: modePrefix + systemPrompt },
            { role: "user", content: userContent },
          ],
          response_format: { type: "json_object" },
        }),
      }
    );

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(
          JSON.stringify({ error: "Rate limited. Please try again shortly." }),
          { status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      if (response.status === 402) {
        return new Response(
          JSON.stringify({ error: "AI credits exhausted." }),
          { status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" } }
        );
      }
      const t = await response.text();
      console.error("AI gateway error:", response.status, t);
      throw new Error("AI gateway error");
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content;

    let parsed;
    try {
      parsed = JSON.parse(content);
    } catch {
      parsed = { raw: content };
    }

    return new Response(JSON.stringify(parsed), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("paper-intelligence error:", e);
    return new Response(
      JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
