import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { action, topic, round, roundLabel, question, answer, previousRounds } = await req.json();
    const apiKey = Deno.env.get("LOVABLE_API_KEY");
    if (!apiKey) throw new Error("Missing LOVABLE_API_KEY");

    if (action === "ask") {
      const prevContext = previousRounds?.length
        ? `\nPrevious rounds:\n${previousRounds.map((r: any) => `Q: ${r.question}\nA: ${r.answer}\nScore: ${r.score}/10`).join("\n\n")}`
        : "";

      const prompt = `You are a senior FAANG ML interviewer conducting Round ${round}/5.
Topic: ${topic}
Round focus: ${roundLabel}
${prevContext}

Ask ONE precise, calibrated question for this round. Requirements:
- No hints or explanations
- FAANG interview difficulty
- Round ${round} focus: ${roundLabel}
- If previous answers were weak, probe that weakness
- If previous answers were strong, increase difficulty significantly

Return ONLY the question text, nothing else.`;

      const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: "google/gemini-3-flash-preview",
          messages: [{ role: "user", content: prompt }],
          temperature: 0.7,
          max_tokens: 300,
        }),
      });

      const data = await res.json();
      const questionText = data.choices?.[0]?.message?.content?.trim() || `Explain the core mechanism of ${topic}.`;
      return new Response(JSON.stringify({ question: questionText }), {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    if (action === "evaluate") {
      const prevContext = previousRounds?.length
        ? `\nPrevious rounds:\n${previousRounds.map((r: any) => `Q: ${r.question}\nA: ${r.answer}\nScore: ${r.score}/10`).join("\n\n")}`
        : "";

      const isLastRound = round >= 5;
      const nextRoundPrompt = isLastRound
        ? ""
        : `\nAlso generate the next question for Round ${round + 1} (focus: ${["Foundational", "Mechanism-level", "Edge cases/failure modes", "Production trade-offs/scaling", "System design/optimization"][round]}).
Adapt difficulty based on this answer's quality.`;

      const prompt = `You are a senior FAANG ML interviewer evaluating a candidate's response.
Topic: ${topic}
Round: ${round}/5 (${roundLabel})
${prevContext}

Question asked: ${question}
Candidate's answer: ${answer}

Evaluate rigorously. Do NOT inflate scores.

Scoring rubric:
0-3: Major misunderstandings
4-6: Basic but incomplete  
7-8: Strong conceptual clarity
9-10: Deep, production-level insight

Return a JSON object with this exact structure:
{
  "evaluation": {
    "correctConcepts": ["list of correct points mentioned"],
    "missingConcepts": ["critical concepts the candidate missed"],
    "misconceptions": ["any incorrect statements"],
    "depthLevel": "Surface|Intermediate|Strong|Expert",
    "score": <number 0-10>
  }${isLastRound ? '' : ',\n  "nextQuestion": "the next interview question"'}
}

Return ONLY valid JSON.`;

      const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify({
          model: "google/gemini-3-flash-preview",
          messages: [{ role: "user", content: prompt }],
          temperature: 0.4,
          max_tokens: 800,
        }),
      });

      const data = await res.json();
      let content = data.choices?.[0]?.message?.content?.trim() || "";
      
      // Extract JSON from markdown code blocks if present
      const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
      if (jsonMatch) content = jsonMatch[1].trim();

      try {
        const parsed = JSON.parse(content);
        return new Response(JSON.stringify(parsed), {
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      } catch {
        // Fallback
        return new Response(JSON.stringify({
          evaluation: {
            correctConcepts: ["Attempted response"],
            missingConcepts: ["Deeper analysis needed"],
            misconceptions: [],
            depthLevel: "Surface",
            score: 4,
          },
          nextQuestion: `Follow up on ${topic}: explain the ${["Foundational", "Mechanism-level", "Edge cases", "Production trade-offs", "System design"][Math.min(round, 4)]} aspects.`,
        }), { headers: { ...corsHeaders, "Content-Type": "application/json" } });
      }
    }

    return new Response(JSON.stringify({ error: "Unknown action" }), {
      status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    return new Response(JSON.stringify({ error: error.message }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
