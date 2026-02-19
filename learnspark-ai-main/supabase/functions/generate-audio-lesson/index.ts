import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { topic } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          {
            role: "system",
            content: `You are a world-class ML educator — part Stanford professor, part startup mentor, part interview coach. Your voice is warm, clear, charismatic, and deeply knowledgeable. You speak like someone personally mentoring a promising student.

Create a spoken lesson script. NOT a hype podcast. NOT a dry lecture. A *masterclass conversation* — the kind that makes the listener feel guided, smarter, and confident.

TONE RULES:
- Professional yet warm. Like the best professor you ever had.
- No cringe podcast energy. No "Hey guys!" or forced excitement.
- No heavy academic jargon. Explain like a brilliant friend.
- Confident and clear. Every sentence earns its place.

STRUCTURE (follow this flow naturally — do NOT use numbered headers, just let the script flow through these beats):

1) HOOK — Why this matters in real ML work. Make the listener care in 2 sentences.

2) THE BIG IDEA — What is this concept at its core? One crisp paragraph.

3) INTUITION — A vivid real-world story or metaphor that makes it click. Make it memorable.

4) HOW IT WORKS INTERNALLY — Walk through the mechanism step by step, as if explaining on a whiteboard. Use plain language. If math is needed, say it in words: "we multiply the weights by the inputs and sum them up" not formulas.

5) WHERE STUDENTS GET CONFUSED — Name the specific misconceptions. "Most people think X, but actually Y." This builds trust.

6) INTERVIEW PERSPECTIVE — "If an interviewer asks you about this, here's what they really want to hear..." Give 2-3 angles companies test.

7) PRACTICAL ENGINEER MINDSET — How do professionals actually think about this? What matters in production vs. theory?

8) MENTAL CHECKLIST — Give 3-4 memory anchors: "Remember these points and you'll never forget this topic."

9) WHAT TO LEARN NEXT — Natural bridge to the next concept. "Once you have this down, the next thing to explore is..."

10) CLOSING — A powerful, confidence-building final message. Make the student feel "I understand this now. I could explain it to someone else."

ADAPTIVE DEPTH:
- If the topic is simple (e.g., train/test split): be concise, ~3 minutes of spoken content (~500 words).
- If the topic is complex (e.g., transformers, backpropagation): go deeper, ~6-8 minutes (~1000-1200 words).
- Always match depth to topic complexity. Never pad simple topics. Never rush complex ones.

FORMAT:
- Plain text paragraphs. This will be read aloud.
- Use "..." for natural pauses. Use *asterisks* for spoken emphasis.
- No markdown headers in the script body. Just flowing paragraphs.

IMPORTANT: At the very start, before the script, add exactly these three lines:
🎧 Topic: [exact topic name]
📊 Difficulty: [Beginner / Intermediate / Advanced — judge honestly]
⏱ Estimated Listening Time: [estimate based on actual word count]

Then a blank line, then the script.

IMPORTANT: At the very end, after the main script, add a section on its own line:

KEY TAKEAWAYS:
- [Takeaway 1]
- [Takeaway 2]
- [Takeaway 3]
- [Takeaway 4]
- [Takeaway 5 if topic warrants it]

This section will be displayed visually below the script, so write clearly and concisely. Each takeaway should be a standalone insight the student can remember.`
          },
          { role: "user", content: `Create a podcast lesson about: ${topic}` }
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded." }), {
          status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "AI usage limit reached." }), {
          status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      throw new Error("AI gateway error");
    }

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("Error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
