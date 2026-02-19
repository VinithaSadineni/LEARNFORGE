import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { topic, action, conversation } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const systemPrompt = `You are a senior ML interview coach at a top tech company (Google/Meta/OpenAI level).

Your role: Conduct a realistic, adaptive ML interview on the topic "${topic}".

RULES:
- Ask ONE question at a time
- Start with fundamentals, progressively go deeper based on answers
- After each candidate answer, give brief feedback (1-2 sentences) then ask the next question
- If the answer is wrong, gently correct and probe understanding
- If the answer is strong, acknowledge and go deeper
- Mix conceptual, mathematical, and practical questions
- After 4-5 exchanges, provide a brief performance summary with strengths and areas to improve
- Be encouraging but honest
- Keep responses concise — this is a conversation, not a lecture
- Never use LaTeX. Write math in plain text.

TONE: Professional, warm, like a real interviewer who wants you to succeed.`;

    const messages: Array<{role: string; content: string}> = [
      { role: "system", content: systemPrompt },
    ];

    if (action === "start") {
      messages.push({ role: "user", content: `Start the interview on "${topic}". Briefly set the scene and ask your first question.` });
    } else {
      messages.push({ role: "user", content: `Here is the conversation so far:\n\n${conversation}\n\nContinue the interview. Give brief feedback on the last answer, then ask the next question. If we've had 4-5 exchanges, wrap up with a performance summary.` });
    }

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages,
        stream: true,
      }),
    });

    if (!response.ok) {
      const t = await response.text();
      console.error("AI error:", response.status, t);
      throw new Error("AI gateway error");
    }

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (e) {
    console.error("Error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
