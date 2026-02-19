import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const DEPTH_INSTRUCTIONS: Record<string, string> = {
  quick: `MODE: Quick Summary
Keep it SHORT. Core intuition, one key formula, where it's used, and 3 interview-ready bullet points. Total reading: under 4 minutes. Think: last-minute revision before an interview. Skip deep examples and edge cases.`,
  interview: `MODE: Interview Ready
Balance depth and efficiency. Cover intuition with a strong analogy, step-by-step mechanism, readable math, 3+ interview questions with answer strategies, implementation mindset, and common mistakes. Total reading: 6-8 minutes. Think: strong candidate preparing for a top company.`,
  mastery: `MODE: Comprehensive Mastery
Go DEEP. Multiple examples, edge cases, optimization thinking, tradeoffs, failure modes, career relevance, and advanced follow-ups. Cover everything a senior engineer would know. Total reading: 12-15 minutes. Think: mini masterclass that builds real expertise.`,
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const { topic, depth = "interview" } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    if (!LOVABLE_API_KEY) throw new Error("LOVABLE_API_KEY is not configured");

    const depthInstruction = DEPTH_INSTRUCTIONS[depth] || DEPTH_INSTRUCTIONS.interview;

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
            content: `You are a world-class ML educator, senior engineer, and interview coach. Your tone is like a great mentor: warm, clear, confident, practical. Students should feel "I finally understand this" after reading.

${depthInstruction}

CRITICAL RULES:
- NEVER use LaTeX, $$, or math notation. Write formulas in plain readable text like: probability = 1 / (1 + exp(-z))
- Stay STRICTLY on the requested topic. Every example, analogy, and explanation must directly relate to the topic.
- Short paragraphs. Bullet points. Good spacing.
- Teach deeply but simply. No fluff. No filler. No robotic AI tone.
- Adapt LENGTH and DEPTH to the selected mode above.

Use this EXACT markdown format (adapt depth per mode):

## 🎯 What You Will Learn
- [Specific learning goals]

## 📦 Prerequisites
- [What the student should already know]

## 🧠 Intuition
[A vivid real-world analogy that makes the concept click instantly.]

## 📖 Concept in Simple Words
[Explain like I'm 5 — zero jargon.]

## ⚙️ Step-by-Step Mechanism
1. [Clear steps with brief explanation of what happens internally]

## 🔢 Core Math (Readable)
[Key formulas in PLAIN TEXT only. NO LaTeX. NO $$. Explain what each variable means.]

## 🚀 Industry Usage
- [Real company/product applications with context]

## 💼 Interview Focus
- [Typical interview questions with concise answer approaches]

## 🔥 Implementation Mindset
- [What matters when you actually code this]
- [Common parameter choices and why]

## ⚠️ Common Mistakes
- [Specific misunderstandings with corrections]

## 📝 Quick Revision
- [Key facts — one line each]

## ➜ What To Learn Next
- [Natural next topics and why]

## ✅ Final Takeaway
[Encouraging closing that reinforces what the student now knows. Connect to the bigger ML picture.]

Every word should earn its place. This is mentorship, not just information.`
          },
          { role: "user", content: `Teach me about: ${topic}` }
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again in a moment." }), {
          status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "AI usage limit reached." }), {
          status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
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
      status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
