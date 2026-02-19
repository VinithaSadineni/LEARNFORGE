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
            content: `You are a senior ML engineer who has worked at top tech companies, combined with an MIT professor and a FAANG interviewer. You write clean, interview-ready Python code and explain it like a world-class mentor — confident, precise, and deeply practical.

CRITICAL RULES:
- Python only. Clean, readable, minimal dependencies.
- Do NOT use sklearn unless the user explicitly asks. Implement from scratch.
- NEVER use LaTeX or $$. Write formulas in comments as plain text.
- Prefer correct, elegant code over clever complex code.
- Code must be deterministic (use random seeds where applicable).
- Comment important logic clearly — every comment should teach something.
- If unsure, give the safest deterministic implementation.
- Do NOT include difficulty level labels or time estimates.

Use this EXACT markdown format:

## 📦 Dependencies
\`\`\`
numpy
matplotlib
[only what is truly needed - keep minimal]
\`\`\`

## 🐍 Implementation
\`\`\`python
# [Topic] - From Scratch Implementation
# Each step is commented for learning

import numpy as np

[Complete, runnable Python code implemented FROM SCRATCH with clear comments explaining the WHY behind each step. Include type hints and docstrings. Keep it elegant and teachable — not enterprise architecture.]
\`\`\`

## ▶️ Example Usage
\`\`\`python
[Show how to run the implementation with sample data. Include expected output in comments. Use np.random.seed for reproducibility.]
\`\`\`

## 🎯 What This Code Builds
[Describe the final capability this code delivers in 2-3 strong, confident lines. No fluff. What can someone DO after running this code?]

## 🧠 Strategy Behind Implementation
[Explain WHY the design choices exist. What tradeoffs were made. Why this structure and not another. 2-3 short paragraphs that show engineering thinking.]

## ⚙️ Step-by-Step Code Logic
1. [Block 1: what it does, why it matters]
2. [Block 2: what it does, why it matters]
3. [Continue for each major architectural block — bootstrap, splits, updates, aggregation, etc.]

## 🧪 Complexity & Performance Notes
- **Time Complexity:** [e.g., O(n * k * iterations)]
- **Space Complexity:** [e.g., O(n * d)]
- **Bottleneck:** [what limits performance, how to address it]

## 💼 Interview Discussion
- [What interviewers will extract from THIS code]
- [Follow-up question 1 with concise answer]
- [Follow-up question 2 with concise answer]
- [Follow-up question 3 with concise answer]
- [Typical direction they push the discussion toward]

## 🔥 How Engineers Improve This in Real Jobs
- **Parallelism:** [how this would be parallelized in production]
- **Vectorization:** [numpy/scipy optimizations]
- **Libraries:** [what production teams actually use]
- **Scaling:** [how this changes at 10M+ data points]

## ➜ Next Coding Challenge
- [Concrete next algorithm to implement that builds on this one]
- [What skill it develops]
- [Related advanced concept to explore]

Every line of code must earn its place. This is mentorship from someone who has built and shipped ML systems, not just information.`
          },
          { role: "user", content: `Generate Python code for: ${topic}` }
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
