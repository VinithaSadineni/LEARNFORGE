import { useState, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";

export type SectionType = "interview" | "implementation" | "production" | "history" | "empty_suggestions";

interface CacheEntry {
  data: any;
  timestamp: number;
}

const cache = new Map<string, CacheEntry>();
const CACHE_TTL = 1000 * 60 * 30; // 30 minutes

export function usePaperIntelligence() {
  const [loading, setLoading] = useState<Record<string, boolean>>({});
  const [results, setResults] = useState<Record<string, any>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});

  const fetchSection = useCallback(
    async (
      section: SectionType,
      paperTitle: string,
      paperContext: string,
      mode: "summary" | "deep" = "summary",
      query?: string
    ) => {
      const cacheKey = `${section}:${paperTitle}:${mode}:${query || ""}`;

      // Check cache
      const cached = cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
        setResults((prev) => ({ ...prev, [cacheKey]: cached.data }));
        return cached.data;
      }

      setLoading((prev) => ({ ...prev, [cacheKey]: true }));
      setErrors((prev) => {
        const next = { ...prev };
        delete next[cacheKey];
        return next;
      });

      try {
        const { data, error } = await supabase.functions.invoke("paper-intelligence", {
          body: { section, paperTitle, paperContext, mode, query },
        });

        if (error) throw error;

        cache.set(cacheKey, { data, timestamp: Date.now() });
        setResults((prev) => ({ ...prev, [cacheKey]: data }));
        return data;
      } catch (e: any) {
        const msg = e?.message || "Failed to generate content";
        setErrors((prev) => ({ ...prev, [cacheKey]: msg }));
        return null;
      } finally {
        setLoading((prev) => ({ ...prev, [cacheKey]: false }));
      }
    },
    []
  );

  const getKey = (section: SectionType, title: string, mode: string, query?: string) =>
    `${section}:${title}:${mode}:${query || ""}`;

  return { fetchSection, loading, results, errors, getKey };
}
