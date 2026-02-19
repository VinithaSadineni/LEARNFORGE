import { useState, useEffect, useCallback } from "react";

const STORAGE_KEY = "learnforge_progress";

export interface Progress {
  solvedProblems: string[];
  viewedPapers: string[];
  problemsOpened: number;
  lastActivity: string;
}

function loadProgress(): Progress {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { solvedProblems: [], viewedPapers: [], problemsOpened: 0, lastActivity: "" };
}

function saveProgress(p: Progress) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(p));
}

export function useProgress() {
  const [progress, setProgress] = useState<Progress>(loadProgress);

  useEffect(() => {
    saveProgress(progress);
  }, [progress]);

  const markSolved = useCallback((problemId: string) => {
    setProgress((prev) => {
      if (prev.solvedProblems.includes(problemId)) return prev;
      return { ...prev, solvedProblems: [...prev.solvedProblems, problemId], lastActivity: new Date().toISOString() };
    });
  }, []);

  const markPaperViewed = useCallback((paperId: string) => {
    setProgress((prev) => {
      if (prev.viewedPapers.includes(paperId)) return prev;
      return { ...prev, viewedPapers: [...prev.viewedPapers, paperId], lastActivity: new Date().toISOString() };
    });
  }, []);

  const incrementOpened = useCallback(() => {
    setProgress((prev) => ({ ...prev, problemsOpened: prev.problemsOpened + 1, lastActivity: new Date().toISOString() }));
  }, []);

  const getStreak = useCallback(() => {
    if (!progress.lastActivity) return 0;
    const last = new Date(progress.lastActivity);
    const now = new Date();
    const diffDays = Math.floor((now.getTime() - last.getTime()) / (1000 * 60 * 60 * 24));
    if (diffDays > 1) return 0;
    return Math.max(1, progress.solvedProblems.length > 0 ? Math.min(progress.solvedProblems.length, 30) : 1);
  }, [progress]);

  return { progress, markSolved, markPaperViewed, incrementOpened, getStreak };
}
