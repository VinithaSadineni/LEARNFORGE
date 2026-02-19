// localStorage-based user progression tracking

const STORAGE_KEY = "learnforge_progression";

export interface ProgressionEntry {
  topic: string;
  module: "text" | "code" | "audio" | "visual";
  timestamp: number;
  depth?: string;
  quizScore?: { correct: number; total: number };
}

export interface ProgressionData {
  entries: ProgressionEntry[];
  streak: number;
  lastActiveDate: string; // YYYY-MM-DD
}

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function load(): ProgressionData {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { entries: [], streak: 0, lastActiveDate: "" };
}

function save(data: ProgressionData) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
}

export function trackActivity(topic: string, module: ProgressionEntry["module"], extra?: Partial<ProgressionEntry>) {
  const data = load();
  const today = getToday();

  // Update streak
  if (data.lastActiveDate !== today) {
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayStr = yesterday.toISOString().slice(0, 10);
    data.streak = data.lastActiveDate === yesterdayStr ? data.streak + 1 : 1;
    data.lastActiveDate = today;
  }

  data.entries.push({
    topic,
    module,
    timestamp: Date.now(),
    ...extra,
  });

  // Keep last 200 entries max
  if (data.entries.length > 200) {
    data.entries = data.entries.slice(-200);
  }

  save(data);
}

export function getProgression(): ProgressionData {
  const data = load();
  // Validate streak is still current
  const today = getToday();
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  const yesterdayStr = yesterday.toISOString().slice(0, 10);
  if (data.lastActiveDate !== today && data.lastActiveDate !== yesterdayStr) {
    data.streak = 0;
  }
  return data;
}

export function getRecentTopics(limit = 8): { topic: string; module: ProgressionEntry["module"]; timestamp: number }[] {
  const data = load();
  const seen = new Set<string>();
  const result: { topic: string; module: ProgressionEntry["module"]; timestamp: number }[] = [];
  for (let i = data.entries.length - 1; i >= 0 && result.length < limit; i--) {
    const key = `${data.entries[i].topic}-${data.entries[i].module}`;
    if (!seen.has(key)) {
      seen.add(key);
      result.push({ topic: data.entries[i].topic, module: data.entries[i].module, timestamp: data.entries[i].timestamp });
    }
  }
  return result;
}

export function getStats() {
  const data = getProgression();
  const uniqueTopics = new Set(data.entries.map((e) => e.topic)).size;
  const codingProblems = data.entries.filter((e) => e.module === "code").length;
  const quizzesTaken = data.entries.filter((e) => e.quizScore).length;
  return { uniqueTopics, codingProblems, quizzesTaken, streak: data.streak, totalSessions: data.entries.length };
}

export function saveQuizResult(topic: string, correct: number, total: number) {
  const data = load();
  // Find most recent text entry for this topic and attach quiz result
  for (let i = data.entries.length - 1; i >= 0; i--) {
    if (data.entries[i].topic === topic && data.entries[i].module === "text" && !data.entries[i].quizScore) {
      data.entries[i].quizScore = { correct, total };
      break;
    }
  }
  save(data);
}
