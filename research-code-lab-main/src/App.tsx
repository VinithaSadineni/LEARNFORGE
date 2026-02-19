import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import AppLayout from "@/components/AppLayout";
import Dashboard from "@/pages/Dashboard";
import TextLearning from "@/pages/TextLearning";
import CodeGeneration from "@/pages/CodeGeneration";
import AudioLessons from "@/pages/AudioLessons";
import VisualStudio from "@/pages/VisualStudio";
import Roadmap from "@/pages/Roadmap";
import InterviewSimulator from "@/pages/InterviewSimulator";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <AppLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/text-learning" element={<TextLearning />} />
            <Route path="/code-generation" element={<CodeGeneration />} />
            <Route path="/audio-lessons" element={<AudioLessons />} />
            <Route path="/visual-studio" element={<VisualStudio />} />
            <Route path="/roadmap" element={<Roadmap />} />
            <Route path="/interview" element={<InterviewSimulator />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </AppLayout>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
