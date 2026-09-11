import React from "react";
import { Github, Heart, Sparkles, Linkedin } from "lucide-react";

export default function Footer() {
  return (
    <footer className="w-full border-t border-white/[0.06] bg-space-950 py-8 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
        <div className="flex items-center gap-2">
          <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
          <span>Arabic AI Tutor — Fine-Tuned Qwen3-1.7B with QLoRA</span>
        </div>

        <div className="flex items-center gap-4">
          <a
            href="https://www.linkedin.com/in/ahmed-wael-9a6a5938a"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-slate-300 transition-colors flex items-center gap-1.5 text-slate-400"
          >
            <Linkedin className="w-3.5 h-3.5 text-[#0a66c2]" />
            <span>LinkedIn</span>
          </a>
          <span>•</span>
          <a
            href="https://github.com/Abo0wael"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-slate-300 transition-colors flex items-center gap-1.5"
          >
            <Github className="w-3.5 h-3.5" />
            <span>GitHub (Abo0wael)</span>
          </a>
          <span>•</span>
          <span>Ahmed Wael © 2026</span>
        </div>
      </div>
    </footer>
  );
}
