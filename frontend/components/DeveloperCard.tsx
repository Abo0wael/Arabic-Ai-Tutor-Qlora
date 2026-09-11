import React from "react";
import { Github, Linkedin, ExternalLink } from "lucide-react";

export default function DeveloperCard() {
  return (
    <section className="w-full max-w-2xl mx-auto px-4 sm:px-6 mb-20 text-center">
      <div className="glass-panel rounded-2xl sm:rounded-3xl p-6 sm:p-8 border border-white/[0.08] relative overflow-hidden shadow-xl">
        {/* Glow decoration */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-72 h-40 bg-indigo-500/10 blur-[80px] rounded-full pointer-events-none" />

        <div className="relative z-10">
          <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-cyan-400 p-[1px] shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-space-900 rounded-2xl flex items-center justify-center text-lg font-bold text-white">
              AW
            </div>
          </div>

          <h3 className="text-lg sm:text-xl font-bold text-white mb-1">
            Ahmed Wael (Abo0wael)
          </h3>
          <p className="text-xs sm:text-sm text-slate-400 mb-6">
            AI & Machine Learning Engineer • Creator of Arabic AI Tutor
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3 sm:gap-4">
            {/* LinkedIn Button */}
            <a
              href="https://www.linkedin.com/in/ahmed-wael-9a6a5938a"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-[#0a66c2]/20 hover:bg-[#0a66c2]/30 border border-[#0a66c2]/40 text-white text-xs sm:text-sm font-semibold transition-all hover:scale-105 active:scale-95 shadow-md shadow-[#0a66c2]/10"
            >
              <Linkedin className="w-4 h-4 text-[#0a66c2]" />
              <span>LinkedIn Profile</span>
              <ExternalLink className="w-3 h-3 opacity-60" />
            </a>

            {/* GitHub Profile Button */}
            <a
              href="https://github.com/Abo0wael"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-white/[0.05] hover:bg-white/[0.1] border border-white/[0.1] text-white text-xs sm:text-sm font-semibold transition-all hover:scale-105 active:scale-95 shadow-md"
            >
              <Github className="w-4 h-4 text-slate-200" />
              <span>GitHub Profile</span>
              <ExternalLink className="w-3 h-3 opacity-60" />
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
