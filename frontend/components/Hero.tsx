import React from "react";
import { ArrowDown, Cpu, Sparkles, BookOpen, Layers, Zap } from "lucide-react";

export default function Hero() {
  return (
    <section className="relative pt-12 pb-16 md:pt-20 md:pb-24 overflow-hidden">
      {/* Background glow decorations */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[550px] h-[350px] bg-gradient-to-tr from-indigo-600/15 via-violet-600/20 to-cyan-500/15 blur-[120px] rounded-full pointer-events-none" />

      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
        {/* Top Mini Pill */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-indigo-950/60 border border-indigo-500/30 text-indigo-300 mb-6 shadow-inner backdrop-blur-sm">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
          <span>QLoRA Behavioral Adaptation (V3.2)</span>
        </div>

        {/* Main Title */}
        <h1 className="text-3xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white leading-tight mb-4">
          <span className="block mb-1">Arabic AI Tutor</span>
          <span className="bg-gradient-to-r from-indigo-400 via-violet-300 to-cyan-400 bg-clip-text text-transparent">
            Fine-Tuned Qwen3-1.7B with QLoRA
          </span>
        </h1>

        {/* Subtitle */}
        <p className="max-w-3xl mx-auto text-base sm:text-lg text-slate-300 leading-relaxed font-normal mb-8">
          Arabic-first AI tutoring with a fine-tuned Qwen3-1.7B model running locally on GPU.
          <br className="hidden sm:inline" />
          <span className="text-slate-400 text-sm sm:text-base mt-1 block">
            تم تكييف سلوك النموذج لتقديم شروحات تعليمية دقيقة ومبسطة، وتصحيح المفاهيم الخاطئة، مع أمثلة عملية باللغة العربية.
          </span>
        </p>

        {/* Badges Grid */}
        <div className="flex flex-wrap items-center justify-center gap-2.5 sm:gap-3 mb-10 max-w-2xl mx-auto">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-xs font-medium text-slate-200">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Qwen3-1.7B Base</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-xs font-medium text-slate-200">
            <Layers className="w-3.5 h-3.5 text-indigo-400" />
            <span>QLoRA (4-bit NF4)</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-xs font-medium text-slate-200">
            <BookOpen className="w-3.5 h-3.5 text-violet-400" />
            <span>Arabic NLP</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-xs font-medium text-slate-200">
            <Zap className="w-3.5 h-3.5 text-emerald-400" />
            <span>Local GPU (RTX 5070)</span>
          </div>

          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/[0.04] border border-white/[0.08] text-xs font-medium text-slate-200">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
            <span>0.093% Trainable Params</span>
          </div>
        </div>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <a
            href="#chat-section"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-7 py-3 rounded-xl bg-gradient-to-r from-indigo-600 via-violet-600 to-cyan-500 hover:from-indigo-500 hover:to-cyan-400 text-white font-semibold text-sm sm:text-base shadow-lg shadow-indigo-600/30 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <span>ابدأ التعلم</span>
            <ArrowDown className="w-4 h-4" />
          </a>

          <a
            href="https://github.com/Abo0wael/Arabic-Ai-Tutor-Qlora"
            target="_blank"
            rel="noopener noreferrer"
            className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-white/[0.05] hover:bg-white/[0.09] text-slate-200 hover:text-white border border-white/[0.1] font-medium text-sm sm:text-base transition-all"
          >
            <span>View Project on GitHub</span>
          </a>
        </div>
      </div>
    </section>
  );
}
