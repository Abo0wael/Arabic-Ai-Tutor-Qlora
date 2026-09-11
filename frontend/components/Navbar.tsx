"use client";

import React, { useEffect, useState } from "react";
import { Sparkles, Github, Activity, Wifi, WifiOff } from "lucide-react";
import { checkHealth, HealthStatus } from "@/lib/tutorApi";

export default function Navbar() {
  const [health, setHealth] = useState<{
    status: "checking" | "online" | "offline";
    data?: HealthStatus;
  }>({ status: "checking" });

  useEffect(() => {
    let mounted = true;

    async function probe() {
      const res = await checkHealth();
      if (!mounted) return;
      if (res.ok && res.data?.model_loaded) {
        setHealth({ status: "online", data: res.data });
      } else {
        setHealth({ status: "offline" });
      }
    }

    probe();
    const interval = setInterval(probe, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/[0.08] bg-space-950/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 via-violet-600 to-cyan-400 p-[1px] shadow-lg shadow-indigo-500/20">
            <div className="w-full h-full bg-space-900 rounded-xl flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-indigo-400" />
            </div>
          </div>
          <div>
            <span className="font-bold text-base sm:text-lg tracking-tight text-white flex items-center gap-2">
              Arabic AI Tutor
              <span className="hidden sm:inline-block text-xs font-normal text-slate-400">
                | المعلم الذكي
              </span>
            </span>
          </div>
        </div>

        {/* Right side: Status Indicator & GitHub */}
        <div className="flex items-center gap-3 sm:gap-4">
          {/* Status Badge */}
          <div
            className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium border transition-colors ${
              health.status === "online"
                ? "bg-emerald-950/40 border-emerald-500/30 text-emerald-300"
                : health.status === "offline"
                ? "bg-rose-950/40 border-rose-500/30 text-rose-300"
                : "bg-amber-950/40 border-amber-500/30 text-amber-300"
            }`}
          >
            {health.status === "online" ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="hidden sm:inline">متصل (RTX 5070)</span>
                <span className="sm:hidden">متصل</span>
              </>
            ) : health.status === "offline" ? (
              <>
                <WifiOff className="w-3 h-3 text-rose-400" />
                <span className="hidden sm:inline">النموذج غير متصل</span>
                <span className="sm:hidden">غير متصل</span>
              </>
            ) : (
              <>
                <Activity className="w-3 h-3 animate-spin text-amber-400" />
                <span>فحص الاتصال...</span>
              </>
            )}
          </div>

          {/* GitHub link */}
          <a
            href="https://github.com/Abo0wael/Arabic-Ai-Tutor-Qlora"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium text-slate-300 bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.08] transition-all hover:text-white"
          >
            <Github className="w-4 h-4" />
            <span className="hidden sm:inline">GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
}
