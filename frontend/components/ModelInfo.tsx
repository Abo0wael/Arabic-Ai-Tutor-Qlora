import React from "react";
import { Cpu, Layers, Database, Percent, CheckCircle2, HardDrive, Timer } from "lucide-react";

interface InfoCardProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  subtext?: string;
  badgeColor?: string;
}

function InfoCard({ icon, label, value, subtext }: InfoCardProps) {
  return (
    <div className="glass-card rounded-xl p-4 sm:p-5 flex flex-col justify-between">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs text-slate-400 font-medium">{label}</span>
        <div className="p-2 rounded-lg bg-white/[0.04] border border-white/[0.06] text-indigo-400">
          {icon}
        </div>
      </div>
      <div>
        <div className="text-lg sm:text-xl font-bold text-white mb-1 tracking-tight">
          {value}
        </div>
        {subtext && (
          <div className="text-xs text-slate-400">{subtext}</div>
        )}
      </div>
    </div>
  );
}

export default function ModelInfo() {
  const cards: InfoCardProps[] = [
    {
      icon: <Cpu className="w-4 h-4 text-cyan-400" />,
      label: "Base Model",
      value: "Qwen3-1.7B",
      subtext: "Pretrained weights as primary knowledge foundation",
    },
    {
      icon: <Layers className="w-4 h-4 text-indigo-400" />,
      label: "Fine-Tuning Method",
      value: "QLoRA (4-bit NF4)",
      subtext: "Double quant + bfloat16 compute",
    },
    {
      icon: <Database className="w-4 h-4 text-violet-400" />,
      label: "Tutoring Dataset",
      value: "501 Examples",
      subtext: "426 train / 50 val / 25 test splits",
    },
    {
      icon: <Percent className="w-4 h-4 text-emerald-400" />,
      label: "Trainable Parameters",
      value: "0.0932%",
      subtext: "~1.57M parameters adapted",
    },
    {
      icon: <CheckCircle2 className="w-4 h-4 text-amber-400" />,
      label: "Training Regime",
      value: "1 Epoch / 54 Steps",
      subtext: "Val loss: 2.5073 (No overfitting)",
    },
    {
      icon: <HardDrive className="w-4 h-4 text-rose-400" />,
      label: "Hardware",
      value: "RTX 5070 Laptop GPU",
      subtext: "7.96 GiB VRAM (~1.27 GiB active)",
    },
    {
      icon: <Timer className="w-4 h-4 text-teal-400" />,
      label: "Training Time",
      value: "~4.2 min",
      subtext: "Fast, efficient local convergence",
    },
  ];

  return (
    <section className="w-full max-w-5xl mx-auto px-4 sm:px-6 mb-16">
      <div className="text-center mb-8">
        <h3 className="text-xl sm:text-2xl font-bold text-white mb-2">
          المواصفات التقنية وتفاصيل التدريب
        </h3>
        <p className="text-xs sm:text-sm text-slate-400 max-w-xl mx-auto">
          تم تكييف سلوك النموذج عبر تدريب محلي منخفض التكلفة مع الاحتفاظ بقاعدة المعرفة الأصلية.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 sm:gap-4">
        {cards.map((card, idx) => (
          <InfoCard key={idx} {...card} />
        ))}
      </div>
    </section>
  );
}
