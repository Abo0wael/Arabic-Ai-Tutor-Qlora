import React from "react";
import { AlertTriangle } from "lucide-react";

export default function Disclaimer() {
  return (
    <section className="w-full max-w-4xl mx-auto px-4 sm:px-6 mb-16">
      <div className="rounded-2xl p-5 bg-white/[0.02] border border-white/[0.06] text-center text-xs sm:text-sm text-slate-400 space-y-2">
        <div className="inline-flex items-center gap-1.5 text-amber-400/90 font-medium text-xs mb-1">
          <AlertTriangle className="w-3.5 h-3.5" />
          <span>إخلاء مسؤولية تجريبي | Experimental Disclaimer</span>
        </div>
        <p className="leading-relaxed">
          هذا نموذج لغوي تجريبي تم تكييفه لأغراض تعليمية واستعراضية. قد تحتوي الإجابات على معلومات غير دقيقة ويجب عدم الاعتماد عليها كنصيحة مهنية أو أكاديمية قاطعة.
        </p>
        <p className="text-[11px] text-slate-500 font-mono">
          &quot;This is an experimental fine-tuned language model. Responses may contain inaccuracies and should not be treated as authoritative professional advice.&quot;
        </p>
      </div>
    </section>
  );
}
