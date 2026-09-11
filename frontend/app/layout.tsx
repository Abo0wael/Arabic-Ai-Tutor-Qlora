import type { Metadata } from "next";
import { Cairo } from "next/font/google";
import "./globals.css";

const cairo = Cairo({
  subsets: ["arabic", "latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  variable: "--font-cairo",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Arabic AI Tutor | معلم الذكاء الاصطناعي العربي",
  description:
    "معلم ذكاء اصطناعي عربي تم تدريبه الدقيق بتقنية QLoRA على نموذج Qwen3-1.7B لتقديم شروحات مبسطة وأمثلة واقعية.",
  keywords: [
    "Arabic AI",
    "AI Tutor",
    "Qwen3",
    "QLoRA",
    "Fine-Tuning",
    "Arabic NLP",
    "ذكاء اصطناعي",
  ],
  authors: [{ name: "Ahmed Wael", url: "https://github.com/Abo0wael" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ar" dir="rtl" className={cairo.variable}>
      <body className="bg-space-950 text-slate-100 min-h-screen selection:bg-indigo-500/30 selection:text-indigo-200">
        <div className="fixed inset-0 bg-grid-pattern pointer-events-none z-0 opacity-40" />
        <div className="relative z-10 flex flex-col min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
