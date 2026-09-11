"use client";

import React, { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import {
  Send,
  Sparkles,
  User,
  RotateCcw,
  Copy,
  Check,
  Clock,
  Zap,
  WifiOff,
  AlertCircle,
  HelpCircle,
} from "lucide-react";
import { generateTutorAnswer, checkHealth } from "@/lib/tutorApi";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  generationTime?: number;
  tokens?: number;
  isError?: boolean;
}

const SUGGESTED_QUESTIONS = [
  "اشرح لي Overfitting بطريقة بسيطة",
  "ما الفرق بين Machine Learning و Deep Learning؟",
  "اشرح Self-Attention بمثال",
  "ما معنى Hallucination في النماذج اللغوية؟",
  "اشرح RAG لطفل في العاشرة",
  "لدي Train Accuracy عالية و Test Accuracy منخفضة، ما المشكلة؟",
];

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [isBackendOffline, setIsBackendOffline] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Auto-scroll when messages change or while loading
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Loading elapsed timer
  useEffect(() => {
    if (isLoading) {
      setElapsedSeconds(0);
      timerRef.current = setInterval(() => {
        setElapsedSeconds((prev) => +(prev + 0.1).toFixed(1));
      }, 100);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isLoading]);

  // Send question handler
  const handleSend = async (textToSend?: string) => {
    const query = (textToSend ?? input).trim();
    if (!query || isLoading) return;

    const userMessage: Message = {
      id: "user-" + Date.now(),
      role: "user",
      content: query,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setIsBackendOffline(false);

    // Prepare conversation history for multi-turn if available
    const historyPayload = messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    const result = await generateTutorAnswer(query, historyPayload);

    if (result.success) {
      const assistantMessage: Message = {
        id: "ai-" + Date.now(),
        role: "assistant",
        content: result.data.response,
        generationTime: result.data.generation_time,
        tokens: result.data.tokens_generated,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } else {
      if (result.error.isOffline) {
        setIsBackendOffline(true);
      }
      const errorMessage: Message = {
        id: "err-" + Date.now(),
        role: "assistant",
        content: result.error.message,
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    }

    setIsLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch {
      // ignore
    }
  };

  const clearChat = () => {
    setMessages([]);
    setIsBackendOffline(false);
  };

  return (
    <section id="chat-section" className="w-full max-w-4xl mx-auto px-4 sm:px-6 mb-20">
      {/* Main Glass Chat Container */}
      <div className="glass-panel rounded-2xl sm:rounded-3xl shadow-2xl shadow-indigo-950/40 overflow-hidden flex flex-col min-h-[560px] border border-white/[0.08]">
        {/* Chat Header Bar */}
        <div className="px-5 py-4 border-b border-white/[0.06] bg-space-900/60 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-indigo-400" />
            </div>
            <div>
              <h2 className="text-sm sm:text-base font-bold text-white">
                جلسة التعلم التفاعلية
              </h2>
              <p className="text-xs text-slate-400">
                إجابات مركزة، أمثلة وتصحيح مفاهيم الذكاء الاصطناعي
              </p>
            </div>
          </div>

          {messages.length > 0 && (
            <button
              onClick={clearChat}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-slate-200 bg-white/[0.03] hover:bg-white/[0.07] border border-white/[0.06] transition-all"
              title="إعادة ضبط المحادثة"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>محادثة جديدة</span>
            </button>
          )}
        </div>

        {/* Offline Banner when server is unreachable */}
        {isBackendOffline && (
          <div className="bg-rose-950/40 border-b border-rose-500/20 px-5 py-3 flex items-start gap-3 text-rose-300 text-xs sm:text-sm">
            <WifiOff className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
            <div>
              <p className="font-semibold mb-0.5">النموذج غير متصل حاليًا.</p>
              <p className="text-rose-300/80 leading-relaxed">
                هذا الديمو يعمل من جهاز محلي (Local RTX 5070 GPU)، وقد يكون غير متاح عندما يكون الجهاز مغلقًا أو لم يتم تشغيل نفق Cloudflare Tunnel بعد.
              </p>
            </div>
          </div>
        )}

        {/* Messages Body */}
        <div className="flex-1 p-4 sm:p-6 overflow-y-auto space-y-6 min-h-[380px]">
          {messages.length === 0 ? (
            /* Suggested Questions State */
            <div className="h-full flex flex-col justify-center py-6">
              <div className="text-center mb-6">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-3">
                  <HelpCircle className="w-6 h-6" />
                </div>
                <h3 className="text-base sm:text-lg font-bold text-white mb-1">
                  ماذا تريد أن تتعلم اليوم؟
                </h3>
                <p className="text-xs sm:text-sm text-slate-400 max-w-md mx-auto">
                  اختر سؤالاً من الأسئلة المقترحة أدناه أو اكتب سؤالك الخاص في الحقل بالأسفل:
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-2xl mx-auto w-full">
                {SUGGESTED_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(q)}
                    className="glass-card text-right p-3.5 rounded-xl text-xs sm:text-sm text-slate-200 hover:text-white group flex items-center justify-between gap-3"
                  >
                    <span className="leading-relaxed">{q}</span>
                    <Sparkles className="w-3.5 h-3.5 text-indigo-400 opacity-0 group-hover:opacity-100 transition-opacity shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* Conversation Messages */
            messages.map((m) => (
              <div
                key={m.id}
                className={`flex gap-3 sm:gap-4 ${
                  m.role === "user" ? "flex-row-reverse" : "flex-row"
                }`}
              >
                {/* Avatar */}
                <div
                  className={`w-8 h-8 rounded-xl shrink-0 flex items-center justify-center text-xs font-semibold ${
                    m.role === "user"
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                      : m.isError
                      ? "bg-rose-900/60 border border-rose-500/30 text-rose-300"
                      : "bg-space-850 border border-white/[0.1] text-cyan-400 shadow-md"
                  }`}
                >
                  {m.role === "user" ? (
                    <User className="w-4 h-4" />
                  ) : (
                    <Sparkles className="w-4 h-4" />
                  )}
                </div>

                {/* Message Bubble */}
                <div
                  className={`max-w-[85%] sm:max-w-[78%] rounded-2xl px-4 sm:px-5 py-3.5 text-xs sm:text-sm leading-relaxed ${
                    m.role === "user"
                      ? "bg-gradient-to-r from-indigo-600 to-violet-600 text-white rounded-tr-sm shadow-md"
                      : m.isError
                      ? "bg-rose-950/30 border border-rose-500/30 text-rose-200 rounded-tl-sm"
                      : "bg-space-900/80 border border-white/[0.08] text-slate-100 rounded-tl-sm shadow-md"
                  }`}
                >
                  {/* Markdown or plain text */}
                  {m.role === "assistant" && !m.isError ? (
                    <div className="prose-rtl text-slate-100 space-y-2">
                      <ReactMarkdown>{m.content}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="whitespace-pre-wrap">{m.content}</p>
                  )}

                  {/* Metadata and Copy Button on Assistant Bubble */}
                  {m.role === "assistant" && !m.isError && (
                    <div className="mt-3 pt-2.5 border-t border-white/[0.06] flex items-center justify-between text-[11px] text-slate-400">
                      <div className="flex items-center gap-3">
                        {m.generationTime && (
                          <span className="flex items-center gap-1 text-slate-400">
                            <Clock className="w-3 h-3 text-cyan-400" />
                            {m.generationTime} ث
                          </span>
                        )}
                        {m.tokens && (
                          <span className="flex items-center gap-1 text-slate-400">
                            <Zap className="w-3 h-3 text-indigo-400" />
                            {m.tokens} رمز
                          </span>
                        )}
                      </div>

                      <button
                        onClick={() => copyToClipboard(m.content, m.id)}
                        className="flex items-center gap-1 text-slate-400 hover:text-slate-200 transition-colors p-1 rounded hover:bg-white/[0.05]"
                        title="نسخ الإجابة"
                      >
                        {copiedId === m.id ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="text-emerald-400">تم النسخ</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>نسخ</span>
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))
          )}

          {/* Loading Animation Bubble */}
          {isLoading && (
            <div className="flex gap-3 sm:gap-4 flex-row">
              <div className="w-8 h-8 rounded-xl shrink-0 flex items-center justify-center bg-space-850 border border-indigo-500/30 text-indigo-400 shadow-md">
                <Sparkles className="w-4 h-4 animate-spin" />
              </div>

              <div className="bg-space-900/90 border border-indigo-500/30 rounded-2xl rounded-tl-sm px-5 py-4 max-w-[85%] text-xs sm:text-sm text-slate-200 shadow-xl">
                <div className="flex items-center gap-2 mb-2">
                  <span className="relative flex h-2.5 w-2.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-cyan-500"></span>
                  </span>
                  <span className="font-semibold text-cyan-300">
                    {elapsedSeconds < 2.0
                      ? "جاري تشغيل النموذج..."
                      : "جاري إنشاء الإجابة..."}
                  </span>
                  <span className="text-slate-400 font-mono text-xs">
                    ({elapsedSeconds} ث)
                  </span>
                </div>

                <div className="flex items-center gap-1.5 text-slate-400 text-xs">
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce" />
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.2s]" />
                  <div className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-bounce [animation-delay:0.4s]" />
                  <span className="mr-2 text-slate-400">
                    معالجة مباشرة على كارت الشاشة المحلي (RTX 5070)
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="p-4 sm:p-5 border-t border-white/[0.06] bg-space-900/90">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSend();
            }}
            className="relative flex items-end gap-2"
          >
            <textarea
              ref={textareaRef}
              rows={2}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="اكتب سؤالك في الذكاء الاصطناعي أو تعلم الآلة... (Enter للإرسال، Shift+Enter لسطر جديد)"
              className="w-full bg-space-950/80 border border-white/[0.1] focus:border-indigo-500 rounded-xl px-4 py-3 text-xs sm:text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 resize-none transition-all leading-relaxed"
              dir="rtl"
              disabled={isLoading}
            />

            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="h-[52px] px-5 rounded-xl bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 disabled:opacity-40 disabled:hover:from-indigo-600 disabled:hover:to-violet-600 text-white font-medium flex items-center justify-center gap-2 transition-all shadow-lg shadow-indigo-600/20 shrink-0"
              title="إرسال"
            >
              <Send className="w-4 h-4 rotate-180" />
              <span className="hidden sm:inline text-xs sm:text-sm">إرسال</span>
            </button>
          </form>

          <div className="mt-2.5 flex items-center justify-between text-[11px] text-slate-500 px-1">
            <span>اضغط Enter للإرسال • Shift + Enter لسطر جديد</span>
            <span className="hidden sm:inline">QLoRA V3.2 • 4-bit NF4</span>
          </div>
        </div>
      </div>
    </section>
  );
}
