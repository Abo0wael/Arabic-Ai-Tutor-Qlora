/**
 * API client for interacting with the local GPU FastAPI backend
 * via direct URL or Cloudflare Tunnel.
 */

export interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_name: string;
  device: string;
  vram_allocated_gb?: number | null;
  vram_reserved_gb?: number | null;
}

export interface GenerateResponse {
  response: string;
  generation_time: number;
  tokens_generated?: number;
}

export interface ApiError {
  isOffline: boolean;
  isTimeout: boolean;
  message: string;
  statusText?: string;
}

export function getApiBaseUrl(): string {
  const envUrl = process.env.NEXT_PUBLIC_API_URL;
  if (envUrl && envUrl.trim() !== "") {
    return envUrl.trim().replace(/\/+$/, "");
  }
  // Default to local backend if running in browser
  if (typeof window !== "undefined") {
    return "http://localhost:8000";
  }
  return "http://127.0.0.1:8000";
}

/**
 * Probes the backend /health endpoint to verify if the local server is online.
 */
export async function checkHealth(): Promise<{
  ok: boolean;
  data?: HealthStatus;
  error?: string;
}> {
  const baseUrl = getApiBaseUrl();
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 4000);

  try {
    const res = await fetch(`${baseUrl}/health`, {
      method: "GET",
      signal: controller.signal,
      headers: { Accept: "application/json" },
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      return { ok: false, error: `HTTP ${res.status}` };
    }

    const data: HealthStatus = await res.json();
    return { ok: true, data };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    return {
      ok: false,
      error: err instanceof Error ? err.message : "Unreachable",
    };
  }
}

/**
 * Sends a prompt to the backend for GPU inference.
 */
export async function generateTutorAnswer(
  prompt: string,
  history?: Array<{ role: "user" | "assistant"; content: string }>,
  options?: {
    max_new_tokens?: number;
    temperature?: number;
    repetition_penalty?: number;
  }
): Promise<{ success: true; data: GenerateResponse } | { success: false; error: ApiError }> {
  const baseUrl = getApiBaseUrl();
  const controller = new AbortController();
  // 60-second timeout for first-token or longer generation
  const timeoutId = setTimeout(() => controller.abort(), 60000);

  const endpoint = history && history.length > 0 ? `${baseUrl}/chat` : `${baseUrl}/generate`;

  const payload =
    history && history.length > 0
      ? {
          messages: [...history, { role: "user", content: prompt }],
          max_new_tokens: options?.max_new_tokens ?? 256,
          temperature: options?.temperature ?? 0.7,
          repetition_penalty: options?.repetition_penalty ?? 1.1,
        }
      : {
          prompt,
          max_new_tokens: options?.max_new_tokens ?? 256,
          temperature: options?.temperature ?? 0.7,
          repetition_penalty: options?.repetition_penalty ?? 1.1,
        };

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const errorText = await res.text().catch(() => "");
      return {
        success: false,
        error: {
          isOffline: false,
          isTimeout: false,
          message: `خطأ من الخادم (${res.status}): ${errorText || res.statusText}`,
          statusText: res.statusText,
        },
      };
    }

    const data: GenerateResponse = await res.json();
    return { success: true, data };
  } catch (err: unknown) {
    clearTimeout(timeoutId);

    if (err instanceof Error && err.name === "AbortError") {
      return {
        success: false,
        error: {
          isOffline: false,
          isTimeout: true,
          message: "استغرق الخادم وقتاً أطول من المتوقع للرد (مهلة 60 ثانية). يرجى المحاولة مرة أخرى.",
        },
      };
    }

    return {
      success: false,
      error: {
        isOffline: true,
        isTimeout: false,
        message:
          "النموذج غير متصل حاليًا. هذا الديمو يعتمد على معالجة محلية على كارت الشاشة (RTX 5070)، وقد يكون غير متاح عندما يكون الجهاز مغلقًا.",
      },
    };
  }
}
