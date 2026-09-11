"""FastAPI entry point for the Arabic AI Tutor local inference backend."""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import HealthResponse, GenerateRequest, GenerateResponse, ChatRequest
from backend.model_service import TutorModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads model once at startup and frees resources on shutdown."""
    print("[FastAPI] Starting Arabic AI Tutor backend...")
    service = TutorModelService.get_instance()
    try:
        service.load_model()
    except Exception as exc:
        print(f"[FastAPI] ERROR during model initialization: {exc}")
    yield
    print("[FastAPI] Shutting down backend...")
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Arabic AI Tutor API",
    description="Local inference server for Fine-Tuned Qwen3-1.7B with QLoRA on NVIDIA RTX 5070",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
# Allow localhost for development + dynamic origins from environment variables (e.g., Vercel URL)
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
]
env_origins = os.getenv("FRONTEND_ORIGIN", "")
if env_origins:
    for origin in env_origins.split(","):
        cleaned = origin.strip().rstrip("/")
        if cleaned and cleaned not in default_origins:
            default_origins.append(cleaned)

# Also allow wildcard if explicitly requested for public tunnel testing
allow_all = os.getenv("ALLOW_ALL_ORIGINS", "true").lower() in ("true", "1", "yes")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allow_all else default_origins,
    allow_credentials=True if not allow_all else False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
async def root():
    return {
        "service": "Arabic AI Tutor Backend",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/health",
        "architecture": "Next.js + Cloudflare Tunnel + Local RTX 5070 GPU",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    service = TutorModelService.get_instance()
    allocated, reserved = service.get_gpu_memory()
    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded,
        model_name="Qwen3-1.7B + QLoRA V3.2",
        device=service.device_name,
        vram_allocated_gb=allocated,
        vram_reserved_gb=reserved,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
async def generate(req: GenerateRequest):
    service = TutorModelService.get_instance()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing or failed to load. Check /health.",
        )

    try:
        response_text, elapsed, tokens = service.generate(
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
        )
        return GenerateResponse(
            response=response_text,
            generation_time=elapsed,
            tokens_generated=tokens,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/chat", response_model=GenerateResponse, tags=["Inference"])
async def chat(req: ChatRequest):
    service = TutorModelService.get_instance()
    if not service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is still initializing or failed to load.",
        )

    history = [msg.model_dump() for msg in req.messages[:-1]]
    latest_prompt = req.messages[-1].content

    try:
        response_text, elapsed, tokens = service.generate(
            prompt=latest_prompt,
            history=history,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
        )
        return GenerateResponse(
            response=response_text,
            generation_time=elapsed,
            tokens_generated=tokens,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
