"""Pydantic data models for the Arabic AI Tutor FastAPI backend."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    model_name: str = "Qwen3-1.7B + QLoRA V3.2"
    device: str = "cuda"
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048, description="User question or prompt in Arabic")
    max_new_tokens: int = Field(default=256, ge=16, le=1024, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=1.5, description="Sampling temperature (0 for greedy)")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")


class GenerateResponse(BaseModel):
    response: str
    generation_time: float
    tokens_generated: Optional[int] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int = Field(default=256, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=1.5)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
