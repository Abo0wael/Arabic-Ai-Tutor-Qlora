# Arabic AI Tutor — Local GPU Backend

FastAPI backend serving the fine-tuned **Qwen3-1.7B + QLoRA V3.2** model directly on your local **NVIDIA RTX 5070 GPU** in 4-bit NF4 quantization.

---

## Prerequisites

- Windows with NVIDIA GPU (RTX 5070 or compatible)
- Python virtual environment `.venv` with CUDA-enabled PyTorch
- Trained adapter at `outputs/adapter_v3_2/`

---

## How to Run

From the project root:

```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run the FastAPI server with Uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Endpoints

- `GET /health`: Model status, device name, and VRAM consumption.
- `POST /generate`: Single prompt generation.
  ```json
  {
    "prompt": "ما هو التعلم العميق؟",
    "max_new_tokens": 256,
    "temperature": 0.7
  }
  ```
- `POST /chat`: Multi-turn conversation format.
  ```json
  {
    "messages": [
      {"role": "user", "content": "اشرح Overfitting"},
      {"role": "assistant", "content": "..."},
      {"role": "user", "content": "كيف أمنعه؟"}
    ]
  }
  ```
- `GET /docs`: Interactive Swagger API documentation.

---

## Cloudflare Tunnel Integration

To expose this backend securely to your Vercel frontend:

```powershell
cloudflared tunnel --url http://localhost:8000
```
