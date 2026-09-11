# Free Deployment Guide: Local GPU (RTX 5070) + Cloudflare Tunnel + Next.js Vercel Frontend

This guide explains how to deploy the **Arabic AI Tutor** project using a **100% free, production-quality hybrid architecture**:

- **Frontend**: Next.js 15 + TypeScript + Tailwind CSS hosted on **Vercel Hobby** (free, public 24/7).
- **Backend / Inference**: FastAPI running locally on your laptop utilizing your **NVIDIA GeForce RTX 5070 GPU** with 4-bit NF4 quantized **Qwen3-1.7B** and the **V3.2 QLoRA adapter**.
- **Secure Networking**: **Cloudflare Tunnel** (`cloudflared`) exposing your local FastAPI server to the web over HTTPS with zero port forwarding and zero hosting fees.

---

## System Architecture

```
┌───────────────────────────┐
│       User Browser        │
└─────────────┬─────────────┘
              │
              ▼ HTTPS
┌───────────────────────────┐
│ Next.js Frontend (Vercel) │
└─────────────┬─────────────┘
              │
              ▼ HTTPS (via NEXT_PUBLIC_API_URL)
┌───────────────────────────┐
│     Cloudflare Tunnel     │
│ (*.trycloudflare.com)     │
└─────────────┬─────────────┘
              │ Secure Tunnel
              ▼
┌───────────────────────────┐
│ Local FastAPI Backend     │
│ (http://localhost:8000)   │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ NVIDIA RTX 5070 GPU       │
│ Qwen3-1.7B + QLoRA V3.2   │
│ (4-bit NF4, ~1.27 GB VRAM)│
└───────────────────────────┘
```

---

## Part 1: Starting the Local Backend

### 1. Requirements
Ensure your Python environment `.venv` has the necessary dependencies installed:
```powershell
# From project root
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]"
```

### 2. Launch FastAPI
Run Uvicorn from the project root:
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Verify Health
Open your browser or run:
```powershell
curl http://localhost:8000/health
```
Expected output:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "Qwen3-1.7B + QLoRA V3.2",
  "device": "NVIDIA GeForce RTX 5070 Laptop GPU",
  "vram_allocated_gb": 1.26,
  "vram_reserved_gb": 2.27
}
```

---

## Part 2: Exposing Local Backend via Cloudflare Tunnel

Cloudflare Tunnel provides a free, secure, encrypted HTTPS tunnel from the public internet directly to your `localhost:8000` without modifying your router or opening any firewall ports.

### Option A: Quick Free Temporary Tunnel (Recommended for Demos)

1. **Install `cloudflared` on Windows**:
   ```powershell
   winget install --id Cloudflare.cloudflared
   ```
   *Or download the standalone `cloudflared.exe` from [Cloudflare Releases](https://github.com/cloudflare/cloudflared/releases).*

2. **Start the Tunnel**:
   ```powershell
   cloudflared tunnel --url http://localhost:8000
   ```

3. **Copy the Public URL**:
   In the terminal output, look for a line like:
   ```text
   +--------------------------------------------------------------------------------------------+
   |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
   |  https://random-words-here.trycloudflare.com                                               |
   +--------------------------------------------------------------------------------------------+
   ```
   Your public API URL is: `https://random-words-here.trycloudflare.com`

---

## Part 3: Deploying Frontend to Vercel

The frontend is ready for immediate deployment on the **Vercel Hobby** tier.

### 1. Push Code to GitHub
Ensure all latest files (`frontend/`, `backend/`, `docs/`) are committed and pushed to your repository:
```powershell
git add .
git commit -m "Add local GPU backend and Next.js frontend"
git push origin main
```

### 2. Import into Vercel
1. Go to [vercel.com](https://vercel.com) and log in with your GitHub account.
2. Click **"Add New..."** -> **"Project"**.
3. Select your repository: `Abo0wael/Arabic-Ai-Tutor-Qlora`.
4. In the **Project Configuration** screen:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click *Edit* and select `frontend` (crucial: do not build from root).
5. Under **Environment Variables**, add:
   - **Name**: `NEXT_PUBLIC_API_URL`
   - **Value**: `https://random-words-here.trycloudflare.com` (from your Cloudflare Tunnel)
6. Click **Deploy**.

Vercel will build and deploy your portfolio frontend in under 60 seconds.

---

## Part 4: Local Development & Offline Behavior

### Local Testing
If you want to test the frontend locally while the backend is running:
```powershell
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

### Laptop Online / Offline Experience
Because inference runs on your physical RTX 5070 GPU:
- **When your laptop & backend are running**: The frontend header shows a glowing green indicator (`متصل - RTX 5070`) and generates responses in ~3–5 seconds.
- **When your laptop is turned off / sleeping**: The Vercel frontend remains online, but gracefully informs visitors:
  > *"النموذج غير متصل حاليًا. هذا الديمو يعتمد على معالجة محلية على كارت الشاشة (RTX 5070)، وقد يكون غير متاح عندما يكون الجهاز مغلقًا."*
