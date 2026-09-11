# Arabic AI Tutor (V3.2) — Results & Experiment Summary

A high-level overview of the engineering specifications, training run, and comparative evaluation for the fine-tuned Arabic AI Tutor.

---

## 1. Quick Stats

| Metric | Specification / Result |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen3-1.7B` |
| **Fine-Tuning Method** | QLoRA (4-bit NormalFloat + Double Quantization) |
| **Compute Precision** | `bf16` |
| **Dataset Size** | 501 curated Arabic AI tutoring conversations |
| **Data Split** | Train: 426 (85%) \| Validation: 50 (10%) \| Test: 25 (5%) |
| **LoRA Rank & Alpha** | $r = 8, \alpha = 16$ ($\text{scaling} = 2.0$), dropout $= 0.05$ |
| **Target Modules** | `q_proj`, `v_proj` |
| **Total Parameters** | 1,722,180,608 |
| **Trainable Parameters** | **1,605,632 (0.0932%)** |
| **Batch Configuration** | Batch size = 1, Gradient Accumulation = 8 (Effective Batch = 8) |
| **Optimizer** | `paged_adamw_8bit` (learning rate $= 1\times 10^{-4}$) |
| **Schedule** | Cosine with 10% warmup (5 warmup steps) |
| **Epochs** | **1 single epoch** |
| **Optimizer Steps** | **54 steps** |
| **Training Duration** | **255.4 seconds (~4.2 minutes)** |
| **Hardware** | NVIDIA GeForce RTX 5070 Laptop GPU (7.96 GiB VRAM) |
| **Peak VRAM Usage** | **3.17 GiB** |
| **Midpoint Validation Loss** | `2.5738` (Step 27) |
| **Final Validation Loss** | **`2.5073`** (Step 54) |
| **Evaluation Benchmark** | 25 deterministic Base vs. Fine-Tuned qualitative comparisons |

---

## 2. Loss & Convergence Profile

```text
Step 05 | Train Loss: 3.2494 | LR: 6.67e-5
Step 10 | Train Loss: 3.2153 | LR: 9.90e-5
Step 15 | Train Loss: 2.7190 | LR: 9.33e-5
Step 20 | Train Loss: 2.7180 | LR: 8.30e-5
Step 25 | Train Loss: 2.5051 | LR: 6.91e-5
Step 27 | Validation Loss: 2.5738  <-- Midpoint checkpoint
Step 30 | Train Loss: 2.5862 | LR: 5.33e-5
Step 35 | Train Loss: 2.4626 | LR: 3.71e-5
Step 40 | Train Loss: 2.5940 | LR: 2.22e-5
Step 45 | Train Loss: 2.4001 | LR: 1.03e-5
Step 50 | Train Loss: 2.4434 | LR: 2.65e-6
Step 54 | Validation Loss: 2.5073  <-- Final evaluation checkpoint
```

Validation loss steadily declined from **2.5738** to **2.5073** without loss explosion or divergence, demonstrating stable convergence within a single epoch.

---

## 3. Evaluation Findings at a Glance

### Observed Strengths:
- **Autonomous Formatting**: When asked for structured explanations (e.g. definition, explanation, example, summary), the model adhered strictly to the format without needing formatting instructions in the system prompt.
- **AI Domain Grounding**: Avoided catastrophic domain confusion on ambiguous terms like *الهلوسة* (hallucination), where the Base model frequently drifted into psychiatric/clinical definitions.
- **Length Calibration**: Removed conversational pleasantries and produced direct, concise answers when requested.

### Documented Limitations:
- **Language Drift (12%)**: 3 out of 25 prompts that opened with a Latin technical term (e.g. *Overfitting*) triggered English continuations due to the minimal neutral prompt lacking explicit language gating.
- **Repetition Loop (4%)**: 1 out of 25 responses entered a checklist repetition loop during greedy decoding on Egyptian dialect input.
- **Scope**: Designed specifically for behavioral adaptation; does not serve as a factual knowledge base or replace RAG for real-time document search.
