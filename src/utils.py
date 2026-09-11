"""Shared configuration, GPU checks, and quantized model loading."""
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
SYSTEM_PROMPT = "You are a helpful Arabic AI assistant. Answer the user's question accurately."


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def config(path: str = "configs/training_config.yaml") -> dict[str, Any]:
    import yaml
    with project_path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg["max_seq_length"] < 32 or cfg["lora"]["r"] < 1:
        raise ValueError("Sequence length must be >=32 and LoRA rank >=1.")
    if not cfg["quantization"]["load_in_4bit"]:
        raise ValueError("This project requires 4-bit QLoRA; full fine-tuning is disabled.")
    return cfg


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def memory(label: str) -> None:
    import torch
    if torch.cuda.is_available():
        print(f"{label}: allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB; "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f} GiB; "
              f"peak={torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)


def gpu_check():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install a CUDA 12.8+ PyTorch build; see README.")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}; VRAM: {props.total_memory/2**30:.2f} GiB; "
          f"torch={torch.__version__}; CUDA runtime={torch.version.cuda}")
    # Exercise a CUDA kernel: detection alone does not prove architecture compatibility.
    (torch.ones(2, device="cuda") * 2).sum().item()
    memory("GPU check")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def tokenizer_for(cfg):
    from transformers import AutoTokenizer
    source = str(project_path(cfg.get("local_model_path", cfg["model_name"])))
    tokenizer = AutoTokenizer.from_pretrained(source, token=os.getenv("HF_TOKEN"),
        cache_dir=str(project_path(cfg["cache_dir"])), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base(cfg):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype = gpu_check()
    q = cfg["quantization"]
    source = str(project_path(cfg.get("local_model_path", cfg["model_name"])))
    model = AutoModelForCausalLM.from_pretrained(
        source, token=os.getenv("HF_TOKEN"), device_map="auto",
        cache_dir=str(project_path(cfg["cache_dir"])), local_files_only=True,
        dtype=dtype, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type=q["quant_type"], bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=dtype),
    )
    if any(str(d) in {"cpu", "disk"} for d in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("Model offloaded to CPU/disk. Free GPU memory before training.")
    memory("Model loaded")
    return model


def run_safely(main) -> None:
    import torch
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        memory("CUDA OOM")
        print("Out of VRAM. Reduce max_seq_length to 256; reduce LoRA rank to 4; "
              "keep batch size 1 and increase gradient accumulation instead; enable "
              "gradient checkpointing; close GPU-heavy apps. Resume a saved checkpoint.")
        raise SystemExit(2)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
