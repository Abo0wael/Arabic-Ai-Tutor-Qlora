"""Model service managing Qwen3-1.7B + QLoRA V3.2 inference on local RTX 5070."""
import sys
import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from src.utils import config, project_path, load_base, tokenizer_for, SYSTEM_PROMPT


class TutorModelService:
    _instance: Optional["TutorModelService"] = None
    _lock = threading.Lock()

    def __init__(self, config_path: str = "configs/training_config.yaml", adapter_dir: str = "outputs/adapter_v3_2"):
        self.config_path = config_path
        self.adapter_dir = adapter_dir
        self.cfg: Optional[Dict[str, Any]] = None
        self.model = None
        self.tokenizer = None
        self.device_name: str = "cpu"
        self.is_loaded: bool = False
        self._gen_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "TutorModelService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def load_model(self) -> None:
        """Load 4-bit base model and V3.2 QLoRA adapter onto GPU once."""
        if self.is_loaded:
            return

        with self._gen_lock:
            if self.is_loaded:
                return

            print(f"[ModelService] Loading configuration from {self.config_path}...")
            self.cfg = config(self.config_path)

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.device_name = props.name
                print(f"[ModelService] Target GPU: {self.device_name} ({props.total_memory / (1024**3):.2f} GiB)")
            else:
                self.device_name = "cpu"
                print("[ModelService] WARNING: CUDA is not available. Running on CPU.")

            adapter_path = project_path(self.adapter_dir)
            if not (adapter_path / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"Adapter directory not found at {adapter_path}. Expected outputs/adapter_v3_2."
                )

            print(f"[ModelService] Loading base model ({self.cfg.get('model_name')}) in 4-bit NF4...")
            base_model = load_base(self.cfg)

            print(f"[ModelService] Attaching QLoRA V3.2 adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.eval()
            self.model.config.use_cache = True

            print("[ModelService] Initializing tokenizer...")
            self.tokenizer = tokenizer_for(self.cfg)

            self.is_loaded = True
            print("[ModelService] Model and adapter successfully loaded and ready for inference.")

    def get_gpu_memory(self) -> Tuple[Optional[float], Optional[float]]:
        """Return allocated and reserved GPU memory in GiB."""
        if torch.cuda.is_available():
            allocated = round(torch.cuda.memory_allocated() / (1024**3), 3)
            reserved = round(torch.cuda.memory_reserved() / (1024**3), 3)
            return allocated, reserved
        return None, None

    def generate(
        self,
        prompt: str,
        *,
        history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.1,
    ) -> Tuple[str, float, int]:
        """Generate response with serialized GPU execution and timing."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model service is not loaded. Call load_model() first.")

        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for item in history:
                messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": prompt.strip()})

        # Apply chat template
        while True:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = self.tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")
            if inputs.input_ids.shape[1] <= 1024 or len(messages) <= 2:
                break
            # Trim older history turns if context budget exceeded
            del messages[1:3]

        inputs = inputs.to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        start_time = time.perf_counter()
        with self._gen_lock:
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **gen_kwargs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                raise RuntimeError("CUDA Out Of Memory. Try shortening the prompt or decreasing max_new_tokens.")

        elapsed = round(time.perf_counter() - start_time, 2)
        generated_tokens = outputs[0, input_length:]
        token_count = len(generated_tokens)
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return decoded, elapsed, token_count
