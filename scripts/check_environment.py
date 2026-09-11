"""Inspect Python, package versions and actual CUDA execution."""
import importlib.metadata
import platform
from src.utils import gpu_check

print("Python:", platform.python_version())
for package in ("torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes", "streamlit"):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
gpu_check()
