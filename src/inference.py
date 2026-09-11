"""CLI inference using the same non-thinking prompt as training."""
import argparse
import torch
from peft import PeftModel
from src.utils import config, project_path, load_base, tokenizer_for, SYSTEM_PROMPT, run_safely


def load_tutor(cfg, adapter=None):
    model = load_base(cfg)
    if adapter:
        path = project_path(adapter)
        if not (path / "adapter_config.json").exists():
            raise ValueError(f"Adapter missing: {path}. Train an adapter first.")
        model = PeftModel.from_pretrained(model, str(path))
    model.eval()
    model.config.use_cache = True
    return model, tokenizer_for(cfg)


def answer(model, tokenizer, question, *, history=None, max_new_tokens=384,
           temperature=0.7, top_p=0.8, repetition_penalty=1.1, system_prompt=None):
    if not question.strip() or not 1 <= max_new_tokens <= 1024 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("Provide a question and valid generation parameters (tokens 1-1024).")
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}] + list(history or [])
    messages.append({"role": "user", "content": question})
    while True:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        if inputs.input_ids.shape[1] <= 1024:
            break
        if len(messages) <= 2:
            raise ValueError("Question too long; shorten it to fit the local context budget.")
        del messages[1:3]
    inputs = inputs.to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  repetition_penalty=repetition_penalty, pad_token_id=tokenizer.pad_token_id)
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    return tokenizer.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    args = parser.parse_args()
    cfg = config(args.config)
    model, tokenizer = load_tutor(cfg, None if args.base else args.adapter or cfg["output_dir"])
    while True:
        try:
            question = args.question or input("Question (exit to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        print(answer(model, tokenizer, question, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p, repetition_penalty=args.repetition_penalty))
        if args.question:
            break


if __name__ == "__main__":
    run_safely(main)
