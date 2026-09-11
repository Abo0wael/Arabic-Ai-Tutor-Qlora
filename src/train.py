"""Memory-conscious QLoRA training with assistant-only dynamic labels."""
import argparse
from dataclasses import dataclass
import torch
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from src.utils import config, project_path, tokenizer_for, load_base, memory, run_safely, save_json


@dataclass
class AssistantCollator:
    tokenizer: object
    max_length: int

    def encode(self, row):
        prompt = self.tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        answer = self.tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        if len(prompt) + len(answer) > self.max_length:
            raise ValueError("Example exceeds max_seq_length; filter it before batching.")
        return prompt + answer, [-100] * len(prompt) + answer

    def __call__(self, rows):
        encoded = [self.encode(row) for row in rows]
        width = max(len(ids) for ids, _ in encoded)
        return {"input_ids": torch.tensor([ids + [self.tokenizer.pad_token_id]*(width-len(ids))
                                           for ids, _ in encoded]),
                "attention_mask": torch.tensor([[1]*len(ids)+[0]*(width-len(ids)) for ids, _ in encoded]),
                "labels": torch.tensor([labels+[-100]*(width-len(labels)) for _, labels in encoded])}


class MemoryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"step={state.global_step}, epoch={state.epoch}, metrics={logs}", flush=True)
        memory("Training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--load-test", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    cfg = config(args.config)
    set_seed(cfg["seed"])
    model = load_base(cfg)
    tokenizer = tokenizer_for(cfg)
    if args.load_test:
        batch = tokenizer("مرحبا", return_tensors="pt").to(model.device)
        with torch.inference_mode():
            model(**batch)
        memory("Load + forward test passed")
        return
    settings = cfg["training"]
    model = prepare_model_for_kbit_training(model,
        use_gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    lora = cfg["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], target_modules=lora["target_modules"],
        bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    data = load_from_disk(str(project_path(cfg["processed_dir"])))
    collator = AssistantCollator(tokenizer, cfg["max_seq_length"])
    for split in data:
        before = len(data[split])
        data[split] = data[split].filter(lambda row: sum(len(tokenizer(row[k],
            add_special_tokens=False)["input_ids"]) for k in ("prompt", "completion")) <= cfg["max_seq_length"])
        print(f"{split}: retained {len(data[split])}/{before}; overlength examples dropped, never silently truncated.")
        if not len(data[split]):
            raise ValueError(f"No usable {split} examples; shorten examples or adjust max_seq_length.")
    if args.smoke_test:
        data["train"] = data["train"].select(range(min(16, len(data["train"]))))
        data["validation"] = data["validation"].select(range(min(2, len(data["validation"]))))
    out = project_path("outputs/smoke_adapter" if args.smoke_test else cfg["output_dir"])
    if out.exists() and any(out.iterdir()) and not args.resume_from_checkpoint:
        raise ValueError(f"Output {out} is not empty. Resume a checkpoint or choose a new output directory.")
    bf16 = torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(output_dir=str(out),
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["eval_batch_size"],
        gradient_accumulation_steps=1 if args.smoke_test else settings["gradient_accumulation_steps"],
        num_train_epochs=settings["epochs"], max_steps=2 if args.smoke_test else -1,
        learning_rate=settings["learning_rate"], warmup_ratio=settings["warmup_ratio"],
        bf16=bf16, fp16=not bf16, optim=settings["optimizer"],
        lr_scheduler_type=settings.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=settings["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1 if args.smoke_test else settings["logging_steps"],
        eval_strategy="steps", eval_steps=2 if args.smoke_test else settings["save_steps"],
        save_strategy="steps", save_steps=2 if args.smoke_test else settings["save_steps"],
        save_total_limit=settings["save_total_limit"], report_to="tensorboard" if settings["tensorboard"] else "none",
        seed=cfg["seed"], data_seed=cfg["seed"], remove_unused_columns=False,
        dataloader_num_workers=0, prediction_loss_only=True, label_names=["labels"])
    trainer = Trainer(model=model, args=training_args, train_dataset=data["train"],
        eval_dataset=data["validation"], data_collator=collator, callbacks=[MemoryCallback()])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    save_json(out / "run_config.json", cfg)
    save_json(out / "metrics.json", metrics)
    save_json(out / "log_history.json", trainer.state.log_history)
    memory("Finished")
    if cfg["push_to_hub"] and not args.smoke_test:
        if not cfg["hub_model_id"]:
            raise ValueError("Set hub_model_id before enabling upload.")
        import os
        model.push_to_hub(cfg["hub_model_id"], token=os.getenv("HF_TOKEN"))
        tokenizer.push_to_hub(cfg["hub_model_id"], token=os.getenv("HF_TOKEN"))


if __name__ == "__main__":
    run_safely(main)
