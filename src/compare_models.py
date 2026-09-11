"""Compare a base model and its adapter without loading two GPU copies."""
import argparse
import json
import hashlib
from transformers import set_seed
from src.inference import load_tutor, answer
from src.prepare_data import clean_rows
from src.utils import config, project_path, save_json, run_safely

EVAL_SYSTEM_PROMPT = "You are a helpful Arabic AI assistant. Answer the user's question accurately."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--adapter")
    parser.add_argument("--questions")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = config(args.config)
    args.questions = args.questions or cfg.get("evaluation_questions", "data/test_questions.json")
    args.output = args.output or cfg.get("comparison_output", "outputs/comparison_results.json")
    if project_path(args.output).exists():
        raise ValueError("Comparison output exists; choose a new --output to preserve previous results.")
    questions = json.loads(project_path(args.questions).read_text(encoding="utf-8"))
    if not cfg.get("split_dir"):
        raise ValueError("Evaluation requires explicit train/validation/test splits.")
    rows = []
    for split in ("train", "validation"):
        clean, _ = clean_rows(project_path(cfg["split_dir"]) / f"{split}.jsonl")
        rows.extend(clean)
    train_questions = {r["messages"][1]["content"].strip().casefold() for r in rows}
    if any(q["question"].strip().casefold() in train_questions for q in questions):
        raise ValueError("Test questions overlap with training or validation data.")
    from src.dataset_integrity import grams, similar
    if any(similar(grams(q['question']), grams(r['messages'][1]['content'])) for q in questions for r in rows):
        raise ValueError('Near-duplicate evaluation question overlaps training or validation data.')
    save_json(project_path(args.output).with_suffix('.provenance.json'), dict(
        config=cfg, adapter=args.adapter or cfg['output_dir'],
        questions_sha256=hashlib.sha256(project_path(args.questions).read_bytes()).hexdigest(),
        split_sha256={s:hashlib.sha256((project_path(cfg['split_dir']) / f'{s}.jsonl').read_bytes()).hexdigest()
                      for s in ('train','validation','test')}))
    model, tokenizer = load_tutor(cfg, args.adapter or cfg["output_dir"])
    results = []
    for item in questions:
        set_seed(cfg["seed"])
        with model.disable_adapter():
            base = answer(model, tokenizer, item["question"], temperature=0,
                          system_prompt=EVAL_SYSTEM_PROMPT)
        set_seed(cfg["seed"])
        tuned = answer(model, tokenizer, item["question"], temperature=0,
                       system_prompt=EVAL_SYSTEM_PROMPT)
        results.append({**item, "base": base, "fine_tuned": tuned})
        save_json(project_path(args.output), results)
        print(f"\nQuestion: {item['question']}\n| Base | Fine-tuned |\n| --- | --- |")
        print("| " + base.replace("\n", " / ").replace("|", "¦") + " | " +
              tuned.replace("\n", " / ").replace("|", "¦") + " |", flush=True)


if __name__ == "__main__":
    run_safely(main)
