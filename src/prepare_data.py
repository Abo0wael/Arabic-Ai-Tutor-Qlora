"""Validate, deduplicate and format chat examples without early tokenization."""
import argparse
import json
import math
from pathlib import Path
from src.utils import config, project_path, tokenizer_for


def clean_rows(path: Path):
    stats = dict(total=0, invalid=0, duplicates=0)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError("JSON input must be an array of examples.")
    else:
        rows = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append(None)
    clean, seen = [], set()
    for row in rows:
        stats["total"] += 1
        messages = row.get("messages") if isinstance(row, dict) else None
        valid = isinstance(messages, list) and len(messages) == 3
        if valid:
            valid = all(isinstance(m, dict) and m.get("role") == role and
                        isinstance(m.get("content"), str) and m["content"].strip()
                        for m, role in zip(messages, ("system", "user", "assistant")))
        if not valid:
            stats["invalid"] += 1
            continue
        messages = [{"role": m["role"], "content": m["content"].strip()} for m in messages]
        key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        clean.append({"messages": messages})
    return clean, stats


def prepare(cfg):
    from datasets import Dataset, DatasetDict
    split_dir = project_path(cfg["split_dir"]) if cfg.get("split_dir") else None
    if split_dir:
        from src.dataset_integrity import validate_splits
        raw_splits = {name: [json.loads(line) for line in (split_dir / f'{name}.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
                      for name in ('train', 'validation', 'test')}
        if any(r.get('concept_id') for values in raw_splits.values() for r in values):
            validate_splits(raw_splits)
        source_splits, stats = {}, {"total": 0, "invalid": 0, "duplicates": 0}
        global_questions = set()
        for split in ("train", "validation", "test"):
            split_rows, split_stats = clean_rows(split_dir / f"{split}.jsonl")
            for key in stats:
                stats[key] += split_stats[key]
            questions = {r["messages"][1]["content"].strip().casefold() for r in split_rows}
            if global_questions & questions:
                raise ValueError(f"Exact question leakage detected in {split} split.")
            global_questions |= questions
            source_splits[split] = split_rows
        rows = [row for values in source_splits.values() for row in values]
    else:
        rows, stats = clean_rows(project_path(cfg["dataset_path"]))
    if len(rows) < 2:
        raise ValueError("At least two valid unique examples are required.")
    # Group identical questions to prevent leakage across train and validation.
    groups = {}
    for row in rows:
        groups.setdefault(row["messages"][1]["content"], []).append(row)
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError("At least two unique questions are required.")
    if split_dir:
        keys = []
    elif cfg["split_strategy"] == "random":
        import random
        random.Random(cfg["seed"]).shuffle(keys)
    elif cfg["split_strategy"] != "chronological":
        raise ValueError("split_strategy must be random or chronological.")
    fraction = cfg["validation_fraction"]
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    n = min(len(keys)-1, max(1, math.ceil(len(keys)*fraction))) if keys else 0
    tokenizer = tokenizer_for(cfg)
    def formatted_rows(rows_to_format):
        result = []
        for row in rows_to_format:
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(messages[:2], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            # Match the non-thinking inference prefix exactly; supervise answer + EOS only.
            completion = messages[2]["content"] + tokenizer.eos_token
            result.append(dict(messages=messages, prompt=prompt, completion=completion,
                               text=prompt + completion))
        return Dataset.from_list(result)
    if split_dir:
        data = DatasetDict({split: formatted_rows(values) for split, values in source_splits.items()})
    else:
        data = DatasetDict(
            train=formatted_rows([row for key in keys[:-n] for row in groups[key]]),
            validation=formatted_rows([row for key in keys[-n:] for row in groups[key]]),
        )
    destination = project_path(cfg["processed_dir"])
    data.save_to_disk(str(destination))
    print({**stats, **{split: len(values) for split, values in data.items()},
           "mean_characters": round(sum(len(r["messages"][2]["content"]) for r in rows)/len(rows))})
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--input")
    args = parser.parse_args()
    cfg = config(args.config)
    if args.input:
        cfg["dataset_path"] = args.input
        cfg.pop("split_dir", None)
    prepare(cfg)


if __name__ == "__main__":
    main()
