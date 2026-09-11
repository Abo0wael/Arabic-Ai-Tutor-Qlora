"""Create Train / Validation / Test Splits for Dataset V3.2.
Preserves V3.1 splits in data/splits/ untouched.
Outputs to data/v3_2/splits/.
Splits are balanced across behavior types.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def create_splits():
    random.seed(42)
    in_path = ROOT / "data/v3_2/clean_dataset_v3_2.jsonl"
    rows = [json.loads(line) for line in in_path.open("r", encoding="utf-8")]
    print(f"Total V3.2 dataset size: {len(rows)}")

    from src.dataset_integrity import grouped_split
    splits = grouped_split(rows)
    train_set, val_set, test_set = (splits[k] for k in ('train', 'validation', 'test'))
    questions = [dict(id=f'v32-{i}', question=r['messages'][1]['content'], reference=r['messages'][2]['content']) for i,r in enumerate(test_set)]
    (in_path.parent / 'evaluation_questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f"Split sizes -> Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    assert len(train_set) + len(val_set) + len(test_set) == len(rows), "Split count mismatch!"

    # Save to data/v3_2/splits/
    out_dir = ROOT / "data/v3_2/splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, s in [("train.jsonl", train_set), ("validation.jsonl", val_set), ("test.jsonl", test_set)]:
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for row in s:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {name} ({len(s)} rows) to: {p}")

if __name__ == "__main__":
    create_splits()
