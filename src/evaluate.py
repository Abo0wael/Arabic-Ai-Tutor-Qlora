"""Offline behavioral heuristics and a separate human scoring template."""
import argparse
import csv
import json
import re
import random
from collections import Counter
from src.utils import project_path, save_json


def checks(question: str, response: str) -> dict:
    letters = [c for c in response if c.isalpha()]
    ratio = sum('\u0600' <= c <= '\u06ff' for c in letters) / max(1, len(letters))
    arabic_question = bool(re.search(r"[\u0621-\u064a]", question))
    headings = ("تعريف", "شرح", "مثال", "خلاصة") if arabic_question else (
        "definition", "explanation", "example", "summary")
    positions = [response.lower().find(h) for h in headings]
    tokens = response.split()
    grams = list(zip(tokens, tokens[1:], tokens[2:], tokens[3:]))
    repeated = sum(n-1 for n in Counter(grams).values()) / max(1, len(grams))
    return dict(empty=not response.strip(), characters=len(response), words=len(tokens),
        arabic_letter_ratio=round(ratio, 3),
        arabic_when_requested=(ratio >= 0.5) if arabic_question else None,
        expected_structure=(all(p >= 0 for p in positions) and positions == sorted(positions))
            if all(h in question.lower() for h in headings) else None,
        repeated_4gram_ratio=round(repeated, 3), excessive_repetition=repeated > 0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/comparison_results.json")
    parser.add_argument("--output", default="outputs/evaluation_results.json")
    parser.add_argument("--manual-output", default="outputs/manual_evaluation.csv")
    args = parser.parse_args()
    path = project_path(args.manual_output)
    key_path = path.with_suffix('.key.json')
    if any(p.exists() for p in (path, key_path, project_path(args.output))):
        raise ValueError('Evaluation output exists; choose new output paths to preserve reviews.')
    rows = json.loads(project_path(args.input).read_text(encoding="utf-8"))
    automatic, manual, blind_key = [], [], []
    rng = random.Random(42)
    for row in rows:
        variants = ['base', 'fine_tuned']
        rng.shuffle(variants)
        for label, variant in zip(('A', 'B'), variants):
            automatic.append(dict(id=row["id"], model=variant, **checks(row["question"], row[variant])))
            blind_key.append(dict(id=row['id'], label=label, model=variant))
            manual.append(dict(id=row["id"], model=label, question=row["question"], response=row[variant],
                correctness="", clarity="", simplicity="", structure="", arabic_quality="",
                overall_usefulness="", notes=""))
    save_json(project_path(args.output), {"note": "Heuristics are not correctness scores. Human review required.",
                                         "results": automatic})
    save_json(key_path, blind_key)
    path = project_path(args.manual_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("Manual scoring file already exists; choose --manual-output to preserve scores.")
    if manual:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manual[0]))
            writer.writeheader()
            writer.writerows(manual)
    print(f"Evaluated {len(automatic)} responses. Human rubric: 1=poor, 3=adequate, 5=excellent; "
          "leave Arabic quality blank for English answers. Verify correctness independently.")


if __name__ == "__main__":
    main()
