"""Comprehensive Quality & Leakage Audit for Dataset V3.2."""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def get_3grams(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) < 3:
        return set([tuple(words)])
    return set(tuple(words[i:i+3]) for i in range(len(words)-2))

def jaccard_sim(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def is_arabic_char(ch):
    return '\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' or '\u08A0' <= ch <= '\u08FF'

def audit_dataset():
    sys.stdout.reconfigure(encoding='utf-8')
    data_path = ROOT / "data/v3_2/clean_dataset_v3_2.jsonl"
    rows = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
    print(f"=== AUDITING DATASET V3.2 (Total: {len(rows)} examples) ===\n")

    # 1. Behavior Distribution
    beh_counts = Counter(r["behavior"] for r in rows)
    print("--- 1. Behavior Distribution ---")
    for b, cnt in sorted(beh_counts.items(), key=lambda x: -x[1]):
        pct = cnt / len(rows) * 100
        print(f"  - {b}: {cnt} ({pct:.1f}%)")

    # 2. Response Length Distribution by Behavior
    print("\n--- 2. Response Length Distribution (Chars & Words) ---")
    beh_lengths = defaultdict(list)
    beh_wlengths = defaultdict(list)
    for r in rows:
        ans = r["messages"][2]["content"]
        beh_lengths[r["behavior"]].append(len(ans))
        beh_wlengths[r["behavior"]].append(len(ans.split()))

    for b in sorted(beh_counts.keys()):
        clist = beh_lengths[b]
        wlist = beh_wlengths[b]
        print(f"  - {b}:")
        print(f"      Chars -> Min: {min(clist)}, Max: {max(clist)}, Avg: {sum(clist)/len(clist):.1f}")
        print(f"      Words -> Min: {min(wlist)}, Max: {max(wlist)}, Avg: {sum(wlist)/len(wlist):.1f}")

    # 3. Exact Duplicate Detection
    prompts = [r["messages"][1]["content"].strip() for r in rows]
    answers = [r["messages"][2]["content"].strip() for r in rows]
    prompt_pairs = [(r["messages"][1]["content"].strip(), r["messages"][2]["content"].strip()) for r in rows]

    dup_prompts = [p for p, c in Counter(prompts).items() if c > 1]
    dup_answers = [a for a, c in Counter(answers).items() if c > 1]
    dup_pairs = [pair for pair, c in Counter(prompt_pairs).items() if c > 1]

    print("\n--- 3. Exact Duplicates ---")
    print(f"  - Exact Duplicate (Prompt, Answer) Pairs: {len(dup_pairs)}")
    print(f"  - Duplicate Prompts: {len(dup_prompts)}")
    print(f"  - Duplicate Answers: {len(dup_answers)}")

    # 4. Near-Duplicate Detection (Jaccard on 3-grams)
    print("\n--- 4. Near-Duplicate Detection ---")
    prompt_3grams = [get_3grams(p) for p in prompts]
    near_dups = 0
    high_sim_pairs = []
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            sim = jaccard_sim(prompt_3grams[i], prompt_3grams[j])
            if sim > 0.85:
                near_dups += 1
                if len(high_sim_pairs) < 5:
                    high_sim_pairs.append((prompts[i], prompts[j], sim))

    print(f"  - Near-Duplicate Prompts (Jaccard > 0.85): {near_dups}")
    for p1, p2, s in high_sim_pairs:
        print(f"    * Sim {s:.2f}: '{p1}' vs '{p2}'")

    # 5. Prompt Echo Detection
    print("\n--- 5. Prompt Echo Detection ---")
    echo_count = 0
    for r in rows:
        p = r["messages"][1]["content"].strip().lower()
        a = r["messages"][2]["content"].strip().lower()
        if p in a and len(p) > 20 and len(a) < len(p) * 1.5:
            echo_count += 1
    print(f"  - Direct Prompt Echoes: {echo_count}")

    # 6. Sentence and Phrase Repetition Analysis across Answers
    print("\n--- 6. Repeated Sentence Analysis across Answers ---")
    all_sentences = []
    for a in answers:
        sents = re.split(r"[\n\.\!\؟]", a)
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) > 25:  # meaningful sentence
                all_sentences.append(s_clean)

    sent_counts = Counter(all_sentences)
    repeated_sents = [(s, c) for s, c in sent_counts.items() if c > 2]
    print(f"  - Sentences repeated > 2 times across the entire dataset: {len(repeated_sents)}")
    for s, c in sorted(repeated_sents, key=lambda x: -x[1])[:8]:
        print(f"    * ({c} times): '{s[:80]}...'")

    # 7. Arabic Character Ratio
    print("\n--- 7. Language & Script Ratio ---")
    total_chars = 0
    ar_chars = 0
    en_chars = 0
    other_chars = 0
    for a in answers:
        for ch in a:
            total_chars += 1
            if is_arabic_char(ch):
                ar_chars += 1
            elif ch.isalpha():
                en_chars += 1
            else:
                other_chars += 1

    print(f"  - Arabic Char Ratio: {ar_chars / total_chars * 100:.2f}%")
    print(f"  - English Char Ratio (technical terms): {en_chars / total_chars * 100:.2f}%")
    print(f"  - Punctuation / Digits / Math Ratio: {other_chars / total_chars * 100:.2f}%")

    # 8. Leakage Checks
    print("\n--- 8. Split and Benchmark Leakage Checks ---")
    # Load splits
    train_file = ROOT / "data/v3_2/splits/train.jsonl"
    val_file = ROOT / "data/v3_2/splits/validation.jsonl"
    test_file = ROOT / "data/v3_2/splits/test.jsonl"

    train_rows = [json.loads(line) for line in train_file.open("r", encoding="utf-8")]
    val_rows = [json.loads(line) for line in val_file.open("r", encoding="utf-8")]
    test_rows = [json.loads(line) for line in test_file.open("r", encoding="utf-8")]

    train_prompts = set(r["messages"][1]["content"].strip() for r in train_rows)
    val_prompts = set(r["messages"][1]["content"].strip() for r in val_rows)
    test_prompts = set(r["messages"][1]["content"].strip() for r in test_rows)

    train_val_leak = train_prompts & val_prompts
    train_test_leak = train_prompts & test_prompts
    val_test_leak = val_prompts & test_prompts

    print(f"  - Train vs Validation Leakage: {len(train_val_leak)}")
    print(f"  - Train vs Test Leakage: {len(train_test_leak)}")
    print(f"  - Validation vs Test Leakage: {len(val_test_leak)}")

    # Knowledge Preservation Benchmark Leakage
    kp_file = ROOT / "data/knowledge_preservation_test.json"
    if kp_file.exists():
        kp_data = json.loads(kp_file.read_text(encoding="utf-8"))
        kp_prompts = set(q["question"].strip() for q in kp_data)
        
        kp_train_leak = kp_prompts & train_prompts
        kp_val_leak = kp_prompts & val_prompts
        kp_test_leak = kp_prompts & test_prompts
        
        # Near-match check with KP benchmark
        kp_3grams = [get_3grams(p) for p in kp_prompts]
        near_kp_leak = 0
        for tp in train_prompts:
            tg = get_3grams(tp)
            for kpg in kp_3grams:
                if jaccard_sim(tg, kpg) > 0.8:
                    near_kp_leak += 1

        print(f"  - Knowledge Preservation Benchmark Exact Leakage into Train: {len(kp_train_leak)}")
        print(f"  - Knowledge Preservation Benchmark Exact Leakage into Val: {len(kp_val_leak)}")
        print(f"  - Knowledge Preservation Benchmark Exact Leakage into Test: {len(kp_test_leak)}")
        print(f"  - Knowledge Preservation Benchmark Near-Leakage (Jaccard > 0.8): {near_kp_leak}")
    else:
        print("  - WARNING: knowledge_preservation_test.json not found!")

    # V3.1 test set leakage
    v3_1_test_file = ROOT / "data/splits/test.jsonl"
    if v3_1_test_file.exists():
        v3_1_test_rows = [json.loads(line) for line in v3_1_test_file.open("r", encoding="utf-8")]
        v3_1_test_prompts = set(r["messages"][1]["content"].strip() for r in v3_1_test_rows)
        v3_1_leak = v3_1_test_prompts & train_prompts
        print(f"  - V3.1 Test Set Exact Leakage into V3.2 Train: {len(v3_1_leak)}")

    print("\n=== AUDIT COMPLETED ===")

if __name__ == "__main__":
    audit_dataset()
