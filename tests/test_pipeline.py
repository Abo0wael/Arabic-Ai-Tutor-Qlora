"""Offline tests for leakage-sensitive cleaning, labels and evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.prepare_data import clean_rows, prepare
from src.evaluate import checks


class DataTests(unittest.TestCase):
    def test_split_keeps_identical_questions_together(self):
        class Tokenizer:
            eos_token = "<eos>"
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"] + messages[1]["content"] + "<assistant>"
        rows = [{"messages": [{"role": "system", "content": "Tutor"},
                               {"role": "user", "content": f"Question {i//2}"},
                               {"role": "assistant", "content": f"Answer {i}"}]} for i in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "data.json").write_text(json.dumps(rows), encoding="utf-8")
            cfg = dict(dataset_path=str(path / "data.json"), processed_dir=str(path / "prepared"),
                       seed=42, validation_fraction=0.1, split_strategy="chronological")
            with patch("src.prepare_data.tokenizer_for", return_value=Tokenizer()):
                data = prepare(cfg)
            train = {r["messages"][1]["content"] for r in data["train"]}
            valid = {r["messages"][1]["content"] for r in data["validation"]}
            self.assertFalse(train & valid)
            self.assertEqual(valid, {"Question 9"})
            self.assertEqual(len(data["train"]), 18)

    def test_invalid_and_duplicates(self):
        row = {"messages": [{"role": r, "content": c} for r, c in
                            zip(("system", "user", "assistant"), ("Tutor", "Question", "Answer"))]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.jsonl"
            path.write_text(json.dumps(row)+"\n"+json.dumps(row)+"\n{}\nnot-json\n", encoding="utf-8")
            rows, stats = clean_rows(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats, {"total": 4, "invalid": 2, "duplicates": 1})

    def test_language_and_empty_checks(self):
        self.assertTrue(checks("ما التعلم؟", "")["empty"])
        self.assertFalse(checks("ما التعلم؟", "This is English")["arabic_when_requested"])
        good = "تعريف: تعلم الأنماط. شرح: نتعلم من البيانات. مثال: تصنيف رسالة. الخلاصة: نختبر التعميم."
        self.assertIsNone(checks("ما التعلم؟", good)["expected_structure"])
        self.assertTrue(checks("اكتب تعريف وشرح ومثال وخلاصة", good)["expected_structure"])
        self.assertTrue(checks("ما التعلم؟", good)["arabic_when_requested"])
        self.assertTrue(checks("ما التعلم؟", "واحد اثنان ثلاثة أربعة "*20)["excessive_repetition"])

    def test_assistant_labels_and_padding(self):
        from src.train import AssistantCollator
        class Tokenizer:
            pad_token_id = 0
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text]}
        batch = AssistantCollator(Tokenizer(), 10)([
            {"prompt": "ab", "completion": "cd"}, {"prompt": "a", "completion": "c"}])
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, 99, 100], [-100, 99, -100, -100]])
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [1, 1, 0, 0]])
        with self.assertRaises(ValueError):
            AssistantCollator(Tokenizer(), 2)([{"prompt": "abc", "completion": "d"}])


if __name__ == "__main__":
    unittest.main()
