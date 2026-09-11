import unittest
from src.dataset_integrity import components, validate_splits
from src.evaluate import checks

def row(concept, question, answer):
    return dict(concept_id=concept, messages=[dict(content='system'),dict(content=question),dict(content=answer)])

class IntegrityTests(unittest.TestCase):
    def test_alias_and_answer_grouping(self):
        rows=[row('self-attention','one','a shared long answer'),row('self_attention','two','different answer'),row('other','three','a shared long answer')]
        self.assertEqual(len(components(rows)),1)

    def test_crossing_rejected(self):
        with self.assertRaises(ValueError):
            validate_splits(dict(train=[row('lora','q','a')],test=[row('lora','other','b')]))

    def test_no_unrequested_structure_penalty(self):
        self.assertIsNone(checks('What is LoRA?', 'A low rank adaptation method.')['expected_structure'])
