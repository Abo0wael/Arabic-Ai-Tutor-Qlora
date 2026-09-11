"""Offline evidence, without claiming automatic scientific correctness."""
import json
import hashlib
from collections import Counter
from src.utils import ROOT, config, tokenizer_for, SYSTEM_PROMPT, save_json
from src.prepare_data import clean_rows
from src.dataset_integrity import validate_splits

def main():
    cfg=config('configs/training_config.yaml')
    folder=ROOT/cfg['split_dir']
    splits={s:[json.loads(l) for l in (folder/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ('train','validation','test')}
    source=ROOT/cfg['dataset_path']
    rows=[json.loads(l) for l in source.read_text(encoding='utf-8').splitlines()]
    _,stats=clean_rows(source)
    assert stats['invalid']==0 and stats['duplicates']==0, stats
    assert Counter(json.dumps(r,sort_keys=True) for r in rows)==Counter(json.dumps(r,sort_keys=True) for v in splits.values() for r in v)
    assert all(r['messages'][0]['content']==SYSTEM_PROMPT for r in rows)
    tokenizer=tokenizer_for(cfg)
    lengths=[len(tokenizer(tokenizer.apply_chat_template(r['messages'][:2],tokenize=False,add_generation_prompt=True,enable_thinking=False)+r['messages'][2]['content']+tokenizer.eos_token,add_special_tokens=False)['input_ids']) for r in rows]
    report=dict(counts={k:len(v) for k,v in splits.items()},structure=stats,
      crossing_concept_or_text_components=validate_splits(splits),
      near_duplicate_threshold='question OR answer word trigram Jaccard >= 0.8',
      max_tokens=max(lengths),over_limit=sum(n>cfg['max_seq_length'] for n in lengths),
      behaviors={k:dict(Counter(r['behavior'] for r in v)) for k,v in splits.items()},
      concepts={k:sorted({r.get('concept_id') or 'unannotated' for r in v}) for k,v in splits.items()},
      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
      warnings=['Unannotated examples are retained in training; manually annotate them before claiming semantic independence.',
                'Scientific accuracy and language quality require human review; lexical checks are not proof.',
                'Old V3.1 reports are legacy heuristic results, not independent factual scores.'],
      training_started=False)
    save_json(ROOT/'data/v3_2/readiness_report.json',report)
    print(json.dumps(report,ensure_ascii=True))
    assert report['over_limit']==0

if __name__=='__main__':
    main()
