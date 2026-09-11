"""Conservative concept/text components. Lexical checks do not prove semantic independence."""
import re
from collections import defaultdict

def normalized(text):
    return ' '.join(re.findall(r'\w+', text.casefold().replace('_', ' ')))

def grams(text):
    w = normalized(text).split()
    return set(zip(w,w[1:],w[2:])) or {tuple(w)}

def similar(a,b):
    return len(a & b)/max(1,len(a | b)) >= .8

def components(rows):
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    prompts = [grams(r['messages'][1]['content']) for r in rows]
    answers = [grams(r['messages'][2]['content']) for r in rows]
    for i,r in enumerate(rows):
        family = normalized(r.get('concept_id') or 'unannotated')
        for j in range(i):
            if (family == normalized(rows[j].get('concept_id') or 'unannotated')
                or similar(prompts[i],prompts[j]) or similar(answers[i],answers[j])):
                parents[root(i)] = root(j)
    groups = defaultdict(list)
    for i,r in enumerate(rows):
        groups[root(i)].append(r)
    return list(groups.values())

def validate_splits(splits):
    rows = [r for v in splits.values() for r in v]
    owner = {id(r): k for k,v in splits.items() for r in v}
    crossing = sum(len({owner[id(r)] for r in g}) > 1 for g in components(rows))
    if crossing:
        raise ValueError(f'{crossing} concept/text components cross splits')
    return crossing

def grouped_split(rows):
    groups = components(rows)
    result = dict(train=[],validation=[],test=[])
    for g in list(groups):
        if any(not r.get('concept_id') for r in g):
            result['train'].extend(g)
            groups.remove(g)
    for split,fraction in (('test',.05),('validation',.10)):
        target = len(rows)*fraction
        while groups and len(result[split]) < target:
            g = min(groups,key=lambda g: abs(target-len(result[split])-len(g)))
            if result[split] and abs(target-len(result[split])-len(g)) >= abs(target-len(result[split])):
                break
            groups.remove(g)
            result[split].extend(g)
    result['train'].extend(r for g in groups for r in g)
    if any(not v for v in result.values()):
        raise ValueError('Insufficient independent concept groups')
    validate_splits(result)
    return result
