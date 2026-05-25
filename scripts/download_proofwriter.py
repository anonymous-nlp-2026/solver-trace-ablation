import json
import random
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from datasets import load_dataset

print("Loading ProofWriter...")
ds = load_dataset('tasksource/proofwriter')
print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}, Val: {len(ds['validation'])}")

# Quick stats on train split only
print("\nSampling stats from train split...")
configs = {}
depths = {}
answers = {}
for i, ex in enumerate(ds['train']):
    configs[ex['config']] = configs.get(ex['config'], 0) + 1
    depths[ex['QDep']] = depths.get(ex['QDep'], 0) + 1
    answers[ex['answer']] = answers.get(ex['answer'], 0) + 1
    
print(f"Configs: {sorted(configs.items(), key=lambda x: -x[1])}")
print(f"Depths: {sorted(depths.items())}")
print(f"Answers: {answers}")

# Filter: depth >= 1, answer in {True, False, Unknown}
# Use train split for train data, test split for eval data
valid_answers = {'True', 'False', 'Unknown'}

train_candidates = [
    ex for ex in ds['train'] 
    if ex['QDep'] >= 1 and ex['answer'] in valid_answers
]
test_candidates = [
    ex for ex in ds['test']
    if ex['QDep'] >= 1 and ex['answer'] in valid_answers
]

print(f"\nFiltered train candidates (depth>=1): {len(train_candidates)}")
print(f"Filtered test candidates (depth>=1): {len(test_candidates)}")

# Sample
random.seed(42)
train_sample = random.sample(train_candidates, min(500, len(train_candidates)))
test_sample = random.sample(test_candidates, min(100, len(test_candidates)))

# Read FOLIO format to match
with open('/root/solver-trace-ablation/data/train_500_seed42.jsonl') as f:
    folio_ex = json.loads(f.readline())
print(f"\nFOLIO format keys: {list(folio_ex.keys())}")

def convert(ex, idx, prefix="pw"):
    """Convert ProofWriter example to FOLIO-compatible format."""
    # Parse theory into individual premises (sentences)
    theory = ex['theory'].strip()
    premises = [s.strip() for s in theory.split('.') if s.strip()]
    premises = [s + '.' for s in premises]  # re-add periods
    
    return {
        "id": f"{prefix}_{ex['id']}",
        "context": theory,
        "question": ex['question'],
        "answer": ex['answer'].lower(),  # FOLIO uses lowercase
        "fol_premises": [],  # ProofWriter has no FOL annotations
        "fol_conclusion": "",  # ProofWriter has no FOL annotations
        "depth": int(ex['QDep']),
        "source": "proofwriter",
        "proofwriter_meta": {
            "config": ex['config'],
            "maxD": ex['maxD'],
            "NFact": ex['NFact'],
            "NRule": ex['NRule'],
            "QLen": ex['QLen'],
            "allProofs": ex.get('allProofs', '')
        }
    }

# Convert and write
out_dir = '/root/solver-trace-ablation/data'

train_out = os.path.join(out_dir, 'proofwriter_real_train_500.jsonl')
with open(train_out, 'w') as f:
    for i, ex in enumerate(train_sample):
        f.write(json.dumps(convert(ex, i, "pw_train"), ensure_ascii=False) + '\n')
print(f"Wrote {len(train_sample)} examples to {train_out}")

eval_out = os.path.join(out_dir, 'proofwriter_real_eval_100.jsonl')
with open(eval_out, 'w') as f:
    for i, ex in enumerate(test_sample):
        f.write(json.dumps(convert(ex, i, "pw_eval"), ensure_ascii=False) + '\n')
print(f"Wrote {len(test_sample)} examples to {eval_out}")

# Verify
print("\n--- Verification ---")
for fpath in [train_out, eval_out]:
    with open(fpath) as f:
        lines = f.readlines()
    print(f"{fpath}: {len(lines)} lines")
    d = json.loads(lines[0])
    print(f"  Keys: {list(d.keys())}")
    print(f"  ID: {d['id']}")
    print(f"  Context[:100]: {d['context'][:100]}")
    print(f"  Question: {d['question']}")
    print(f"  Answer: {d['answer']}")
    print(f"  Depth: {d['depth']}")

# Check answer distribution in samples
for name, fpath in [("train", train_out), ("eval", eval_out)]:
    ans = {}
    with open(fpath) as f:
        for line in f:
            d = json.loads(line)
            ans[d['answer']] = ans.get(d['answer'], 0) + 1
    print(f"\n{name} answer dist: {ans}")

