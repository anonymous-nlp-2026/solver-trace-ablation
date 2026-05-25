"""
FOLIO 数据拆分脚本。

输入: tasksource/folio (HuggingFace) + 现有 phase0 数据
输出:
  - data/phase0_100.jsonl   — Phase 0 baseline (100条, 与原 proofwriter_fol_100.jsonl 相同)
  - data/train_500.jsonl    — 训练集 (500条, 与 phase0 不重叠)
  - data/test_100.jsonl     — Held-out 测试集 (100条, 与 phase0 和 train 均不重叠)
  - data/train_500_seed{42,123,456}.jsonl — 训练集的 3 种 shuffle
依赖: datasets, json
"""

import json
import os
import random
from collections import Counter

from datasets import load_dataset

OUTPUT_DIR = "/root/solver-trace-ablation/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE0_FILE = os.path.join(OUTPUT_DIR, "proofwriter_fol_100.jsonl")


def parse_fol_premises(fol_text):
    if not fol_text:
        return []
    return [l.strip() for l in fol_text.strip().split("\n") if l.strip()]


def estimate_depth(fol_premises):
    n = len(fol_premises)
    if n <= 2:
        return 0
    elif n <= 3:
        return 1
    elif n <= 5:
        return 2
    elif n <= 7:
        return 3
    else:
        return 5


def map_label(label):
    label = label.strip().lower()
    if label == "true":
        return "true"
    elif label == "false":
        return "false"
    return "unknown"


def process_example(example, idx):
    fol_premises = parse_fol_premises(example.get("premises-FOL", ""))
    fol_conclusion = (example.get("conclusion-FOL") or "").strip()
    if not fol_premises or not fol_conclusion:
        return None
    return {
        "id": f"folio_{example.get('example_id', idx)}",
        "context": example["premises"],
        "question": example["conclusion"],
        "answer": map_label(example["label"]),
        "fol_premises": fol_premises,
        "fol_conclusion": fol_conclusion,
        "depth": estimate_depth(fol_premises),
    }


def load_phase0_ids():
    ids = set()
    with open(PHASE0_FILE) as f:
        for line in f:
            obj = json.loads(line)
            ids.add(obj["id"])
    return ids


def write_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(data):>4d} examples -> {path}")


def print_stats(name, subset):
    answers = Counter(d["answer"] for d in subset)
    depths = Counter(d["depth"] for d in subset)
    avg_premises = sum(len(d["fol_premises"]) for d in subset) / len(subset)
    xor_count = sum(
        1 for d in subset
        if any("⊕" in p for p in d["fol_premises"]) or "⊕" in d["fol_conclusion"]
    )
    print(f"\n=== {name} ===")
    print(f"  Count: {len(subset)}")
    print(f"  Avg premises: {avg_premises:.1f}")
    print(f"  Answer dist: {dict(sorted(answers.items()))}")
    print(f"  Depth dist: {dict(sorted(depths.items()))}")
    print(f"  XOR samples: {xor_count}")


def main():
    # 1. Load phase0 IDs
    phase0_ids = load_phase0_ids()
    print(f"Phase 0 IDs loaded: {len(phase0_ids)}")

    # 2. Load full FOLIO
    print("Loading tasksource/folio...")
    ds = load_dataset("tasksource/folio")
    all_examples = []
    for split in ds:
        for i, ex in enumerate(ds[split]):
            processed = process_example(ex, f"{split}_{i}")
            if processed:
                all_examples.append(processed)
    print(f"Total valid examples: {len(all_examples)}")

    # 3. Split: phase0 vs remaining
    phase0 = [ex for ex in all_examples if ex["id"] in phase0_ids]
    remaining = [ex for ex in all_examples if ex["id"] not in phase0_ids]
    print(f"Phase 0 matched: {len(phase0)}, Remaining: {len(remaining)}")

    if len(phase0) != 100:
        print(f"WARNING: expected 100 phase0 matches, got {len(phase0)}")

    # 4. Shuffle remaining with seed 42 -> train 500 + test 100
    random.seed(42)
    random.shuffle(remaining)
    train = remaining[:500]
    test = remaining[500:600]

    # 5. Verify no overlap
    train_ids = {d["id"] for d in train}
    test_ids = {d["id"] for d in test}
    assert len(phase0_ids & train_ids) == 0, "phase0-train overlap!"
    assert len(phase0_ids & test_ids) == 0, "phase0-test overlap!"
    assert len(train_ids & test_ids) == 0, "train-test overlap!"
    print("Overlap check: PASSED (all disjoint)")

    # 6. Write main splits
    print("\nWriting splits:")
    write_jsonl(os.path.join(OUTPUT_DIR, "phase0_100.jsonl"), phase0)
    write_jsonl(os.path.join(OUTPUT_DIR, "train_500.jsonl"), train)
    write_jsonl(os.path.join(OUTPUT_DIR, "test_100.jsonl"), test)

    # 7. Multi-seed shuffles
    print("\nWriting seed shuffles:")
    for seed in [42, 123, 456]:
        shuffled = train.copy()
        random.seed(seed)
        random.shuffle(shuffled)
        write_jsonl(os.path.join(OUTPUT_DIR, f"train_500_seed{seed}.jsonl"), shuffled)

    # 8. Stats
    print_stats("Phase 0 (100)", phase0)
    print_stats("Train (500)", train)
    print_stats("Test (100)", test)

    print("\nDone.")


if __name__ == "__main__":
    main()
