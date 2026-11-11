"""Estimate how much personal data is left in the 'clean' chat rows of dataset v2.

The base model flags a random sample on the CPU with few threads, so the laptop stays
free. A flag is only a candidate: the flagged rows are printed for a human look.
"""
import json
import random
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from oeg_pii_lib import BASE_MODEL, REPORTS, ROOT, decode, predict_probs, spans

SAMPLE = 1500


def main():
    torch.set_num_threads(4)
    rows = []
    with open(ROOT / "data_v2" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if not spans(row["tags"]) and len(row["tokens"]) >= 4 and not any(c.isdigit() for t in row["tokens"] for c in t):
                rows.append(row)
    print("candidate rows", len(rows), flush=True)
    sample = random.Random(3).sample(rows, min(SAMPLE, len(rows)))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, use_safetensors=True).eval()
    pred = decode(predict_probs(model, tok, sample, device="cpu", batch_size=32), model.config.id2label, 0.5)
    flagged, kinds = [], Counter()
    for row, found in zip(sample, pred):
        if found:
            parts = [f"{k}={' '.join(row['tokens'][s:e])}" for s, e, k in found]
            kinds.update(k for _, _, k in found)
            flagged.append({"found": parts, "text": " ".join(row["tokens"])[:140]})
    report = {"sample": len(sample), "flagged_rows": len(flagged),
              "flagged_pct": round(100 * len(flagged) / len(sample), 2),
              "by_type": kinds.most_common(), "examples": flagged[:60]}
    (REPORTS / "chat_noise.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("sample", "flagged_rows", "flagged_pct", "by_type")}, ensure_ascii=False))
    for item in flagged[:40]:
        print(item["found"], "::", item["text"])


if __name__ == "__main__":
    main()
