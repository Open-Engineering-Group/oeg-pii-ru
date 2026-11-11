"""Check that the N1X GPU can train the base model, and measure how fast.

Uses the same seed and batches as the CPU measurement in evaluate_baseline.py,
so the numbers compare directly. Writes reports/gpu_check.json.
"""
import csv
import json
import platform
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "base_model"
TRAIN = ROOT / "assets" / "pii_train" / "train.csv"
OUT = ROOT / "reports" / "gpu_check.json"
STEPS, BATCH, MAX_LEN = 20, 16, 512


def load_rows():
    csv.field_size_limit(10_000_000)
    with open(TRAIN, encoding="utf-8", newline="") as f:
        return [{"tokens": json.loads(r["tokens"]), "tags": json.loads(r["ner_tags"])} for r in csv.DictReader(f)]


def run(rows, tok, dtype):
    model = AutoModelForTokenClassification.from_pretrained(MODEL, use_safetensors=True).to("cuda")
    model.train()
    label2id = {v: k for k, v in model.config.id2label.items()}
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    rng = random.Random(42)
    times, losses, lengths = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(STEPS):
        sample = rng.sample(rows, BATCH)
        enc = tok(
            [r["tokens"] for r in sample],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(enc["input_ids"], -100)
        for b, row in enumerate(sample):
            last = None
            for pos, word in enumerate(enc.word_ids(b)):
                if word is not None and word != last:
                    labels[b, pos] = label2id[row["tags"][word]]
                last = word
        ids, mask = enc["input_ids"].to("cuda"), enc["attention_mask"].to("cuda")
        labels = labels.to("cuda")
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
        losses.append(round(float(loss), 4))
        lengths.append(ids.shape[1])
    per_step = sum(times[2:]) / len(times[2:])
    steps_per_epoch = len(rows) // BATCH + 1
    result = {
        "seconds_per_step": round(per_step, 3),
        "minutes_per_epoch": round(per_step * steps_per_epoch / 60, 1),
        "hours_10_epochs": round(per_step * steps_per_epoch * 10 / 3600, 2),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "lengths": lengths,
        "first_last_loss": [losses[0], losses[-1]],
        "finite_losses": all(l == l and abs(l) != float("inf") for l in losses),
    }
    del model, optim
    torch.cuda.empty_cache()
    return result


def main():
    report = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    print(report, flush=True)
    OUT.parent.mkdir(exist_ok=True)
    if not torch.cuda.is_available():
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    report["device"] = torch.cuda.get_device_name(0)
    a = torch.randn(4000, 4000, device="cuda")
    started = time.perf_counter()
    (a @ a).sum().item()
    report["matmul_4000_seconds"] = round(time.perf_counter() - started, 3)
    print(report, flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = load_rows()
    tok = AutoTokenizer.from_pretrained(MODEL)
    report["fp32"] = run(rows, tok, None)
    print("fp32", report["fp32"], flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["bf16"] = run(rows, tok, torch.bfloat16)
    print("bf16", report["bf16"], flush=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
