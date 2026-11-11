"""Fine-tune the base model on the N1X GPU. Keeps the base head and adds rows for new types.

Picks the best epoch and decoding threshold on dev only. pii_benchmark is not read.
"""
import json
import random
import shutil
import sys
import time

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from oeg_pii_lib import BASE_MODEL, DATA_V2 as DATA_DIR, MODEL_DIR, NEW_TYPES, REPORTS, decode, predict_probs, score

EPOCHS, LR, BATCH, MAX_LEN, O_WEIGHT, SEED = 4, 2e-5, 32, 256, 0.5, 42
# Gentle mode: after each step, rest for THROTTLE times the step's own duration.
# 1.0 keeps the GPU busy about half the time, so the laptop runs cooler and the run takes about twice as long.
THROTTLE = next((float(a.split("=", 1)[1]) for a in sys.argv if a.startswith("--throttle=")), 0.0)
THRESHOLDS = [0.5, 0.3, 0.2, 0.1, 0.05]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extend_head(model):
    old = model.config.id2label
    labels = [old[i] for i in range(len(old))]
    for name in NEW_TYPES:
        labels += [f"B-{name}", f"I-{name}"]
    new_head = torch.nn.Linear(model.classifier.in_features, len(labels))
    with torch.no_grad():
        new_head.weight[: len(old)] = model.classifier.weight
        new_head.bias[: len(old)] = model.classifier.bias
    model.classifier = new_head
    model.num_labels = len(labels)
    model.config.num_labels = len(labels)
    model.config.id2label = dict(enumerate(labels))
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}
    return model


def encode(rows, tok, label2id):
    out = []
    for at in range(0, len(rows), 2000):
        part = rows[at : at + 2000]
        enc = tok([r["tokens"] for r in part], is_split_into_words=True, truncation=True, max_length=MAX_LEN)
        for b, row in enumerate(part):
            labels, last = [], None
            for word in enc.word_ids(b):
                labels.append(label2id[row["tags"][word]] if word is not None and word != last else -100)
                last = word
            out.append((enc["input_ids"][b], labels))
    return out


def batches(data, rng):
    """Group rows of similar length, so padding stays small and steps stay fast."""
    index = list(range(len(data)))
    rng.shuffle(index)
    pools = [sorted(index[i : i + BATCH * 50], key=lambda k: len(data[k][0])) for i in range(0, len(index), BATCH * 50)]
    groups = [pool[i : i + BATCH] for pool in pools for i in range(0, len(pool), BATCH)]
    rng.shuffle(groups)
    return groups


def collate(group, data, pad_id):
    width = max(len(data[k][0]) for k in group)
    ids = torch.full((len(group), width), pad_id)
    labels = torch.full((len(group), width), -100)
    for b, k in enumerate(group):
        x, y = data[k]
        ids[b, : len(x)] = torch.tensor(x)
        labels[b, : len(y)] = torch.tensor(y)
    return ids, (ids != pad_id).long(), labels


def evaluate_dev(model, tok, dev):
    model.eval()
    id2label = model.config.id2label
    result = {}
    all_probs = {name: predict_probs(model, tok, rows) for name, rows in dev.items()}
    for threshold in THRESHOLDS:
        per_set, recall, precision, f1 = {}, [], [], []
        for name, rows in dev.items():
            # The model alone: noisy rules must not decide which epoch or threshold wins.
            s = score(rows, decode(all_probs[name], id2label, threshold))
            per_set[name] = {"any_type": s["any_type"], "exact": s["exact"]}
            recall.append(s["any_type"]["r"])
            precision.append(s["any_type"]["p"])
            f1.append(s["any_type"]["f1"])
        result[threshold] = {"sets": per_set, "min_recall": min(recall), "min_precision": min(precision),
                             "min_f1": min(f1), "mean_f1": round(sum(f1) / len(f1), 3)}
    model.train()
    return result


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    REPORTS.mkdir(exist_ok=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, use_safetensors=True)
    model = extend_head(model).to("cuda")
    label2id = model.config.label2id

    started = time.perf_counter()
    train_rows = read_jsonl(DATA_DIR / "train.jsonl")
    if "--smoke" in sys.argv:
        train_rows = train_rows[:3200]
    data = encode(train_rows, tok, label2id)
    dev = {n: read_jsonl(DATA_DIR / f"dev_{n}.jsonl") for n in ("clean", "no_punct", "asr")}
    if "--smoke" in sys.argv:
        dev = {n: rows[:200] for n, rows in dev.items()}
    print(f"encoded {len(data)} rows in {time.perf_counter() - started:.0f}s", flush=True)

    weights = torch.ones(len(label2id), device="cuda")
    weights[label2id["O"]] = O_WEIGHT
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(SEED)
    epochs = 1 if "--smoke" in sys.argv else EPOCHS
    total = sum(len(batches(data, random.Random(SEED))) for _ in range(epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: max(0.0, 1 - s / total))
    pad_id = tok.pad_token_id

    history, best = [], None
    step, t0 = 0, time.perf_counter()
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for group in batches(data, rng):
            step_start = time.perf_counter()
            ids, mask, labels = collate(group, data, pad_id)
            ids, mask, labels = ids.to("cuda"), mask.to("cuda"), labels.to("cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            running += loss.detach().item()
            if THROTTLE:
                time.sleep((time.perf_counter() - step_start) * THROTTLE)
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}/{total} loss {running / 200:.4f} "
                      f"{(time.perf_counter() - t0) / step:.3f}s/step", flush=True)
                running = 0.0
        result = evaluate_dev(model, tok, dev)
        history.append({"epoch": epoch + 1, "dev": result, "minutes": round((time.perf_counter() - t0) / 60, 1)})
        print(json.dumps({"epoch": epoch + 1, "t0.5": result[0.5]["sets"]}, ensure_ascii=False), flush=True)
        # Judge an epoch by its weakest dev set, on F1: comparing recall first let a 0.01 point
        # recall gain outvote a large precision loss and kept an undertrained epoch.
        threshold = max(result, key=lambda t: (result[t]["min_f1"], result[t]["mean_f1"]))
        key = (result[threshold]["min_f1"], result[threshold]["mean_f1"])
        if best is None or key > best["key"]:
            best = {"key": key, "epoch": epoch + 1, "threshold": threshold}
            tmp = MODEL_DIR.with_name(MODEL_DIR.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            model.save_pretrained(tmp, safe_serialization=True)
            tok.save_pretrained(tmp)
            shutil.rmtree(MODEL_DIR, ignore_errors=True)
            tmp.rename(MODEL_DIR)
            (MODEL_DIR / "selection.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        (REPORTS / "train_history.json").write_text(json.dumps({"history": history, "best": best}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("best", best, f"total {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
