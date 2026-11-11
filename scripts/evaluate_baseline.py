"""Baseline check for OEG PII RU, run before any training.

1. Reproduce the base model's published per-type scores on pii_benchmark.
2. Score the same sentences after an ASR-like rewrite, to test whether call
   transcripts are a real gap. The rewrite is a proxy, not real ASR output.
3. Check train/benchmark overlap and measure CPU training speed.

Spans are compared at word level. "any_type_recall" counts a gold span as caught
when any predicted span of any type overlaps it: for anonymization, a wrong label
on a masked value is harmless, and a missed value is a leak.
"""
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "base_model"
TRAIN = ROOT / "assets" / "pii_train" / "train.csv"
TEST = ROOT / "assets" / "pii_benchmark" / "test.csv"
OUT = ROOT / "reports" / "baseline.json"
MAX_LEN, STRIDE, BATCH = 512, 128, 32

COARSE = {
    **dict.fromkeys(["FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"], "PERSON"),
    **dict.fromkeys(["COUNTRY", "REGION", "DISTRICT", "CITY", "STREET", "HOUSE"], "LOCATION"),
}
DIGITS = "ноль один два три четыре пять шесть семь восемь девять".split()
TEENS = ("десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать "
         "шестнадцать семнадцать восемнадцать девятнадцать").split()
TENS = "_ _ двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split()
HUNDREDS = "_ сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split()
SYMBOLS = {"@": "собака", ".": "точка", "-": "дефис", "_": "подчёркивание", "/": "слэш", ":": "двоеточие"}


def load(path):
    csv.field_size_limit(10_000_000)
    with open(path, encoding="utf-8", newline="") as f:
        return [
            {"text": r["text"], "tokens": json.loads(r["tokens"]), "tags": json.loads(r["ner_tags"])}
            for r in csv.DictReader(f)
        ]


def spans(tags):
    """BIO to (start, end, type), end exclusive. A stray I- opens a span."""
    out, start, kind = [], None, None
    for i, tag in enumerate(tags):
        prefix, _, name = tag.partition("-")
        if prefix == "B" or (prefix == "I" and name != kind):
            if kind:
                out.append((start, i, kind))
            start, kind = i, name
        elif prefix == "O":
            if kind:
                out.append((start, i, kind))
            start, kind = None, None
    if kind:
        out.append((start, len(tags), kind))
    return out


def merge_adjacent(items):
    """Join touching spans of one type, as the model card requires for predictions."""
    merged = []
    for s, e, k in sorted(items):
        if merged and merged[-1][2] == k and merged[-1][1] == s:
            merged[-1] = (merged[-1][0], e, k)
        else:
            merged.append((s, e, k))
    return merged


def speak_group(group):
    if len(group) == 1:
        return [DIGITS[int(group)]]
    if group[0] == "0":
        return [DIGITS[int(c)] for c in group]
    value, words = int(group), []
    if value >= 100:
        words.append(HUNDREDS[value // 100])
        value %= 100
        if value and value < 10:
            words.append(DIGITS[value])
            return words
    if 10 <= value < 20:
        words.append(TEENS[value - 10])
    elif value >= 20:
        words.append(TENS[value // 10])
        if value % 10:
            words.append(DIGITS[value % 10])
    return words


def speak_number(digits, rng):
    """Say a digit run the way people do: one by one, or in groups of two or three."""
    size = rng.choice([1, 2, 3]) if len(digits) > 1 else 1
    words = []
    for i in range(0, len(digits), size):
        words += speak_group(digits[i : i + size])
    return words


def rewrite_token(token, mode, rng):
    token = token.lower()
    if mode == "no_punct":
        return [p for p in [re.sub(r"[^\w@./:+-]", "", token).strip(".,:;-")] if p]
    # Sentence punctuation is never spoken. Symbols are said aloud only in addresses
    # such as e-mail, URL and IP; separators inside numbers and words are silent.
    token = re.sub(r"^\W+|\W+$", "", token)
    is_address = bool(re.search(r"@|https?:|www\.|\w\.[a-z]{2,}\b|\d\.\d+\.\d", token))
    pieces = []
    for part in re.findall(r"\d+|[^\W\d_]+|[@./:_-]", token):
        if part.isdigit():
            pieces += speak_number(part, rng)
        elif part in SYMBOLS:
            if is_address:
                pieces.append(SYMBOLS[part])
        else:
            pieces.append(part)
    return pieces


def build_variant(rows, mode, seed=42):
    rng = random.Random(seed)
    out = []
    for row in rows:
        tokens, tags = [], []
        for token, tag in zip(row["tokens"], row["tags"]):
            pieces = rewrite_token(token, mode, rng)
            prefix, _, name = tag.partition("-")
            for j, piece in enumerate(pieces):
                tokens.append(piece)
                if prefix == "O":
                    tags.append("O")
                else:
                    tags.append(f"B-{name}" if prefix == "B" and j == 0 else f"I-{name}")
        out.append({"tokens": tokens, "tags": tags})
    return out


@torch.inference_mode()
def predict(model, tok, rows):
    id2label = model.config.id2label
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]))
    preds = [None] * len(rows)
    started = time.perf_counter()
    for at in range(0, len(order), BATCH):
        ids = order[at : at + BATCH]
        enc = tok(
            [rows[i]["tokens"] or ["."] for i in ids],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            stride=STRIDE,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        owners = enc["overflow_to_sample_mapping"].tolist()
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
        labels = logits.argmax(-1).tolist()
        for n, i in enumerate(ids):
            preds[i] = ["O"] * len(rows[i]["tokens"])
        seen = defaultdict(set)
        for chunk, owner in enumerate(owners):
            i = ids[owner]
            for pos, word in enumerate(enc.word_ids(chunk)):
                if word is None or word in seen[i] or word >= len(preds[i]):
                    continue
                seen[i].add(word)
                preds[i][word] = id2label[labels[chunk][pos]]
        if at % (BATCH * 20) == 0:
            print(f"  {at}/{len(rows)} {time.perf_counter() - started:.0f}s", flush=True)
    return preds, time.perf_counter() - started


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(100 * p, 1), "r": round(100 * r, 1), "f1": round(200 * p * r / (p + r), 1) if p + r else 0.0}


def match(gold, pred, exact, fold=lambda k: k):
    """One-to-one greedy matching. Returns tp, fp, fn counters keyed by folded type."""
    tp, fp, fn = Counter(), Counter(), Counter()
    used = set()
    for gs, ge, gk in gold:
        hit = None
        for j, (ps, pe, pk) in enumerate(pred):
            if j in used or fold(pk) != fold(gk):
                continue
            if ((ps, pe) == (gs, ge)) if exact else (ps < ge and gs < pe):
                hit = j
                break
        if hit is None:
            fn[fold(gk)] += 1
        else:
            used.add(hit)
            tp[fold(gk)] += 1
    for j, (_, _, pk) in enumerate(pred):
        if j not in used:
            fp[fold(pk)] += 1
    return tp, fp, fn


def score(rows, preds):
    exact = [Counter(), Counter(), Counter()]
    common = [Counter(), Counter(), Counter()]
    fold = lambda k: COARSE.get(k, k)
    caught, total = Counter(), Counter()
    false_alarm_sentences = clean_sentences = 0
    for row, pred_tags in zip(rows, preds):
        gold = spans(row["tags"])
        pred = merge_adjacent(spans(pred_tags))
        for bucket, part in zip(exact, match(gold, pred, True)):
            bucket.update(part)
        keep_g = [s for s in gold if fold(s[2]) in ("PERSON", "LOCATION")]
        keep_p = [s for s in pred if fold(s[2]) in ("PERSON", "LOCATION")]
        for bucket, part in zip(common, match(keep_g, keep_p, False, fold)):
            bucket.update(part)
        for gs, ge, gk in gold:
            total[gk] += 1
            if any(ps < ge and gs < pe for ps, pe, _ in pred):
                caught[gk] += 1
        if not gold:
            clean_sentences += 1
            false_alarm_sentences += bool(pred)
    return {
        "exact_21_micro": f1(*(sum(c.values()) for c in exact)),
        "overlap_person_location": f1(*(sum(c.values()) for c in common)),
        "any_type_recall": round(100 * sum(caught.values()) / max(1, sum(total.values())), 1),
        "missed_spans": sum(total.values()) - sum(caught.values()),
        "gold_spans": sum(total.values()),
        "sentences_without_pii": clean_sentences,
        "false_alarm_sentences": false_alarm_sentences,
        "per_type_exact_f1": {
            k: f1(exact[0][k], exact[1][k], exact[2][k])["f1"] for k in sorted(total)
        },
        "per_type_any_recall": {k: round(100 * caught[k] / total[k], 1) for k in sorted(total)},
    }


def train_speed(tok, train_rows, label2id, steps=20, batch_size=16):
    model = AutoModelForTokenClassification.from_pretrained(MODEL, use_safetensors=True)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    rng = random.Random(42)
    times, lengths = [], []
    for _ in range(steps):
        rows = rng.sample(train_rows, batch_size)
        enc = tok(
            [r["tokens"] for r in rows],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(enc["input_ids"], -100)
        for b, row in enumerate(rows):
            last = None
            for pos, word in enumerate(enc.word_ids(b)):
                if word is not None and word != last:
                    labels[b, pos] = label2id[row["tags"][word]]
                last = word
        started = time.perf_counter()
        loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
        times.append(time.perf_counter() - started)
        lengths.append(enc["input_ids"].shape[1])
    per_step = sum(times[2:]) / len(times[2:])
    steps_per_epoch = len(train_rows) // batch_size + 1
    return {
        "seconds_per_step": round(per_step, 2),
        "sampled_batch_lengths": lengths,
        "steps_per_epoch": steps_per_epoch,
        "hours_per_epoch": round(per_step * steps_per_epoch / 3600, 2),
        "hours_10_epochs": round(per_step * steps_per_epoch * 10 / 3600, 1),
    }


def main():
    OUT.parent.mkdir(exist_ok=True)
    train_rows, test_rows = load(TRAIN), load(TEST)
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    train_texts = {norm(r["text"]) for r in train_rows}
    report = {
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "test_rows_also_in_train": sum(norm(r["text"]) in train_texts for r in test_rows),
    }
    print(report, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL, use_safetensors=True).eval()
    report["labels"] = len(model.config.id2label)
    report["parameters_million"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)

    variants = {
        "clean": test_rows,
        "no_punct_lower": build_variant(test_rows, "no_punct"),
        "asr_proxy": build_variant(test_rows, "asr"),
    }
    if "--asr-only" in sys.argv:
        rows = variants["asr_proxy"]
        preds, seconds = predict(model, tok, rows)
        result = score(rows, preds)
        result["samples"] = [" ".join(r["tokens"])[:200] for r in rows[:400:80]]
        (OUT.parent / "baseline_asr_v2.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print("done", flush=True)
        return
    report["variants"] = {}
    for name, rows in variants.items():
        print("variant", name, flush=True)
        preds, seconds = predict(model, tok, rows)
        result = score(rows, preds)
        result["inference_seconds"] = round(seconds, 1)
        result["sentences_per_second"] = round(len(rows) / seconds, 1)
        report["variants"][name] = result
        print(json.dumps({k: result[k] for k in list(result)[:7]}, ensure_ascii=False), flush=True)
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["asr_samples"] = [" ".join(r["tokens"])[:200] for r in variants["asr_proxy"][:400:80]]

    del model
    print("training speed", flush=True)
    report["train_speed_cpu"] = train_speed(tok, train_rows, {v: k for k, v in AutoModelForTokenClassification.from_pretrained(MODEL, use_safetensors=True).config.id2label.items()})
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["train_speed_cpu"]), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
