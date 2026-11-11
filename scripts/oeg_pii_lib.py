"""Shared code for OEG PII RU: data loading, speech rewrite, span scoring, prediction, rules."""
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "assets" / "base_model"
TRAIN_CSV = ROOT / "assets" / "pii_train" / "train.csv"
TEST_CSV = ROOT / "assets" / "pii_benchmark" / "test.csv"
HANDMADE = ROOT / "tests" / "handmade.txt"
DATA_DIR = ROOT / "data"
DATA_V2 = ROOT / "data_v2"
HANDMADE_V2 = ROOT / "tests" / "handmade_v2.txt"
MODEL_DIR = ROOT / "models" / "oeg-pii-v1"
REPORTS = ROOT / "reports"

NEW_TYPES = ["DATE_OF_BIRTH", "BANK_ACCOUNT", "CORR_ACCOUNT", "BIK", "KPP", "TG_NICK", "MAX_NICK"]

DIGITS = "ноль один два три четыре пять шесть семь восемь девять".split()
TEENS = ("десять одиннадцать двенадцать тринадцать четырнадцать пятнадцать "
         "шестнадцать семнадцать восемнадцать девятнадцать").split()
TENS = "_ _ двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split()
HUNDREDS = "_ сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split()
SYMBOLS = {"@": "собака", ".": "точка", "-": "дефис", "_": "подчёркивание", "/": "слэш", ":": "двоеточие"}
TRANSLIT = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "ж",
    "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т",
    "u": "у", "v": "в", "w": "в", "x": "кс", "y": "й", "z": "з",
}


def load_csv(path):
    csv.field_size_limit(10_000_000)
    with open(path, encoding="utf-8", newline="") as f:
        return [
            {"text": r["text"], "tokens": json.loads(r["tokens"]), "tags": json.loads(r["ner_tags"])}
            for r in csv.DictReader(f)
        ]


def load_handmade(path=HANDMADE):
    """Parse lines like 'мой номер {PHONE:8 916 555 12 34}'. Lines starting with # are comments."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens, tags, at = [], [], 0
        for m in re.finditer(r"\{([A-Z_]+):([^{}]+)\}", line):
            for word in line[at : m.start()].split():
                tokens.append(word)
                tags.append("O")
            for j, word in enumerate(m.group(2).split()):
                tokens.append(word)
                tags.append(("B-" if j == 0 else "I-") + m.group(1))
            at = m.end()
        for word in line[at:].split():
            tokens.append(word)
            tags.append("O")
        rows.append({"tokens": tokens, "tags": tags})
    return rows


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
    token = re.sub(r"^\W+|\W+$", "", token) if not token.startswith("@") else "@" + re.sub(r"^\W+|\W+$", "", token)
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


def build_variant(rows, mode, seed=42, translit_prob=0.0):
    rng = random.Random(seed)
    out = []
    for row in rows:
        tokens, tags = [], []
        for token, tag in zip(row["tokens"], row["tags"]):
            pieces = rewrite_token(token, mode, rng)
            prefix, _, name = tag.partition("-")
            if name in ("EMAIL", "URL", "TG_NICK", "MAX_NICK") and rng.random() < translit_prob:
                pieces = ["".join(TRANSLIT.get(c, c) for c in p) for p in pieces]
            for j, piece in enumerate(pieces):
                tokens.append(piece)
                if prefix == "O":
                    tags.append("O")
                else:
                    tags.append(f"B-{name}" if prefix == "B" and j == 0 else f"I-{name}")
        out.append({"tokens": tokens, "tags": tags})
    return out


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
    merged = []
    for s, e, k in sorted(items):
        if merged and merged[-1][2] == k and merged[-1][1] == s:
            merged[-1] = (merged[-1][0], e, k)
        else:
            merged.append((s, e, k))
    return merged


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"p": round(100 * p, 2), "r": round(100 * r, 2), "f1": round(200 * p * r / (p + r), 2) if p + r else 0.0}


def score(rows, pred_spans):
    """Word-level scoring. pred_spans holds a merged span list per row.

    any_type: a gold span is caught if any predicted span overlaps it, and a predicted
    span is correct if it overlaps any gold span. For masking this is the metric that
    matters: a wrong label is harmless, a miss is a leak.
    exact: same type and same boundaries, one-to-one.
    """
    exact_tp, exact_fp, exact_fn = Counter(), Counter(), Counter()
    caught, total = Counter(), Counter()
    pred_ok = pred_bad = clean = alarms = 0
    missed = []
    for row, pred in zip(rows, pred_spans):
        gold = spans(row["tags"])
        used = set()
        for g in gold:
            hit = next((j for j, p in enumerate(pred) if j not in used and p == g), None)
            if hit is None:
                exact_fn[g[2]] += 1
            else:
                used.add(hit)
                exact_tp[g[2]] += 1
        for j, p in enumerate(pred):
            if j not in used:
                exact_fp[p[2]] += 1
        for gs, ge, gk in gold:
            total[gk] += 1
            if any(ps < ge and gs < pe for ps, pe, _ in pred):
                caught[gk] += 1
            elif len(missed) < 40:
                missed.append({"type": gk, "value": " ".join(row["tokens"][gs:ge]), "text": " ".join(row["tokens"])[:160]})
        for ps, pe, _ in pred:
            if any(ps < ge and gs < pe for gs, ge, _ in gold):
                pred_ok += 1
            else:
                pred_bad += 1
        if not gold:
            clean += 1
            alarms += bool(pred)
    n_gold = sum(total.values())
    return {
        "any_type": _prf(pred_ok, pred_bad, n_gold - sum(caught.values())),
        "exact": _prf(sum(exact_tp.values()), sum(exact_fp.values()), sum(exact_fn.values())),
        "gold_spans": n_gold,
        "missed_spans": n_gold - sum(caught.values()),
        "false_spans": pred_bad,
        "sentences_without_pii": clean,
        "false_alarm_sentences": alarms,
        "any_recall_by_type": {k: round(100 * caught[k] / total[k], 1) for k in sorted(total)},
        "exact_f1_by_type": {
            k: _prf(exact_tp[k], exact_fp[k], exact_fn[k])["f1"] for k in sorted(total)
        },
        "missed_examples": missed,
    }


def luhn_ok(d):
    total = 0
    for i, c in enumerate(reversed(d)):
        n = int(c)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def inn_ok(d):
    def check(digits, coefs):
        return sum(int(a) * b for a, b in zip(digits, coefs)) % 11 % 10
    if len(d) == 10:
        return check(d, [2, 4, 10, 3, 5, 9, 4, 6, 8]) == int(d[9])
    if len(d) == 12:
        return (check(d, [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == int(d[10])
                and check(d, [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]) == int(d[11]))
    return False


def snils_ok(d):
    if len(d) != 11:
        return False
    total = sum(int(c) * (9 - i) for i, c in enumerate(d[:9]))
    while total > 101:
        total %= 101
    control = 0 if total in (100, 101) else total
    return control == int(d[9:])


_SPOKEN = {w: 1 for w in DIGITS} | {w: 2 for w in TEENS} | {w: 2 for w in TENS if w != "_"} | {w: 3 for w in HUNDREDS if w != "_"}
_NICK = re.compile(r"^@[a-z0-9_]{4,32}$", re.I)
_LINK = re.compile(r"(?:t\.me|telegram\.me|max\.ru)/@?[\w-]+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"^(?:\+?7|8)\d{10}$")


def spoken_digit_count(words):
    """Digits said by a run of number words: 'сто двадцать три' is 3, 'двадцать три' is 2."""
    count, i = 0, 0
    while i < len(words):
        size = _SPOKEN[words[i]]
        j = i + 1
        if size == 3 and j < len(words) and _SPOKEN[words[j]] == 2:
            j += 1
        if size >= 2 and j < len(words) and words[j] in DIGITS[1:] and words[j - 1] not in TEENS:
            j += 1
        count += size
        i = j
    return count


def rule_spans(tokens, min_spoken_digits=7):
    """High-precision rules for what patterns and checksums can decide alone."""
    found = []
    low = [re.sub(r"^[^\w@+]+|[^\w]+$", "", t.lower()) for t in tokens]
    for i, t in enumerate(low):
        if _NICK.match(t):
            found.append((i, i + 1, "TG_NICK"))
        elif _LINK.search(t):
            found.append((i, i + 1, "MAX_NICK" if "max.ru" in t else "TG_NICK"))
        elif _EMAIL.search(t):
            found.append((i, i + 1, "EMAIL"))
    i = 0
    while i < len(low):
        if low[i] in _SPOKEN:
            j = i
            while j < len(low) and low[j] in _SPOKEN:
                j += 1
            if spoken_digit_count(low[i:j]) >= min_spoken_digits:
                found.append((i, j, "SPOKEN_NUMBER"))
            i = j
        else:
            i += 1
    i = 0
    while i < len(tokens):
        if re.fullmatch(r"[+(]?\d[\d()\-.]*", low[i] or tokens[i]):
            j, digits = i, ""
            while j < len(tokens) and re.fullmatch(r"[+(]?\d[\d()\-.]*", low[j] or tokens[j]) and len(digits) < 20:
                digits += re.sub(r"\D", "", low[j] or tokens[j])
                j += 1
            kind = None
            if _PHONE.match(digits):
                kind = "PHONE"
            elif len(digits) == 16 and luhn_ok(digits):
                kind = "CREDIT_CARD"
            elif len(digits) in (10, 12) and inn_ok(digits):
                kind = "INN"
            elif len(digits) == 11 and snils_ok(digits):
                kind = "SNILS"
            elif len(digits) == 20:
                kind = "CORR_ACCOUNT" if digits.startswith("301") else "BANK_ACCOUNT"
            if kind:
                found.append((i, j, kind))
            i = j
        else:
            i += 1
    return found


def combine(model_spans, extra):
    out = list(model_spans)
    for s, e, k in extra:
        if not any(ps < e and s < pe for ps, pe, _ in model_spans):
            out.append((s, e, k))
    return sorted(out)


def predict_probs(model, tok, rows, device="cuda", batch_size=64, max_len=512, stride=128):
    """Per row, per word: (best non-O label id, P(O), P(best non-O)). First subtoken decides."""
    import torch

    o_id = model.config.label2id["O"]
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]))
    results = [None] * len(rows)
    with torch.inference_mode():
        for at in range(0, len(order), batch_size):
            ids = order[at : at + batch_size]
            enc = tok(
                [rows[i]["tokens"] or ["."] for i in ids],
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                stride=stride,
                return_overflowing_tokens=True,
                padding=True,
                return_tensors="pt",
            )
            owners = enc["overflow_to_sample_mapping"].tolist()
            input_ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
            chunks = []
            for k in range(0, input_ids.shape[0], batch_size):
                with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(input_ids=input_ids[k : k + batch_size], attention_mask=mask[k : k + batch_size]).logits
                chunks.append(logits.float().softmax(-1))
            probs = torch.cat(chunks)
            p_o = probs[:, :, o_id].tolist()
            masked = probs.clone()
            masked[:, :, o_id] = -1
            best_p, best_id = masked.max(-1)
            best_p, best_id = best_p.tolist(), best_id.tolist()
            seen = defaultdict(set)
            for i in ids:
                results[i] = [(o_id, 1.0, 0.0)] * len(rows[i]["tokens"])
            for chunk, owner in enumerate(owners):
                i = ids[owner]
                for pos, word in enumerate(enc.word_ids(chunk)):
                    if word is None or word in seen[i] or word >= len(results[i]):
                        continue
                    seen[i].add(word)
                    results[i][word] = (best_id[chunk][pos], p_o[chunk][pos], best_p[chunk][pos])
    return results


def decode(probs, id2label, threshold):
    """A word leaves O when P(not O) is at least the threshold."""
    out = []
    for row in probs:
        tags = [id2label[b] if (1 - po) >= threshold else "O" for b, po, _ in row]
        out.append(merge_adjacent(spans(tags)))
    return out
