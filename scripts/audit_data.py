"""Audit the training data and its relation to the test sets. CPU only, no model.

Writes reports/data_audit.json.
"""
import json
import re
from collections import Counter, defaultdict

from oeg_pii_lib import DATA_DIR, REPORTS, TEST_CSV, TRAIN_CSV, load_csv, load_handmade, spans


def norm(tokens):
    return " ".join(tokens).lower()


def values(rows):
    out = defaultdict(Counter)
    for r in rows:
        for s, e, k in spans(r["tags"]):
            out[k][norm(r["tokens"][s:e])] += 1
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    source = load_csv(TRAIN_CSV)
    test = load_csv(TEST_CSV)
    built = read_jsonl(DATA_DIR / "train.jsonl")
    handmade = load_handmade()
    report = {}

    lengths = sorted(len(r["tokens"]) for r in source)
    report["source"] = {
        "rows": len(source),
        "rows_without_pii": sum(not spans(r["tags"]) for r in source),
        "tokens_median": lengths[len(lengths) // 2],
        "tokens_p95": lengths[int(len(lengths) * 0.95)],
        "exact_duplicate_rows": len(source) - len({norm(r["tokens"]) for r in source}),
    }

    templates = Counter(re.sub(r"\d+", "#", norm(r["tokens"])) for r in source)
    report["source"]["top_shapes_after_masking_digits"] = [[t[:90], c] for t, c in templates.most_common(8)]
    report["source"]["distinct_shapes"] = len(templates)

    # Label conflicts: the same word sequence is tagged as an entity in one place and as O in another.
    tagged = defaultdict(Counter)
    for r in source:
        for tok, tag in zip(r["tokens"], r["tags"]):
            tagged[tok.lower()][tag.partition("-")[2] or "O"] += 1
    conflicts = []
    for tok, c in tagged.items():
        total = sum(c.values())
        if total >= 20 and len(tok) >= 3 and c["O"] and c["O"] < total and min(c["O"], total - c["O"]) / total >= 0.1:
            conflicts.append([tok, dict(c.most_common(3))])
    conflicts.sort(key=lambda x: -sum(x[1].values()))
    report["source"]["ambiguous_tokens"] = len(conflicts)
    report["source"]["ambiguous_examples"] = conflicts[:25]

    # Spans holding attached punctuation or spaced-out tokens, which affects real input.
    spaced = sum(bool(re.search(r"\w \. \w|\w @ \w", " ".join(r["tokens"]))) for r in source)
    report["source"]["rows_with_split_email_or_domain"] = spaced

    src_vals, test_vals = values(source), values(test)
    per_type = {}
    for k in sorted(set(src_vals) | set(test_vals)):
        tv = test_vals.get(k, Counter())
        sv = src_vals.get(k, Counter())
        seen = sum(c for v, c in tv.items() if v in sv)
        per_type[k] = {
            "train_spans": sum(sv.values()),
            "train_unique": len(sv),
            "test_spans": sum(tv.values()),
            "test_values_seen_in_train_pct": round(100 * seen / sum(tv.values()), 1) if tv else None,
            "top_train_values": [v for v, _ in sv.most_common(5)],
        }
    report["per_type"] = per_type

    # Script of name spans in train and test: foreign names were a visible miss.
    def script_share(rows):
        latin = cyr = 0
        for r in rows:
            for s, e, k in spans(r["tags"]):
                if k in ("FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"):
                    text = " ".join(r["tokens"][s:e])
                    latin += bool(re.search(r"[A-Za-z]", text))
                    cyr += bool(re.search(r"[А-Яа-яЁё]", text))
        return {"latin": latin, "cyrillic": cyr}
    report["names_script"] = {"train": script_share(source), "test": script_share(test)}

    built_counts = Counter()
    built_unique = defaultdict(set)
    cue_before = defaultdict(Counter)
    for r in built:
        for s, e, k in spans(r["tags"]):
            built_counts[k] += 1
            built_unique[k].add(norm(r["tokens"][s:e]))
            if s:
                cue_before[k][r["tokens"][s - 1].lower()] += 1
    report["built"] = {
        "rows": len(built),
        "rows_without_pii": sum(not spans(r["tags"]) for r in built),
        "spans_per_type": dict(built_counts.most_common()),
        "unique_per_type": {k: len(v) for k, v in built_unique.items()},
        "top_previous_word": {k: cue_before[k].most_common(5) for k in
                              ["TG_NICK", "MAX_NICK", "DATE_OF_BIRTH", "BANK_ACCOUNT", "BIK", "KPP", "CORR_ACCOUNT"]},
    }

    punct = sum(bool(re.search(r"[.,!?]", " ".join(r["tokens"]))) for r in built)
    digit_rows = sum(any(re.search(r"\d", t) for t in r["tokens"]) for r in built)
    report["built"]["rows_with_punctuation"] = punct
    report["built"]["rows_with_digits"] = digit_rows

    report["handmade"] = {"rows": len(handmade), "spans": sum(len(spans(r["tags"])) for r in handmade)}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_audit.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("source", "names_script", "built")}, ensure_ascii=False, indent=0)[:9000])
    print(json.dumps({k: {x: v[x] for x in ("train_spans", "train_unique", "test_spans", "test_values_seen_in_train_pct")}
                      for k, v in per_type.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
