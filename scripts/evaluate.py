"""Final measurement on data the training never saw: pii_benchmark and the handmade test.

Runs the base model, OEG PII RU alone and OEG PII RU with rules. Writes reports/final.json.
Results here must not be used to change thresholds or to pick a model.
"""
import json

from transformers import AutoModelForTokenClassification, AutoTokenizer

from oeg_pii_lib import (BASE_MODEL, DATA_V2, HANDMADE_V2, MODEL_DIR, REPORTS, TEST_CSV, build_variant, combine, decode,
                        load_csv, load_handmade, predict_probs, rule_spans, score, spans)


def train_values():
    seen = set()
    with open(DATA_V2 / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for s, e, k in spans(row["tags"]):
                seen.add((k, " ".join(row["tokens"][s:e]).lower()))
    return seen


def unseen_recall(rows, pred, seen):
    """Recall only on values absent from training, which memorizing cannot help with."""
    caught = total = 0
    for row, found in zip(rows, pred):
        for s, e, k in spans(row["tags"]):
            if (k, " ".join(row["tokens"][s:e]).lower()) not in seen:
                total += 1
                caught += any(ps < e and s < pe for ps, pe, _ in found)
    return {"spans": total, "recall": round(100 * caught / total, 2) if total else None}


def brief(s):
    return {k: s[k] for k in ("any_type", "exact", "gold_spans", "missed_spans", "false_spans",
                              "sentences_without_pii", "false_alarm_sentences", "unseen_values")}


def main():
    test = load_csv(TEST_CSV)
    sets = {
        "clean": test,
        "no_punct": build_variant(test, "no_punct"),
        "asr_proxy": build_variant(test, "asr"),
        "handmade_seen": load_handmade(),
        "handmade_v2": load_handmade(HANDMADE_V2),
    }
    seen = train_values()
    threshold = json.loads((MODEL_DIR / "selection.json").read_text(encoding="utf-8"))["threshold"]
    report = {"threshold": threshold, "systems": {}}
    for system, path, t in (("base", BASE_MODEL, 0.5), ("oeg_pii", MODEL_DIR, threshold)):
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForTokenClassification.from_pretrained(path, use_safetensors=True).to("cuda").eval()
        for name, rows in sets.items():
            probs = predict_probs(model, tok, rows)
            pred = decode(probs, model.config.id2label, t)
            result = score(rows, pred)
            result["unseen_values"] = unseen_recall(rows, pred, seen)
            report["systems"].setdefault(system, {})[name] = result
            print(system, name, json.dumps(brief(result), ensure_ascii=False), flush=True)
            if system == "oeg_pii":
                ruled = [combine(p, rule_spans(r["tokens"])) for p, r in zip(pred, rows)]
                result = score(rows, ruled)
                result["unseen_values"] = unseen_recall(rows, ruled, seen)
                report["systems"].setdefault("oeg_pii_rules", {})[name] = result
                print("oeg_pii_rules", name, json.dumps(brief(result), ensure_ascii=False), flush=True)
        del model
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "final.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
