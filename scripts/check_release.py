"""Check the release folder works on its own: load it, mask examples, restore them exactly.

Also checks that the raw-text path in oeg_pii_mask.py agrees with the word-level evaluation
on the fresh handmade test, so the numbers in the model card hold for real users.
"""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from oeg_pii_mask import PiiMasker  # noqa: E402
from oeg_pii_lib import HANDMADE_V2, load_handmade, score, spans  # noqa: E402

RELEASE = Path(sys.argv[1])
EXAMPLES = [
    "Здравствуйте, меня зовут Ирина Соколова, телефон 8 916 555 12 34, почта irina.s@example.ru",
    "алло да это олег петрович кузнецов номер восемь девятьсот шестнадцать пятьсот пятьдесят пять двенадцать тридцать четыре",
    "Реквизиты: р/с 40702810900000012345, БИК 044525225, ИНН 7707083893",
    "пишите в телеграм @lena_morozova88 или в max max.ru/olga_smirnova77",
    "дата рождения 14.02.1991, паспорт 4510 123456",
    "подскажите как отменить подписку, тариф стоит 450 рублей",
]


def main():
    torch.set_num_threads(4)
    masker = PiiMasker(str(RELEASE), device="cpu")
    out = {"examples": []}
    for text in EXAMPLES:
        started = time.perf_counter()
        masked, mapping = masker.mask(text)
        out["examples"].append({"text": text, "masked": masked, "mapping": mapping,
                                "restored_ok": PiiMasker.restore(masked, mapping) == text,
                                "ms": round(1000 * (time.perf_counter() - started))})
        print(masked, "| restored:", PiiMasker.restore(masked, mapping) == text, flush=True)

    # Raw-text path on the fresh test: join words with spaces, map character spans back to words.
    rows = load_handmade(HANDMADE_V2)
    preds = []
    for row in rows:
        text = " ".join(row["tokens"])
        starts, at = [], 0
        for w in row["tokens"]:
            starts.append((at, at + len(w)))
            at += len(w) + 1
        found = []
        for s, e, k, _ in masker.find(text):
            idx = [i for i, (ws, we) in enumerate(starts) if ws < e and s < we]
            if idx:
                found.append((idx[0], idx[-1] + 1, k))
        preds.append(found)
    result = score(rows, preds)
    out["raw_text_handmade_v2"] = {k: result[k] for k in ("any_type", "exact", "missed_spans", "false_spans")}
    print(json.dumps(out["raw_text_handmade_v2"], ensure_ascii=False), flush=True)
    (ROOT / "reports" / "release_check.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
