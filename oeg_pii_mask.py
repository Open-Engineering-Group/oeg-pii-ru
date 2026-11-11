"""OEG PII RU: find personal data in Russian text and replace it with reversible placeholders.

    from oeg_pii_mask import PiiMasker
    masker = PiiMasker("oegtech/oeg-pii-ru-lite")          # or a local folder
    masked, mapping = masker.mask("Меня зовут Ирина, телефон 8 916 555 12 34")
    # 'Меня зовут [ИМЯ_1], телефон [ТЕЛЕФОН_1]'
    masker.restore(masked, mapping)                    # the original text

The model labels whole words, so the text is split into words first. The mapping
holds the real values: keep it on your side and never send it with the masked text.
"""
import re

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

WORD = re.compile(r"@?\w+(?:[-'.@/:_]\w+)*/?|[^\w\s]")

PLACEHOLDER = {
    "FIRST_NAME": "ИМЯ", "LAST_NAME": "ФАМИЛИЯ", "MIDDLE_NAME": "ОТЧЕСТВО",
    "COUNTRY": "СТРАНА", "REGION": "РЕГИОН", "DISTRICT": "РАЙОН", "CITY": "ГОРОД", "STREET": "УЛИЦА", "HOUSE": "ДОМ",
    "EMAIL": "EMAIL", "PHONE": "ТЕЛЕФОН", "URL": "ССЫЛКА", "IP_ADDRESS": "IP",
    "PASSPORT": "ПАСПОРТ", "INN": "ИНН", "SNILS": "СНИЛС", "OMS": "ПОЛИС_ОМС", "CREDIT_CARD": "КАРТА",
    "DRIVER_LICENSE": "ВОДИТЕЛЬСКОЕ", "MILITARY_ID": "ВОЕННЫЙ_БИЛЕТ", "BIRTH_CERTIFICATE": "СВИДЕТЕЛЬСТВО_О_РОЖДЕНИИ",
    "DATE_OF_BIRTH": "ДАТА_РОЖДЕНИЯ", "BANK_ACCOUNT": "РАСЧЁТНЫЙ_СЧЁТ", "CORR_ACCOUNT": "КОРР_СЧЁТ",
    "BIK": "БИК", "KPP": "КПП", "TG_NICK": "TELEGRAM", "MAX_NICK": "MAX",
}


class PiiMasker:
    def __init__(self, model="oegtech/oeg-pii-ru-lite", device=None, threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model, use_safetensors=True).to(self.device).eval()
        self.id2label = self.model.config.id2label
        self.o_id = self.model.config.label2id["O"]
        self.threshold = threshold

    @torch.inference_mode()
    def _word_labels(self, words):
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=512, stride=128,
                       return_overflowing_tokens=True, padding=True, return_tensors="pt")
        probs = self.model(input_ids=enc["input_ids"].to(self.device),
                           attention_mask=enc["attention_mask"].to(self.device)).logits.float().softmax(-1)
        labels, seen = ["O"] * len(words), set()
        for chunk in range(probs.shape[0]):
            for pos, word in enumerate(enc.word_ids(chunk)):
                if word is None or word in seen:
                    continue
                seen.add(word)
                p = probs[chunk, pos].clone()
                if 1 - float(p[self.o_id]) >= self.threshold:
                    p[self.o_id] = -1
                    labels[word] = self.id2label[int(p.argmax())]
        return labels

    def find(self, text):
        """Return a list of (start, end, type, value) character spans."""
        matches = list(WORD.finditer(text))
        if not matches:
            return []
        labels = self._word_labels([m.group() for m in matches])
        found, current = [], None
        for m, label in zip(matches, labels):
            prefix, _, kind = label.partition("-")
            if prefix == "O":
                current = None
                continue
            if current and current[2] == kind and prefix == "I" or (current and current[2] == kind and prefix == "B" and text[current[1]:m.start()].strip() == ""):
                current[1] = m.end()
            else:
                current = [m.start(), m.end(), kind]
                found.append(current)
        return [(s, e, k, text[s:e]) for s, e, k in found]

    def mask(self, text):
        """Replace personal data with [TYPE_N]. The same value gets the same placeholder."""
        mapping, by_value, counters, out, at = {}, {}, {}, [], 0
        for s, e, kind, value in self.find(text):
            key = (kind, value)
            if key not in by_value:
                counters[kind] = counters.get(kind, 0) + 1
                tag = f"[{PLACEHOLDER.get(kind, kind)}_{counters[kind]}]"
                by_value[key] = tag
                mapping[tag] = value
            out.append(text[at:s])
            out.append(by_value[key])
            at = e
        out.append(text[at:])
        return "".join(out), mapping

    @staticmethod
    def restore(masked, mapping):
        for tag, value in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            masked = masked.replace(tag, value)
        return masked


if __name__ == "__main__":
    import sys

    masker = PiiMasker(sys.argv[1] if len(sys.argv) > 1 else "oegtech/oeg-pii-ru-lite")
    for line in sys.stdin:
        masked, mapping = masker.mask(line.rstrip("\n"))
        print(masked)
