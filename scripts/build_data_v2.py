"""Dataset v2. Fixes what the audit of v1 found:

1. Only 4% of rows had no PII, so the model learned that almost everything is PII.
   v2 adds real chat lines without PII, filtered hard, as a large share of rows.
2. Document numbers were few and repeated (placeholders such as 1234 567890).
   v2 replaces most of them with fresh values in many formats.
3. Only 5% of names were in Latin script. v2 raises that share.
4. New entities were placed almost only after a cue in formal sentences.
   v2 places every type inside real chat, sometimes with no cue at all.
5. Speech was too clean. v2 adds fillers, typos and dropped or merged words.

Values come from Faker (MIT) and generators. pii_benchmark and tests/ are not read.
The chat source has real people's messages, so rows with anything that looks like
personal data are dropped, and it is not redistributed.
"""
import json
import random
import re
import zlib
from pathlib import Path

from faker import Faker

import build_data as v1
from oeg_pii_lib import ROOT, TRAIN_CSV, build_variant, load_csv, spans

OUT = ROOT / "data_v2"
CHAT = ROOT / "assets" / "russian_dialogues" / "part.jsonl"
CHAT_LIMIT = 90_000
DOC_TYPES = {"PASSPORT", "MILITARY_ID", "BIRTH_CERTIFICATE", "DRIVER_LICENSE", "SNILS", "OMS", "INN", "CREDIT_CARD"}
WORD = re.compile(r"\w+(?:[-']\w+)*|[^\w\s]")
RISKY = re.compile(r"\d|@|https?:|www\.|\.\w{2,}/|зовут|живу|адрес|телефон|номер|паспорт|родил|фамили|мой ник|карт[аеуы]\b", re.I)
FILLERS = ["ну", "э", "эм", "вот", "как бы", "так", "да", "значит", "короче", "в общем", "сейчас"]
LATIN_LOCALES = ["en_US", "en_GB", "de_DE", "fr_FR", "es_ES", "it_IT", "pl_PL", "tr_TR"]

CUES = {
    "person": ["меня зовут", "это", "спросите", "клиент", "ФИО", "фамилия имя отчество", "звонит", "передайте"],
    "phone": ["телефон", "тел", "мой номер", "звоните на", "номер для связи", "моб", "перезвоните на"],
    "email": ["почта", "e-mail", "электронка", "пишите на", "мейл", "email:"],
    "url": ["сайт", "ссылка", "вот страница", "профиль"],
    "address": ["адрес", "живу", "доставка", "прописка", "проживаю", "везите на"],
    "passport": ["паспорт", "паспорт рф", "серия и номер", "паспортные данные", "пасп."],
    "inn": ["инн", "инн:", "мой инн", "инн организации"],
    "snils": ["снилс", "снилс:", "страховое свидетельство", "мой снилс"],
    "oms": ["омс", "полис", "полис омс", "номер полиса"],
    "card": ["карта", "номер карты", "на карту", "карточка"],
    "license": ["права", "в/у", "водительское", "удостоверение водителя"],
    "military": ["военный билет", "военник", "в/б"],
    "birth": ["свидетельство о рождении", "свид-во о рождении"],
    "ip": ["ip", "айпи", "адрес сервера"],
}
EXTRA_DOB_CUES = ["г.р.", "год рождения", "года рождения", "днюха", "появился на свет", "дата рожд.", ""]
NEGATIVE_NUMBERS = ["рублей", "руб", "штук", "процентов", "минут", "часов", "дней", "километров", "человек", "тысяч", "градусов", "лет", "раз"]


def fake_set(seed):
    fakes = {"ru": Faker("ru_RU")}
    for loc in LATIN_LOCALES:
        fakes[loc] = Faker(loc)
    for i, f in enumerate(fakes.values()):
        f.seed_instance(seed + i)
    return fakes


def valid_snils(rng):
    body = "".join(rng.choice("0123456789") for _ in range(9))
    total = sum(int(c) * (9 - i) for i, c in enumerate(body))
    while total > 101:
        total %= 101
    control = 0 if total in (100, 101) else total
    return body, f"{control:02d}"


def doc_value(kind, rng, fakes):
    d = lambda n: v1.digits(rng, n)
    if kind == "PASSPORT":
        s, n = d(4), d(6)
        return rng.choice([[s, n], [s[:2], s[2:], n], [s + n], [f"{s[:2]} {s[2:]}".replace(" ", "-"), n], ["серия", s, "номер", n], [f"{s}-{n}"], [f"№{n}"]])
    if kind == "SNILS":
        b, c = valid_snils(rng)
        return rng.choice([[f"{b[:3]}-{b[3:6]}-{b[6:]} {c}".split()[0], c], [b + c], [b[:3], b[3:6], b[6:], c], [f"{b[:3]}-{b[3:6]}-{b[6:]}-{c}"]])
    if kind == "INN":
        return [rng.choice([fakes["ru"].businesses_inn(), fakes["ru"].individuals_inn()])]
    if kind == "OMS":
        v = d(16)
        return rng.choice([[v], [v[:4], v[4:8], v[8:12], v[12:]], [v[:6], v[6:]]])
    if kind == "CREDIT_CARD":
        v = fakes["ru"].credit_card_number()
        v = v if len(v) >= 16 else v + d(16 - len(v))
        return rng.choice([[v], [v[:4], v[4:8], v[8:12], v[12:]], ["-".join([v[:4], v[4:8], v[8:12], v[12:16]])]])
    if kind == "DRIVER_LICENSE":
        s, n = d(4), d(6)
        return rng.choice([[s, n], [s[:2], s[2:], n], [s + n]])
    if kind == "MILITARY_ID":
        letters = "".join(rng.choice("АБВГДЕЖИКЛМНПРСТ") for _ in range(2))
        n = d(7)
        return rng.choice([[letters, n], [letters + n], [letters.lower(), n], ["серия", letters, "номер", n]])
    letters = rng.choice(["I", "II", "III", "IV", "V", "VI", "X", "XI"]) + "-" + "".join(rng.choice("АБВГДЕЖИКЛМН") for _ in range(2))
    n = d(6)
    return rng.choice([[letters, n], [letters + "№" + n], ["серия", letters, "номер", n]])


def name_parts(rng, fakes):
    if rng.random() < 0.3:
        f = fakes[rng.choice(LATIN_LOCALES)]
        first, last = f.first_name(), f.last_name()
        return [("FIRST_NAME", first), ("LAST_NAME", last)]
    f = fakes["ru"]
    male = rng.random() < 0.5
    first = f.first_name_male() if male else f.first_name_female()
    last = f.last_name_male() if male else f.last_name_female()
    middle = f.middle_name_male() if male else f.middle_name_female()
    style = rng.randint(0, 4)
    if style == 0:
        return [("FIRST_NAME", first)]
    if style == 1:
        return [("LAST_NAME", last), ("FIRST_NAME", first), ("MIDDLE_NAME", middle)]
    if style == 2:
        return [("FIRST_NAME", first), ("MIDDLE_NAME", middle)]
    if style == 3:
        return [("FIRST_NAME", first), ("LAST_NAME", last)]
    return [("LAST_NAME", last)]


def tagged(pairs):
    tokens, tags = [], []
    for label, value in pairs:
        words = value if isinstance(value, list) else value.split()
        for i, w in enumerate(words):
            tokens.append(w)
            tags.append(("B-" if i == 0 else "I-") + label)
    return tokens, tags


def dob_value(rng):
    roll = rng.random()
    y = rng.randint(1945, 2015)
    if roll < 0.12:
        return [str(y)]
    if roll < 0.18:
        return [f"{y}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"]
    return v1.make_dob(rng)


def fragment(rng, fakes, names):
    """A cue plus a value of any type. The cue is dropped at times so context cannot be everything."""
    kind = rng.choice(["person", "person", "phone", "phone", "email", "url", "address", "passport", "inn", "snils", "oms",
                       "card", "license", "military", "birth", "ip", "dob", "dob", "tg", "tg", "max", "max", "bank", "bank"])
    if kind in ("tg", "max"):
        return v1.fragment(kind, rng, names)
    if kind == "bank":
        return v1.fragment(rng.choice(["account", "corr", "bik", "kpp"]), rng, names)
    f = fakes["ru"]
    if kind == "dob":
        cue, pairs = rng.choice(v1.CUES["dob"] + EXTRA_DOB_CUES), [("DATE_OF_BIRTH", dob_value(rng))]
    elif kind == "person":
        cue, pairs = rng.choice(CUES["person"]), name_parts(rng, fakes)
    elif kind == "phone":
        number = re.sub(r"\D", "", f.phone_number())[-10:].rjust(10, "9")
        style = rng.randint(0, 3)
        value = [f"+7{number}", f"8{number}", f"+7 ({number[:3]}) {number[3:6]}-{number[6:8]}-{number[8:]}",
                 f"8 {number[:3]} {number[3:6]} {number[6:8]} {number[8:]}"][style]
        cue, pairs = rng.choice(CUES["phone"]), [("PHONE", value)]
    elif kind == "email":
        cue, pairs = rng.choice(CUES["email"]), [("EMAIL", f.email())]
    elif kind == "url":
        cue, pairs = rng.choice(CUES["url"]), [("URL", f.url())]
    elif kind == "ip":
        cue, pairs = rng.choice(CUES["ip"]), [("IP_ADDRESS", f.ipv4())]
    elif kind == "address":
        cue = rng.choice(CUES["address"])
        pairs = [("CITY", f.city_name()), ("STREET", rng.choice(["ул. ", "улица ", "проспект ", ""]) + f.street_name()),
                 ("HOUSE", rng.choice(["д. ", "дом ", ""]) + str(rng.randint(1, 180)))]
        if rng.random() < 0.3:
            pairs = pairs[: rng.randint(1, 2)]
    else:
        label = {"passport": "PASSPORT", "inn": "INN", "snils": "SNILS", "oms": "OMS", "card": "CREDIT_CARD",
                 "license": "DRIVER_LICENSE", "military": "MILITARY_ID", "birth": "BIRTH_CERTIFICATE"}[kind]
        cue, pairs = rng.choice(CUES[kind]), [(label, doc_value(label, rng, fakes))]
    cue_tokens = [] if rng.random() < 0.15 else cue.split()
    tokens, tags = tagged(pairs)
    return cue_tokens + tokens, ["O"] * len(cue_tokens) + tags


def reshape(word, rng):
    """Swap each digit for a digit and each letter for a letter of the same script and case."""
    out = []
    for c in word:
        if c.isdigit():
            out.append(rng.choice("0123456789"))
        elif c.isalpha():
            pool = "АБВГДЕЖИКЛМНПРСТ" if "А" <= c.upper() <= "Я" else "IVXLABCEHKMT"
            pick = rng.choice(pool)
            out.append(pick if c.isupper() else pick.lower())
        else:
            out.append(c)
    return "".join(out)


def swap_values(row, rng, fakes):
    """Replace document numbers with fresh values, and turn some names into Latin ones."""
    tokens, tags, at = [], [], 0
    for s, e, k in spans(row["tags"]):
        tokens += row["tokens"][at:s]
        tags += row["tags"][at:s]
        words = row["tokens"][s:e]
        if k in DOC_TYPES and rng.random() < 0.9:
            # Keep the shape: a span may hold only the series or only the number,
            # so a whole new document here would duplicate what the sentence already says.
            words = [reshape(w, rng) for w in words]
        elif k in ("FIRST_NAME", "LAST_NAME") and rng.random() < 0.2:
            f = fakes[rng.choice(LATIN_LOCALES)]
            words = (f.first_name() if k == "FIRST_NAME" else f.last_name()).split()
        tokens += words
        tags += [("B-" if i == 0 else "I-") + k for i in range(len(words))]
        at = e
    tokens += row["tokens"][at:]
    tags += row["tags"][at:]
    return {"tokens": tokens, "tags": tags}


def negative_number(rng):
    """A short number that is not PII, next to an ordinary unit. Digits become words in the speech variants."""
    tokens = [str(rng.randint(1, 99999)), rng.choice(NEGATIVE_NUMBERS)]
    return tokens, ["O"] * len(tokens)


def insert(row, pieces, rng):
    tokens, tags = list(row["tokens"]), list(row["tags"])
    for new_tokens, new_tags in pieces:
        slots = [i for i in range(len(tokens) + 1) if i == len(tokens) or not tags[i].startswith("I-")]
        # Mostly at the edges or after punctuation, where a person would really say it.
        breaks = [i for i in slots if i in (0, len(tokens)) or tokens[i - 1] in {".", ",", "!", "?", ":", ";"}]
        at = rng.choice(breaks) if rng.random() < 0.75 else rng.choice(slots)
        tokens[at:at] = new_tokens
        tags[at:at] = new_tags
    return {"tokens": tokens, "tags": tags}


def speech_noise(row, rng):
    """Mimic an imperfect transcript: fillers, dropped short words, merged and mistyped words."""
    tokens, tags = [], []
    for token, tag in zip(row["tokens"], row["tags"]):
        roll = rng.random()
        if tag == "O" and len(token) <= 2 and roll < 0.05:
            continue
        if roll > 0.96 and tokens:
            for filler in rng.choice(FILLERS).split():
                tokens.append(filler)
                inside = tag.startswith("I-")
                tags.append("I-" + tag[2:] if inside else "O")
        if tag == "O" and len(token) > 4 and rng.random() < 0.02:
            i = rng.randrange(1, len(token) - 1)
            token = token[:i] + token[i + 1:]
        tokens.append(token.replace("ё", "е") if rng.random() < 0.5 else token)
        tags.append(tag)
    return {"tokens": tokens, "tags": tags}


def chat_typos(row, rng):
    out = []
    for token, tag in zip(row["tokens"], row["tags"]):
        if tag == "O" and len(token) > 4 and rng.random() < 0.03:
            i = rng.randrange(1, len(token) - 1)
            token = token[:i] + token[i + 1] + token[i] + token[i + 2:]
        out.append(token)
    return {"tokens": out, "tags": row["tags"]}


def load_chat(blocklist):
    """Keep short lines with no digits, links, names or personal-data talk."""
    seen, rows, total = set(), [], 0
    with open(CHAT, encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            for text in (item.get("question", ""), item.get("answer", "")):
                total += 1
                text = str(text).strip()
                if not 8 <= len(text) <= 220 or RISKY.search(text) or text.lower() in seen:
                    continue
                tokens = WORD.findall(text)
                if not 3 <= len(tokens) <= 40:
                    continue
                words = [t.lower() for t in tokens]
                if any(w in blocklist or w[:-1] in blocklist or w[:-2] in blocklist for w in words if len(w) >= 3):
                    continue
                # A capital letter inside a line usually marks a name or a place.
                if any(t[0].isupper() for t in tokens[1:] if t[0].isalpha()):
                    continue
                seen.add(text.lower())
                rows.append({"tokens": tokens, "tags": ["O"] * len(tokens)})
            if len(rows) >= CHAT_LIMIT * 2:
                break
    return rows, total


def build_blocklist(train, fakes):
    names = set()
    for row in train:
        for s, e, k in spans(row["tags"]):
            if k in ("FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "CITY", "REGION", "COUNTRY", "STREET"):
                for t in row["tokens"][s:e]:
                    if len(t) >= 3 and t.isalpha():
                        names.add(t.lower())
    f = fakes["ru"]
    for _ in range(6000):
        for v in (f.first_name_male(), f.first_name_female(), f.last_name_male(), f.last_name_female(), f.city_name()):
            names.add(v.lower())
    return names


def clean_with_model(chat):
    """Drop chat lines where the base model sees anything. Names left in as O would teach misses."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from oeg_pii_lib import BASE_MODEL, decode, predict_probs

    cache = CHAT.with_name("filtered.jsonl")
    if cache.exists():
        with open(cache, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    torch.set_num_threads(6)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL, use_safetensors=True).eval()
    kept = []
    for at in range(0, len(chat), 4000):
        part = chat[at : at + 4000]
        # A low threshold: losing a clean line costs nothing, keeping a name costs recall.
        found = decode(predict_probs(model, tok, part, device="cpu", batch_size=32), model.config.id2label, 0.2)
        kept += [r for r, f in zip(part, found) if not f]
        print(f"filter {at + len(part)}/{len(chat)} kept {len(kept)}", flush=True)
    with open(cache, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return kept


def fix_starts(tags):
    """A span whose first word was dropped must still open with B-."""
    out, prev = [], "O"
    for tag in tags:
        if tag.startswith("I-") and prev.partition("-")[2] != tag[2:]:
            tag = "B-" + tag[2:]
        out.append(tag)
        prev = tag
    return out


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if row["tokens"]:
                row = {"tokens": row["tokens"], "tags": fix_starts(row["tags"])}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compose(hosts, chat, seed, names, fakes, scale=1):
    rng = random.Random(seed)
    swapped = lambda: [swap_values(r, rng, fakes) for r in hosts]
    add = lambda rows: [v1.add_fragments(r, rng, names) for r in rows]
    rows = {"source_clean": swapped()}
    rows["source_fragments"] = add(swapped())
    rows["source_no_punct"] = [chat_typos(r, rng) for r in build_variant(add(swapped()), "no_punct", seed + 1)]
    rows["source_asr"] = [speech_noise(r, rng) for r in build_variant(add(swapped()), "asr", seed + 2, translit_prob=0.5)]
    rows["source_asr2"] = [speech_noise(r, rng) for r in build_variant(add(swapped()), "asr", seed + 3, translit_prob=0.5)]

    rng.shuffle(chat)
    n = len(chat)
    neg, pos = chat[: int(n * 0.6)], chat[int(n * 0.6) :]
    # A third of the clean lines get a harmless number, so digits alone do not signal PII.
    neg = [insert(r, [negative_number(rng)], rng) if rng.random() < 0.33 else r for r in neg]
    k = len(neg)
    rows["chat_negative"] = neg[: k // 3]
    rows["chat_negative_no_punct"] = [chat_typos(r, rng) for r in build_variant(neg[k // 3 : 2 * k // 3], "no_punct", seed + 4)]
    rows["chat_negative_asr"] = [speech_noise(r, rng) for r in build_variant(neg[2 * k // 3 :], "asr", seed + 5)]

    def with_pii(row):
        count = rng.choice([1, 1, 1, 2, 3])
        return insert(row, [fragment(rng, fakes, names) for _ in range(count)], rng)

    m = len(pos)
    rows["chat_pii"] = [with_pii(r) for r in pos[: m // 3]]
    rows["chat_pii_no_punct"] = [chat_typos(r, rng) for r in build_variant([with_pii(r) for r in pos[m // 3 : 2 * m // 3]], "no_punct", seed + 6)]
    rows["chat_pii_asr"] = [speech_noise(r, rng) for r in build_variant([with_pii(r) for r in pos[2 * m // 3 :]], "asr", seed + 7, translit_prob=0.5)]
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    source = load_csv(TRAIN_CSV)
    train, dev = [], []
    for row in source:
        (dev if zlib.crc32(row["text"].encode()) % 20 == 0 else train).append(row)
    names = v1.harvest_names(train)
    fakes = fake_set(1234)
    blocklist = build_blocklist(train, fakes)
    chat, total = load_chat(blocklist)
    print(f"chat lines read={total} kept={len(chat)}", flush=True)
    random.Random(5).shuffle(chat)
    chat = clean_with_model(chat[: CHAT_LIMIT + CHAT_LIMIT // 5])
    print(f"chat lines after model filter={len(chat)}", flush=True)
    cut = int(len(chat) * 0.95)
    chat_train, chat_dev = chat[:cut][:CHAT_LIMIT], chat[cut:]

    parts = compose(train, chat_train, 200, names, fakes)
    stats = {k: len(v) for k, v in parts.items()}
    all_rows = [c for rows in parts.values() for r in rows for c in v1.chunk(r)]
    random.Random(7).shuffle(all_rows)
    save(OUT / "train.jsonl", all_rows)

    dev_fakes = fake_set(98765)
    dev_parts = compose(dev, chat_dev, 900, names, dev_fakes)
    groups = {
        "clean": ["source_fragments", "chat_negative", "chat_pii"],
        "no_punct": ["source_no_punct", "chat_negative_no_punct", "chat_pii_no_punct"],
        "asr": ["source_asr", "chat_negative_asr", "chat_pii_asr"],
    }
    dev_sizes = {}
    for name, keys in groups.items():
        rows = [c for k in keys for r in dev_parts[k] for c in v1.chunk(r)]
        save(OUT / f"dev_{name}.jsonl", rows)
        dev_sizes[name] = len(rows)
    report = {
        "chat_lines_read": total,
        "chat_lines_kept": len(chat),
        "parts": stats,
        "train_rows": len(all_rows),
        "dev_rows": dev_sizes,
        "train_rows_without_pii": sum(not spans(r["tags"]) for r in all_rows),
    }
    (ROOT / "reports" / "data_v2_build.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    rng = random.Random(1)
    for key in ["chat_negative", "chat_negative_asr", "chat_pii", "chat_pii_asr", "source_asr", "source_no_punct"]:
        for row in rng.sample(parts[key], 4):
            marked = []
            for tok, tag in zip(row["tokens"], row["tags"]):
                marked.append(tok if tag == "O" else f"[{tok}|{tag}]")
            print(key, "::", " ".join(marked)[:220], flush=True)


if __name__ == "__main__":
    main()
