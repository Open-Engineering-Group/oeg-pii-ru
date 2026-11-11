"""Build the training and dev sets.

Hosts are pii_train sentences (MIT). New entities are programmatically generated,
fictional values placed next to short cues and inserted at entity boundaries.
pii_benchmark is never read here: it stays for the final measurement.
"""
import json
import random
import re
import zlib

from oeg_pii_lib import DATA_DIR, TRAIN_CSV, build_variant, load_csv, spans

MONTHS_GEN = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
ORD_GEN = ("первого второго третьего четвёртого пятого шестого седьмого восьмого девятого десятого "
           "одиннадцатого двенадцатого тринадцатого четырнадцатого пятнадцатого шестнадцатого семнадцатого "
           "восемнадцатого девятнадцатого двадцатого").split()
ORD_YEAR = {0: "", 1: "первого", 2: "второго", 3: "третьего", 4: "четвёртого", 5: "пятого", 6: "шестого",
            7: "седьмого", 8: "восьмого", 9: "девятого"}
TEN_WORD = {5: "пятьдесят", 6: "шестьдесят", 7: "семьдесят", 8: "восемьдесят", 9: "девяносто"}
TEN_ORD = {5: "пятидесятого", 6: "шестидесятого", 7: "семидесятого", 8: "восьмидесятого", 9: "девяностого"}

CUES = {
    "dob": ["дата рождения", "дата рождения:", "д.р.", "др", "родился", "родилась", "год и дата рождения",
            "родился я", "у меня день рождения", "рожд."],
    "tg": ["телеграм", "телеграм:", "тг", "tg:", "telegram:", "ник в телеге", "пишите в телеграм", "мой телеграм",
           "в телеге я", "аккаунт телеграм", "ищите в тг"],
    "max": ["макс:", "max:", "в максе", "ник в максе", "мой макс", "пишите в макс", "мессенджер max",
            "аккаунт в max", "ищите в max", "профиль max"],
    "account": ["р/с", "р/с:", "расчётный счёт", "расчетный счет", "счёт получателя", "номер счёта", "рс", "лицевой счёт"],
    "corr": ["к/с", "к/с:", "корреспондентский счёт", "корр счёт", "кс", "корр. счет"],
    "bik": ["бик", "бик:", "бик банка", "банковский идентификационный код"],
    "kpp": ["кпп", "кпп:", "код причины постановки"],
}
NEGATIVES = [
    ["договор", "от", "{date}"], ["встреча", "назначена", "на", "{date}"], ["срок", "оплаты", "до", "{date}"],
    ["отчёт", "за", "{date}", "готов"], ["заказ", "номер", "{short}", "отправлен"], ["артикул", "{short}"],
    ["сумма", "{short}", "рублей"], ["макс", "скорость", "сто", "двадцать"], ["max", "значение", "не", "задано"],
    ["телеграм", "сегодня", "не", "работает"], ["у", "меня", "собака", "и", "кот"], ["подчёркивание", "в", "тексте", "лишнее"],
    ["день", "рождения", "компании", "отмечаем", "в", "пятницу"], ["позвоните", "в", "девять", "утра"],
    ["примерно", "сто", "двадцать", "три", "человека"], ["банк", "работает", "до", "восьми"], ["счёт", "уже", "оплачен"],
    ["тг", "канал", "закрыт"], ["в", "максе", "пока", "тихо"], ["кпп", "на", "въезде", "закрыт"],
]
WORDS = ("alex nika dev cat rock sun wolf fox star neo max_fan ghost coder pixel lion mira art home city green "
         "blue red best real pro true").split()
TR = str.maketrans({"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
                    "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
                    "у": "u", "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y",
                    "ь": "", "э": "e", "ю": "yu", "я": "ya"})


def harvest_names(rows):
    names = set()
    for row in rows:
        for s, e, k in spans(row["tags"]):
            if k in ("FIRST_NAME", "LAST_NAME") and e - s == 1:
                latin = re.sub(r"[^a-z]", "", row["tokens"][s].lower().translate(TR))
                if len(latin) >= 3:
                    names.add(latin)
    return sorted(names)


def make_nick(rng, names):
    a, b = rng.choice(names), rng.choice(names)
    w = rng.choice(WORDS)
    n = str(rng.randint(1, 9999))
    nick = rng.choice([a, f"{a}_{b}", f"{a}{b}", f"{a}{n}", f"{a}_{n}", f"{w}_{a}", f"{a}_{w}", f"{w}{n}", f"the_{a}", f"{a[0]}_{b}"])
    if rng.random() < 0.2:
        nick = nick.capitalize()
    nick = nick[:32]
    return nick if len(nick) >= 5 else nick + n


def digits(rng, n, prefix=""):
    return prefix + "".join(rng.choice("0123456789") for _ in range(n - len(prefix)))


def group(value, rng, sizes):
    if rng.random() < 0.65:
        return [value]
    out, at = [], 0
    while at < len(value):
        size = rng.choice(sizes)
        out.append(value[at : at + size])
        at += size
    return out


def spoken_year(y):
    if y >= 2000:
        rest = y - 2000
        if rest == 0:
            return ["двухтысячного"]
        return ["две", "тысячи"] + ([ORD_GEN[rest - 1]] if rest <= 20 else ["двадцать", ORD_YEAR[rest % 10]])
    t, u = (y % 100) // 10, y % 10
    base = ["тысяча", "девятьсот"]
    if u == 0:
        return base + [TEN_ORD[t]]
    return base + [TEN_WORD[t], ORD_YEAR[u]]


def make_dob(rng):
    d, m, y = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1950, 2012)
    style = rng.randint(0, 6)
    if style == 0:
        return [f"{d:02d}.{m:02d}.{y}"]
    if style == 1:
        return [f"{d:02d}/{m:02d}/{y}"]
    if style == 2:
        return [f"{d:02d}.{m:02d}.{y % 100:02d}"]
    if style == 3:
        return [str(d), MONTHS_GEN[m - 1], str(y)]
    if style == 4:
        return [str(d), MONTHS_GEN[m - 1], str(y), "г."]
    if style == 5:
        return [f"{y}-{m:02d}-{d:02d}"]
    day = ORD_GEN[d - 1] if d <= 20 else ("двадцать " + ORD_YEAR[d - 20]).split()
    day = day if isinstance(day, list) else [day]
    tail = ["года"] if rng.random() < 0.7 else []
    return day + [MONTHS_GEN[m - 1]] + (spoken_year(y) if rng.random() < 0.7 else [TEN_ORD[(y % 100) // 10]] if y % 10 == 0 and y < 2000 else spoken_year(y)[-2:]) + tail


def fragment(kind, rng, names):
    """Return token list and tag list for a cue followed by a value."""
    label = {"dob": "DATE_OF_BIRTH", "tg": "TG_NICK", "max": "MAX_NICK", "account": "BANK_ACCOUNT",
             "corr": "CORR_ACCOUNT", "bik": "BIK", "kpp": "KPP"}[kind]
    cue = rng.choice(CUES[kind]).split()
    if kind == "dob":
        value = make_dob(rng)
    elif kind == "tg":
        nick = make_nick(rng, names)
        value = [rng.choice([f"@{nick}", f"@{nick}", f"t.me/{nick}", f"https://t.me/{nick}", nick])]
    elif kind == "max":
        nick = make_nick(rng, names)
        value = [rng.choice([f"@{nick}", f"max.ru/{nick}", f"https://max.ru/{nick}", nick, nick])]
    elif kind == "account":
        value = group(digits(rng, 20, rng.choice(["40702", "40817", "40802", "42301"])), rng, [4, 5])
    elif kind == "corr":
        value = group(digits(rng, 20, "30101810"), rng, [4, 5])
    elif kind == "bik":
        value = group(digits(rng, 9, "04"), rng, [2, 3])
    else:
        value = [digits(rng, 9)]
    if rng.random() < 0.12:
        cue = []
    tags = ["O"] * len(cue) + [("B-" if i == 0 else "I-") + label for i in range(len(value))]
    return cue + value, tags


def negative(rng):
    template = rng.choice(NEGATIVES)
    tokens = []
    for t in template:
        if t == "{date}":
            tokens += [f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2015, 2027)}"]
        elif t == "{short}":
            tokens += [digits(rng, rng.randint(2, 6))]
        else:
            tokens.append(t)
    return tokens, ["O"] * len(tokens)


def add_fragments(row, rng, names):
    tokens, tags = list(row["tokens"]), list(row["tags"])
    roll = rng.random()
    if roll < 0.15:
        pieces = [negative(rng)]
    elif roll < 0.25:
        pieces = [fragment(k, rng, names) for k in ("account", "bik", "corr", "kpp") if rng.random() < 0.8]
    elif roll < 0.75:
        pieces = [fragment(rng.choice(["dob", "tg", "tg", "max", "max", "account", "corr", "bik", "kpp"]), rng, names)]
    else:
        return {"tokens": tokens, "tags": tags}
    for new_tokens, new_tags in pieces:
        slots = [i for i in range(len(tokens) + 1) if i == len(tokens) or not tags[i].startswith("I-")]
        at = rng.choice(slots) if rng.random() < 0.4 else len(tokens)
        tokens[at:at] = new_tokens
        tags[at:at] = new_tags
    return {"tokens": tokens, "tags": tags}


def chunk(row, limit=100):
    """Split long rows at entity boundaries so training never truncates a label."""
    if len(row["tokens"]) <= limit:
        return [row]
    out, at = [], 0
    n = len(row["tokens"])
    while at < n:
        end = min(at + limit, n)
        while end < n and row["tags"][end].startswith("I-") and end > at + 1:
            end -= 1
        out.append({"tokens": row["tokens"][at:end], "tags": row["tags"][at:end]})
        at = end
    return out


def save(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if row["tokens"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    rows = load_csv(TRAIN_CSV)
    train, dev = [], []
    for row in rows:
        (dev if zlib.crc32(row["text"].encode()) % 20 == 0 else train).append(row)
    names = harvest_names(train)
    print(f"hosts train={len(train)} dev={len(dev)} names={len(names)}", flush=True)

    def expand(hosts, seed):
        rng = random.Random(seed)
        out = [{"tokens": r["tokens"], "tags": r["tags"]} for r in hosts]
        written = [add_fragments(r, rng, names) for r in hosts]
        out += written
        out += build_variant([add_fragments(r, rng, names) for r in hosts], "no_punct", seed + 1)
        out += build_variant([add_fragments(r, rng, names) for r in hosts], "asr", seed + 2, translit_prob=0.5)
        out += build_variant([add_fragments(r, rng, names) for r in hosts], "asr", seed + 3, translit_prob=0.5)
        return [c for r in out for c in chunk(r)]

    train_rows = expand(train, 100)
    random.Random(7).shuffle(train_rows)
    save(DATA_DIR / "train.jsonl", train_rows)
    rng = random.Random(999)
    dev_sets = {
        "clean": [add_fragments(r, rng, names) for r in dev],
        "no_punct": build_variant([add_fragments(r, rng, names) for r in dev], "no_punct", 1000),
        "asr": build_variant([add_fragments(r, rng, names) for r in dev], "asr", 1001, translit_prob=0.5),
    }
    for name, data in dev_sets.items():
        save(DATA_DIR / f"dev_{name}.jsonl", [c for r in data for c in chunk(r)])
    print(f"train rows={len(train_rows)} dev={ {k: len(v) for k, v in dev_sets.items()} }", flush=True)
    for row in train_rows[:6]:
        print(" ".join(row["tokens"])[:150], flush=True)


if __name__ == "__main__":
    main()
