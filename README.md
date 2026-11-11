# OEG PII RU

[English](#english) · [Русский](#русский)

<a id="english"></a>

## English

Training and evaluation code for OEG models that remove personal data from Russian text.
Current release: [oegtech/oeg-pii-ru-lite](https://huggingface.co/oegtech/oeg-pii-ru-lite) 0.1.0 (preview).
It finds personal data in Russian call transcripts and chats and replaces it with reversible
placeholders such as `[ТЕЛЕФОН_1]`.

Weights, the model description, results and limitations are in the Hugging Face model card.

### Usage

```python
from oeg_pii_mask import PiiMasker

masker = PiiMasker("oegtech/oeg-pii-ru-lite")
masked, mapping = masker.mask("Меня зовут Ирина, телефон 8 916 555 12 34")
original = masker.restore(masked, mapping)
```

### Contents

| File | Purpose |
|---|---|
| `oeg_pii_mask.py` | Find, replace and restore personal data |
| `scripts/fetch_assets.py` | Download the base model and datasets, pinned by commit |
| `scripts/evaluate_baseline.py` | Check the base model and the call simulation |
| `scripts/build_data.py`, `scripts/build_data_v2.py` | Build the training data |
| `scripts/audit_data.py`, `scripts/check_chat_noise.py` | Audit the data |
| `scripts/train.py` | Training (`--throttle=1.0` for a cooler, slower run) |
| `scripts/evaluate.py` | Final evaluation |
| `scripts/oeg_pii_lib.py` | Shared code: speech simulation, metrics, rules |
| `tests/` | Handmade tests (fictional values) |
| `reports/` | Run reports |

### Reproduce

Requires Python 3.12 or 3.13, `torch`, `transformers`, `huggingface_hub`, `Faker`.
A fragment of `Den4ikAI/russian_dialogues` is downloaded separately to
`assets/russian_dialogues/part.jsonl` (the first 200 MB of `dataset.jsonl`; the commit is recorded
in the reports). It contains real Telegram messages and is not redistributed here.

```bash
python scripts/fetch_assets.py
python scripts/build_data_v2.py
python scripts/train.py --throttle=1.0
python scripts/evaluate.py
```

### License

Apache 2.0. The model is based on `redmadrobot-rnd/rubert-base-pii-ner` (Apache 2.0).

<a id="русский"></a>

## Русский

Код обучения и оценки моделей OEG для обезличивания персональных данных. Текущий выпуск:
[oegtech/oeg-pii-ru-lite](https://huggingface.co/oegtech/oeg-pii-ru-lite) 0.1.0 (preview).
Модель находит персональные данные в русских расшифровках звонков и переписках и заменяет их
метками вида `[ТЕЛЕФОН_1]` с возможностью обратного восстановления.

Веса, описание модели, результаты и ограничения — в карточке на Hugging Face.

### Использование

```python
from oeg_pii_mask import PiiMasker

masker = PiiMasker("oegtech/oeg-pii-ru-lite")
masked, mapping = masker.mask("Меня зовут Ирина, телефон 8 916 555 12 34")
original = masker.restore(masked, mapping)
```

### Что здесь лежит

| Файл | Назначение |
|---|---|
| `oeg_pii_mask.py` | Поиск, замена и восстановление персональных данных |
| `scripts/fetch_assets.py` | Скачивание базовой модели и наборов с фиксацией версий |
| `scripts/evaluate_baseline.py` | Проверка базовой модели и имитации звонков |
| `scripts/build_data.py`, `scripts/build_data_v2.py` | Сборка обучающих данных |
| `scripts/audit_data.py`, `scripts/check_chat_noise.py` | Аудит данных |
| `scripts/train.py` | Обучение (`--throttle=1.0` для щадящего режима) |
| `scripts/evaluate.py` | Финальная оценка |
| `scripts/oeg_pii_lib.py` | Общий код: имитация речи, метрики, правила |
| `tests/` | Ручные тесты (вымышленные значения) |
| `reports/` | Отчёты о прогонах |

### Воспроизведение

Нужны Python 3.12 или 3.13, `torch`, `transformers`, `huggingface_hub`, `Faker`.
Фрагмент `Den4ikAI/russian_dialogues` скачивается отдельно в `assets/russian_dialogues/part.jsonl`
(первые 200 МБ файла `dataset.jsonl`, коммит указан в отчётах). В нём настоящие сообщения из
Telegram, поэтому здесь он не распространяется.

```bash
python scripts/fetch_assets.py
python scripts/build_data_v2.py
python scripts/train.py --throttle=1.0
python scripts/evaluate.py
```

### Лицензия

Apache 2.0. Модель основана на `redmadrobot-rnd/rubert-base-pii-ner` (Apache 2.0).
