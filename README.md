# RAG Feodosia

Маршрутизация обращений граждан в Администрацию г. Феодосия — Telegram-бот, который определяет в какой из 18 поддепартаментов направить обращение.

На 70 реальных обращениях УГХ:
- `Accuracy на подразделении = 0.814`
- `Latency = 1.3 сек` на CPU без GPU

## Запуск

Нужен Docker и Docker Compose.

```bash
cp .env.example .env
# открыть .env и подставить TELEGRAM_BOT_TOKEN от @BotFather

docker compose up -d
docker compose logs -f ml
```

Первый старт ~5 минут — `ml` качает модели (E5-large 2.2 ГБ + bge-reranker 1.1 ГБ) в `hf_cache` volume. Потом — секунды.

Проверка вручную:

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Батареи холодные третий день"}'
```

## Структура

```
rag_feodosia/
├── bot/             Telegram-бот (aiogram). Принимает текст, дёргает ml, отвечает.
├── ml/              ML pipeline + FastAPI:
│   ├── api.py            POST /classify, GET /health
│   ├── pipeline.py       оркестратор route()
│   ├── retriever.py      beam search v2 (ANN L1 + L2)
│   ├── reranker.py       cross-encoder rerank + softmax
│   ├── confidence.py     3 порога (RAW_FLOOR / GAP_MIN / SCORE_HIGH)
│   ├── index.py          наполнение Qdrant из preambles/
│   ├── chunker.py        разбиение текста на чанки по 900 символов
│   ├── preambles_loader  чтение .md → карточки
│   ├── singletons.py     ленивые синглтоны Embedder/CE/Qdrant
│   ├── config.py         параметры через env
│   └── schemas.py        RouteResult dataclass
├── preambles/       Knowledge base — 22 .md файла:
│   ├── L1/               4 dept-карточки (УГХ, УГР, Образование, Культура)
│   └── L2/<dept>/        18 subdept-карточек (gas, water, heat, ...)
├── data/            Runtime — appeals.jsonl с логом обращений
├── docker-compose.yml    qdrant + ml + bot + 3 volume
├── Dockerfile.ml
├── Dockerfile.bot
└── .env.example     TELEGRAM_BOT_TOKEN
```

## Как это работает

Гражданин пишет боту: «Батареи холодные третий день».

**1. Embedding.** `intfloat/multilingual-e5-large` превращает текст в вектор из 1024 чисел. Карточки подразделений в `preambles/` тоже превращены в такие векторы и лежат в Qdrant.

**2. Beam search.** Сначала ищем ближайшие L1-карточки (4 управления) по cosine similarity. Отсекаем заведомо нерелевантные — те у кого сходство меньше 80% от лучшего. Потом ищем top-40 чанков L2 (поддепартаментов) только в оставшихся управлениях. Считаем для каждого `path_score = 0.3 × L1_cos + 0.7 × L2_cos`.

**3. Cross-encoder rerank.** Берём 10 лучших чанков (по одному на subdept). Каждый прогоняем через `bge-reranker-v2-m3` парой `(запрос, "{Управление}. {Поддепартамент}. {текст чанка}")`. Получаем 10 raw-score'ов. Префикс с названием отдела важен — даёт реранкеру контекст и снижает каскадные ошибки.

**4. MaxP-per-dept.** Для каждого из 4 управлений берём максимальный raw-score среди его поддепартаментов. Получаем 4 числа. Софтмакс по этим 4 — `score_dept` для confidence.

**5. Confidence — три порога:**

- `RAW_FLOOR` — если сырой CE-логит топ-кандидата меньше порога, значит ни одна карточка не похожа на запрос. Отдаём `unknown`. По умолчанию выключен (`-999`), включается опционально.
- `GAP_MIN` — если разница между топ-1 и топ-2 в softmax < 0.001, значит модель не различает темы (multi-issue запрос). Отдаём `unknown`.
- `SCORE_HIGH (0.347)` + `GAP_HIGH (0.004)` — если оба условия выполнены, это `high confidence` → автоматическая маршрутизация. Иначе `medium` → отметка для оператора.

Отдаём `RouteResult(department, subdepartment, confidence, ...)`.

## Конфигурация

Параметры через env-переменные в `docker-compose.yml`:

```
MAXP_PER_DEPT  = on        # MaxP-per-dept агрегация
CE_FINAL_LIMIT = 10        # сколько кандидатов в CE rerank
RAW_FLOOR      = -999.0    # off; 0.003 — мягкая OOS-защита, 0.005 — строгая
SCORE_HIGH     = 0.347
GAP_HIGH       = 0.004
GAP_MIN        = 0.001
```

## Управление

```bash
docker compose ps                  # статус
docker compose logs -f ml bot      # логи
docker compose down                # остановить
docker compose down -v             # + удалить volumes (модели придётся качать заново)

docker compose exec ml cat /app/data/appeals.jsonl   # посмотреть лог обращений
```

## Изменение карточек подразделений

```bash
# отредактировать preambles/L2/<dept>/<sub>.md
docker compose restart ml
```

При старте `ml` перестроит Qdrant-коллекцию из текущих `.md`. Никакого переобучения — pipeline сразу работает с новыми карточками.

Добавить новый отдел: создать новый файл `preambles/L2/<dept>/<new_sub>.md` с YAML frontmatter и содержанием в 4 секциях (определение / типичные ситуации / фразы граждан / ключевые слова) — и `docker compose restart ml`.
