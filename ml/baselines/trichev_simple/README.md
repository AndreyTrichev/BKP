# Baseline: trichev_simple

Vendored из [AndreyTrichev/rag](https://github.com/AndreyTrichev/rag) (commit `3b32407`, 2026-03-22) — ранний прототип того же стека (E5 + Qdrant + bge-reranker), который автор делал до экспериментов REPORT2/3.

**Назначение:** точка отсчёта для дипломной работы. Показывает прирост метрик за счёт архитектурных улучшений (beam search v2, adaptive prune, path scoring, context injection, MaxP-per-dept со всеми 4 dept, 3-слойная confidence).

---

## Структура (симметрична продвинутому `ml/`)

```
ml/baselines/trichev_simple/
├── config.py        — параметры (env vars + пороги, изолированный COLLECTION_NAME)
├── singletons.py    — lazy embedder/CE/Qdrant (свои инстансы, не пересекаются с ml/)
├── retriever.py     — encode_query + ann_l2 + query_l1 + filter_l2_by_dept
├── reranker.py      — softmax + compact_text + select_l2_for_ce + ce_rerank
├── confidence.py    — determine_confidence (2 порога: SCORE_HIGH / SCORE_MEDIUM)
├── index.py         — create_collection (наполняет citizen_complaints_baseline)
├── pipeline.py      — route() + get_routing_context() + MaxP-агрегация
├── knowledge_base.py — vendored, 16-subdept схема preambles
└── documents/       — 4 .txt из baseline-репо (источники для KB)
```

`route(text)` возвращает `ml.schemas.RouteResult` напрямую — совместим с продвинутым pipeline'ом, не требует адаптера.

## Что отключено vs продвинутый pipeline

| Компонент | Продвинутый (`ml/`) | Baseline (`ml/baselines/trichev_simple/`) |
|---|---|---|
| Retrieval L1 (preambles) | ANN top-4 → adaptive prune (`REL_PRUNE=0.8`) | используется только для richer payload |
| Retrieval L2 | beam top-40 в beam-департаментах | прямой ANN top-`L2_PREFILTER_LIMIT=8` |
| Path scoring | `W_L1·L1_cos + W_L2·L2_cos` | ❌ нет (только L2_cos) |
| Diversify | 1 чанк per **subdept** | 1 чанк per **dept** (грубее) |
| CE кандидатов | top `CE_FINAL_LIMIT=10` | top `CE_CANDIDATES_LIMIT=4` |
| Context injection в CE | `"{Dept}. {Sub}. {chunk}"` | сырой `chunk` |
| Confidence пороги | 3 слоя: RAW_FLOOR / GAP_MIN / SCORE_HIGH+GAP_HIGH | 2 порога: SCORE_HIGH+GAP_HIGH / SCORE_MEDIUM |
| Fallback в unknown | RAW_FLOOR (опц.) или GAP_MIN | только если top < SCORE_MEDIUM |
| Subdept-схема | 18 (`+trade/beaches/tourism`) | 16 (старая, без них) |

## Опубликованные метрики

На 132 примерах из `dataset.json` (синтетика baseline-репо):
- `accuracy_dept = 0.682`
- `accuracy_sub = 0.699`
- `F1_macro = 0.578`
- `latency = 307 мс`

Против продвинутого `dept_top1 = 0.886, sub_top1 = 0.814` на 70 живых УГХ.

## Использование

```python
from ml.baselines.trichev_simple import route, create_collection

create_collection(recreate=True)
result = route("Батареи холодные третий день")
# RouteResult(department='urban_economy', subdepartment='heat', ...)
```

CLI-прогон на 70 живых:
```bash
python eval/run_baseline.py
```

## Vendoring-патчи (минимальные)

При копировании из репо мы внесли 2 правки и переписали в модули:
1. `from knowledge_base import` → `from .knowledge_base import` (relative import)
2. `COLLECTION_NAME = "citizen_complaints"` → `"citizen_complaints_baseline"` (изоляция от продвинутого)
3. Монолит `pipeline_simple.py` (281 строка) разнесён на 7 модулей по 30-150 строк каждый.

Никаких изменений в логике, порогах, retrieval-параметрах.
