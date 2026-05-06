# ML Baselines

Простые модели для сравнения с основным pipeline. Каждый baseline экспортирует совместимый интерфейс:

```python
def route(text: str) -> ml.schemas.RouteResult: ...
def create_collection(recreate: bool = False) -> None: ...
```

Это позволяет прогонять любой baseline через тот же `eval/run_baseline.py`.

---

## Доступные baseline-ы

| Папка | Подход | Источник | Опубликованные метрики |
|---|---|---|---|
| `trichev_simple/` | ANN+CE без улучшений REPORT2/3 | github.com/AndreyTrichev/rag | acc_dept=0.682, acc_sub=0.699 |

## Сравнение с продвинутым

| Подход | Что показывает | sub_top1 на 70 живых |
|---|---|---|
| `trichev_simple` (baseline) | прототип RAG до улучшений REPORT2/3 | ожидаем ~0.65-0.70 |
| **Полный pipeline (`ml/pipeline.py`)** | + beam search v2, context injection, MaxP+3 порога | **0.814** ✅ |

См. также `eval/run_baseline.py` — единый скрипт для прогона любого baseline на 70 живых.
