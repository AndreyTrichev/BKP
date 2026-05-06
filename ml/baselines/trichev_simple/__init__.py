"""Baseline: ранний прототип ANN+CE без улучшений REPORT2/3.

Vendored из https://github.com/AndreyTrichev/rag (commit 3b32407, 2026-03-22),
рефакторен в модульную структуру по аналогии с продвинутым `ml/` пакетом.

Используется в дипломной работе как точка отсчёта: показывает прирост
метрик от архитектурных улучшений (beam search v2, adaptive prune,
context injection в CE, MaxP-per-dept со всеми dept, RAW_FLOOR fallback).
"""
from .index import create_collection
from .pipeline import get_routing_context, route

__all__ = ["route", "create_collection", "get_routing_context"]
