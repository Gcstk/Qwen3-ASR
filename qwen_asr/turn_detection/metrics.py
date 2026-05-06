# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_numpy_1d(x: Sequence[Any], dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def binary_precision_recall_f1(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    pos_label: int = 1,
) -> Dict[str, float]:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_pred = _as_numpy_1d(y_pred, dtype=np.int64)

    tp = int(((y_true == pos_label) & (y_pred == pos_label)).sum())
    fp = int(((y_true != pos_label) & (y_pred == pos_label)).sum())
    fn = int(((y_true == pos_label) & (y_pred != pos_label)).sum())
    tn = int(((y_true != pos_label) & (y_pred != pos_label)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def binary_roc_auc(y_true: Sequence[int], y_score: Sequence[float], pos_label: int = 1) -> float:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_score = _as_numpy_1d(y_score, dtype=np.float64)

    pos = (y_true == pos_label).astype(np.int64)
    n_pos = int(pos.sum())
    n_neg = int(len(pos) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    rank_sum_pos = float(ranks[pos == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(1.0, float(n_pos * n_neg))
    return float(auc)


def binary_average_precision(y_true: Sequence[int], y_score: Sequence[float], pos_label: int = 1) -> float:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_score = _as_numpy_1d(y_score, dtype=np.float64)

    pos = (y_true == pos_label).astype(np.int64)
    n_pos = int(pos.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    pos_sorted = pos[order]
    tp = np.cumsum(pos_sorted)
    fp = np.cumsum(1 - pos_sorted)
    precision = tp / np.maximum(1, tp + fp)
    ap = float((precision * pos_sorted).sum() / max(1, n_pos))
    return ap


def binary_brier_score(y_true: Sequence[int], y_prob: Sequence[float], pos_label: int = 1) -> float:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_prob = _as_numpy_1d(y_prob, dtype=np.float64)
    target = (y_true == pos_label).astype(np.float64)
    return float(np.mean((y_prob - target) ** 2))


def expected_calibration_error(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    pos_label: int = 1,
    n_bins: int = 15,
) -> float:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_prob = np.clip(_as_numpy_1d(y_prob, dtype=np.float64), 0.0, 1.0)
    target = (y_true == pos_label).astype(np.float64)

    if len(y_prob) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for i in range(int(n_bins)):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == int(n_bins) - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(target[mask].mean())
        ece += abs(avg_conf - avg_acc) * (count / len(y_prob))
    return float(ece)


def threshold_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float,
    pos_label: int = 1,
) -> Dict[str, float]:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_prob = _as_numpy_1d(y_prob, dtype=np.float64)
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    metrics = binary_precision_recall_f1(y_true, y_pred, pos_label=pos_label)
    metrics["threshold"] = float(threshold)
    metrics["positive_rate"] = float(y_pred.mean()) if len(y_pred) > 0 else float("nan")
    return metrics


def _candidate_thresholds(y_prob: Sequence[float]) -> np.ndarray:
    y_prob = _as_numpy_1d(y_prob, dtype=np.float64)
    if len(y_prob) == 0:
        return np.asarray([0.5], dtype=np.float64)
    uniq = np.unique(y_prob)
    padded = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            uniq,
            np.asarray([1.0], dtype=np.float64),
        ]
    )
    return np.unique(padded)


def pick_high_precision_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    target_precision: float = 0.97,
    pos_label: int = 1,
) -> Dict[str, float]:
    best: Optional[Dict[str, float]] = None
    for thr in _candidate_thresholds(y_prob):
        metrics = threshold_metrics(y_true, y_prob, threshold=float(thr), pos_label=pos_label)
        if metrics["precision"] + 1e-12 < float(target_precision):
            continue
        if best is None:
            best = metrics
            continue
        if metrics["recall"] > best["recall"] + 1e-12:
            best = metrics
            continue
        if abs(metrics["recall"] - best["recall"]) <= 1e-12 and metrics["f1"] > best["f1"] + 1e-12:
            best = metrics
            continue
        if (
            abs(metrics["recall"] - best["recall"]) <= 1e-12
            and abs(metrics["f1"] - best["f1"]) <= 1e-12
            and metrics["threshold"] < best["threshold"]
        ):
            best = metrics
    if best is None:
        best = threshold_metrics(y_true, y_prob, threshold=1.0, pos_label=pos_label)
        best["target_precision_unmet"] = 1.0
    else:
        best["target_precision_unmet"] = 0.0
    best["target_precision"] = float(target_precision)
    return best


def pick_best_f1_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    pos_label: int = 1,
) -> Dict[str, float]:
    best: Optional[Dict[str, float]] = None
    for thr in _candidate_thresholds(y_prob):
        metrics = threshold_metrics(y_true, y_prob, threshold=float(thr), pos_label=pos_label)
        if best is None or metrics["f1"] > best["f1"] + 1e-12:
            best = metrics
            continue
        if abs(metrics["f1"] - best["f1"]) <= 1e-12 and metrics["precision"] > best["precision"] + 1e-12:
            best = metrics
            continue
        if (
            abs(metrics["f1"] - best["f1"]) <= 1e-12
            and abs(metrics["precision"] - best["precision"]) <= 1e-12
            and metrics["threshold"] < best["threshold"]
        ):
            best = metrics
    return best or threshold_metrics(y_true, y_prob, threshold=0.5, pos_label=pos_label)


def summarize_binary_classification(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    pos_label: int = 1,
    target_precision: float = 0.97,
    default_threshold: float = 0.5,
    ece_bins: int = 15,
) -> Dict[str, Any]:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_prob = np.clip(_as_numpy_1d(y_prob, dtype=np.float64), 0.0, 1.0)

    out: Dict[str, Any] = {
        "num_examples": int(len(y_true)),
        "positive_rate_true": float((y_true == pos_label).mean()) if len(y_true) > 0 else float("nan"),
        "positive_rate_prob_mean": float(y_prob.mean()) if len(y_prob) > 0 else float("nan"),
        "auroc": binary_roc_auc(y_true, y_prob, pos_label=pos_label),
        "auprc": binary_average_precision(y_true, y_prob, pos_label=pos_label),
        "brier": binary_brier_score(y_true, y_prob, pos_label=pos_label),
        "ece": expected_calibration_error(y_true, y_prob, pos_label=pos_label, n_bins=ece_bins),
    }
    out["threshold_default_0_5"] = threshold_metrics(y_true, y_prob, threshold=default_threshold, pos_label=pos_label)
    out["threshold_high_precision"] = pick_high_precision_threshold(
        y_true,
        y_prob,
        target_precision=target_precision,
        pos_label=pos_label,
    )
    out["threshold_balanced_f1"] = pick_best_f1_threshold(y_true, y_prob, pos_label=pos_label)
    return out


def latency_summary(latencies_ms: Sequence[float]) -> Dict[str, float]:
    arr = _as_numpy_1d(latencies_ms, dtype=np.float64)
    if len(arr) == 0:
        return {
            "count": 0.0,
            "mean_ms": float("nan"),
            "p50_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
        }
    return {
        "count": float(len(arr)),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def normalize_binary_label(label: Any) -> Optional[int]:
    if label is None:
        return None
    if isinstance(label, (int, np.integer)):
        if int(label) in (0, 1):
            return int(label)
    s = str(label).strip().lower()
    if s == "complete":
        return 1
    if s == "incomplete":
        return 0
    return None


def canonical_prediction_label(text: Any) -> str:
    s = "" if text is None else str(text)
    s = " ".join(s.strip().lower().split())
    if s.startswith("incomplete"):
        return "incomplete"
    if s.startswith("complete"):
        return "complete"
    first = s.split(" ", 1)[0] if s else ""
    if first in {"complete", "incomplete"}:
        return first
    return s


def build_slice_groups(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, List[int]]]:
    groups: Dict[str, Dict[str, List[int]]] = {}

    def _add(slice_name: str, bucket_name: str, idx: int):
        groups.setdefault(slice_name, {}).setdefault(bucket_name, []).append(idx)

    for idx, rec in enumerate(records):
        pause_ms = rec.get("pause_ms", rec.get("pause_duration_ms"))
        if pause_ms is not None and str(pause_ms) != "":
            val = float(pause_ms)
            _add("pause_length", "short_pause" if val < 500.0 else "long_pause", idx)

        utterance_type = rec.get("utterance_type", rec.get("completion_type"))
        if utterance_type not in (None, ""):
            _add("utterance_type", str(utterance_type), idx)

        language = rec.get("language")
        if language not in (None, ""):
            _add("language", str(language), idx)

        speaking_rate = rec.get("speaking_rate")
        if speaking_rate not in (None, ""):
            val = float(speaking_rate)
            if val < 3.0:
                bucket = "slow"
            elif val < 5.0:
                bucket = "medium"
            else:
                bucket = "fast"
            _add("speaking_rate", bucket, idx)

        noise_level = rec.get("noise_level")
        if noise_level not in (None, ""):
            _add("noise_level", str(noise_level), idx)

        snr_db = rec.get("snr_db")
        if snr_db not in (None, ""):
            val = float(snr_db)
            if val < 10.0:
                bucket = "noisy"
            elif val < 20.0:
                bucket = "medium_snr"
            else:
                bucket = "clean"
            _add("snr_db", bucket, idx)

        overlap = rec.get("overlap_speech", rec.get("has_overlap"))
        if overlap not in (None, ""):
            overlap_val = bool(overlap) if isinstance(overlap, bool) else str(overlap).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "overlap",
            }
            _add("overlap_speech", "overlap" if overlap_val else "no_overlap", idx)

        tail_dragging = rec.get("tail_dragging", rec.get("tail_drag", rec.get("has_tail_drag")))
        if tail_dragging not in (None, ""):
            drag_val = bool(tail_dragging) if isinstance(tail_dragging, bool) else str(tail_dragging).strip().lower() in {
                "1",
                "true",
                "yes",
                "y",
                "tail_drag",
            }
            _add("tail_dragging", "tail_dragging" if drag_val else "normal_tail", idx)

        candidate_offset_ms = rec.get("candidate_offset_ms", rec.get("vad_offset_ms"))
        if candidate_offset_ms not in (None, ""):
            val = float(candidate_offset_ms)
            if val < -200.0:
                bucket = "vad_early"
            elif val > 200.0:
                bucket = "vad_late"
            else:
                bucket = "vad_on_time"
            _add("candidate_offset_ms", bucket, idx)

    return groups


def summarize_slices(
    records: Sequence[Dict[str, Any]],
    y_true: Sequence[int],
    y_prob: Sequence[float],
    pos_label: int = 1,
    target_precision: float = 0.97,
) -> Dict[str, Any]:
    y_true = _as_numpy_1d(y_true, dtype=np.int64)
    y_prob = _as_numpy_1d(y_prob, dtype=np.float64)
    groups = build_slice_groups(records)
    out: Dict[str, Any] = {}
    for slice_name, buckets in groups.items():
        out[slice_name] = {}
        for bucket_name, indices in buckets.items():
            idx = np.asarray(indices, dtype=np.int64)
            if len(idx) == 0:
                continue
            out[slice_name][bucket_name] = summarize_binary_classification(
                y_true[idx],
                y_prob[idx],
                pos_label=pos_label,
                target_precision=target_precision,
            )
    return out


def log_softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x, axis=axis, keepdims=True)
    shifted = x - x_max
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    return shifted - logsumexp


def finite_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        fv = float(value)
    except Exception:
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv

