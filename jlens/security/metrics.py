"""Calibration and split-locked evaluation metrics for security readouts."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from jlens.security.readouts import METHODS
from jlens.security.runtime import atomic_write_json

SCORE_FIELDS = ("max_logit", "rank_score")


def _method_values(sample: dict[str, Any], method: str) -> tuple[int, float]:
    if method == "model_final":
        scores = sample["model_final"]["security_scores"]
        return scores["best_security_rank"], scores["max_security_logit"]
    aggregate = sample[method]["aggregate"]
    return aggregate["best_security_rank"], aggregate["max_security_logit"]


def readout_score_rows(readouts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in readouts["samples"]:
        for method in METHODS:
            best_rank, max_logit = _method_values(sample, method)
            rows.append(
                {
                    "id": sample["id"],
                    "split": sample.get("split"),
                    "category": sample["category"],
                    "is_injection": int(sample["is_injection"]),
                    "carrier": sample["carrier"],
                    "attack_family": sample.get("attack_family"),
                    "method": method,
                    "best_rank": int(best_rank),
                    "max_logit": float(max_logit),
                    "rank_score": -math.log10(max(int(best_rank), 1)),
                    "elapsed_seconds": float(sample["elapsed_seconds"]),
                }
            )
    return rows


def _binary_metrics(
    labels: list[int],
    scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    predictions = [int(score >= threshold) for score in scores]
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "fpr": float(fpr),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def calibrate_threshold(
    rows: list[dict[str, Any]],
    *,
    method: str,
    score_field: str = "max_logit",
    max_fpr: float | None = 0.05,
) -> dict[str, Any]:
    """Choose the highest-F1 threshold, optionally under an FPR constraint."""
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")
    if score_field not in SCORE_FIELDS:
        raise ValueError(f"Unknown score field: {score_field!r}")
    method_rows = [row for row in rows if row["method"] == method]
    labels = [int(row["is_injection"]) for row in method_rows]
    scores = [float(row[score_field]) for row in method_rows]
    if set(labels) != {0, 1}:
        raise ValueError("Calibration requires both positive and negative labels")

    unique_scores = sorted(set(scores), reverse=True)
    candidates = [math.nextafter(unique_scores[0], math.inf), *unique_scores]
    evaluated = [_binary_metrics(labels, scores, threshold) for threshold in candidates]
    eligible = (
        evaluated
        if max_fpr is None
        else [metrics for metrics in evaluated if metrics["fpr"] <= max_fpr]
    )
    if not eligible:
        eligible = evaluated
    best = max(
        eligible,
        key=lambda metrics: (
            metrics["f1"],
            metrics["recall"],
            metrics["precision"],
            metrics["threshold"],
        ),
    )
    return {
        "method": method,
        "score_field": score_field,
        "max_fpr_constraint": max_fpr,
        "n_samples": len(method_rows),
        **best,
    }


def calibrate_all_methods(
    rows: list[dict[str, Any]],
    *,
    score_field: str = "max_logit",
    max_fpr: float | None = 0.05,
) -> dict[str, dict[str, Any]]:
    return {
        method: calibrate_threshold(
            rows,
            method=method,
            score_field=score_field,
            max_fpr=max_fpr,
        )
        for method in METHODS
    }


def evaluate_method(
    rows: list[dict[str, Any]],
    *,
    method: str,
    threshold: float,
    score_field: str = "max_logit",
) -> dict[str, Any]:
    method_rows = [row for row in rows if row["method"] == method]
    labels = [int(row["is_injection"]) for row in method_rows]
    scores = [float(row[score_field]) for row in method_rows]
    if set(labels) != {0, 1}:
        raise ValueError("Evaluation requires both positive and negative labels")
    return {
        "method": method,
        "score_field": score_field,
        "n_samples": len(method_rows),
        **_binary_metrics(labels, scores, threshold),
    }


def evaluate_all_methods(
    rows: list[dict[str, Any]],
    calibration: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        evaluate_method(
            rows,
            method=method,
            threshold=calibration[method]["threshold"],
            score_field=calibration[method]["score_field"],
        )
        for method in METHODS
    ]


def category_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["category"])].append(row)
    summary = []
    for (method, category), group in sorted(grouped.items()):
        ranks = [row["best_rank"] for row in group]
        logits = [row["max_logit"] for row in group]
        summary.append(
            {
                "method": method,
                "category": category,
                "n": len(group),
                "mean_best_rank": sum(ranks) / len(ranks),
                "mean_max_logit": sum(logits) / len(logits),
                "top_10_rate": sum(rank <= 10 for rank in ranks) / len(ranks),
                "top_50_rate": sum(rank <= 50 for rank in ranks) / len(ranks),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_split_evaluation(
    output_dir: Path,
    *,
    split: str,
    rows: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    calibration: dict[str, dict[str, Any]],
) -> dict[str, Path]:
    """Persist score rows, category summaries and locked-threshold metrics."""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    score_path = split_dir / "sample_scores.csv"
    category_path = split_dir / "category_summary.csv"
    metrics_path = split_dir / "metrics.json"
    _write_csv(score_path, rows)
    _write_csv(category_path, category_summary(rows))
    atomic_write_json(
        metrics_path,
        {
            "split": split,
            "calibration_thresholds": calibration,
            "metrics": metrics,
        },
    )
    return {
        "sample_scores": score_path,
        "category_summary": category_path,
        "metrics": metrics_path,
    }
