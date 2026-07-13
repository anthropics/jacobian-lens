"""Dependency-light SVG charts for notebooks and static reports."""

from __future__ import annotations

from collections import defaultdict
from html import escape
from typing import Any

COLORS = {
    "jlens": "#2563eb",
    "logit_lens": "#f59e0b",
    "model_final": "#10b981",
    "clean": "#60a5fa",
    "hard_negative": "#fbbf24",
    "injection": "#ef4444",
}


def _svg_document(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f"{body}</svg>"
    )


def metric_bar_chart_svg(
    metrics: list[dict[str, Any]],
    *,
    title: str,
) -> str:
    """Grouped 0..1 metric bars for J-lens, logit-lens and model-final."""
    metric_names = ("roc_auc", "pr_auc", "precision", "recall", "f1", "fpr")
    methods = [row["method"] for row in metrics]
    width, height = 920, 430
    left, right, top, bottom = 70, 20, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(metric_names)
    bar_width = min(30, group_width / (len(methods) + 1))
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-size="18" font-family="sans-serif">{escape(title)}</text>'
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_height * (1 - value)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
            'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        body.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" font-family="sans-serif">{value:.1f}</text>'
        )

    for metric_index, metric_name in enumerate(metric_names):
        center = left + group_width * (metric_index + 0.5)
        total_bar_width = bar_width * len(methods)
        start_x = center - total_bar_width / 2
        for method_index, row in enumerate(metrics):
            value = max(0.0, min(1.0, float(row[metric_name])))
            bar_height = plot_height * value
            x = start_x + method_index * bar_width
            y = top + plot_height - bar_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 3:.1f}" '
                f'height="{bar_height:.1f}" fill="{COLORS[row["method"]]}"/>'
            )
        body.append(
            f'<text x="{center:.1f}" y="{height - bottom + 23}" '
            'text-anchor="middle" font-size="12" font-family="sans-serif">'
            f"{escape(metric_name)}</text>"
        )

    legend_x = left
    legend_y = height - 25
    for index, method in enumerate(methods):
        x = legend_x + index * 180
        body.append(
            f'<rect x="{x}" y="{legend_y - 11}" width="13" height="13" '
            f'fill="{COLORS[method]}"/>'
        )
        body.append(
            f'<text x="{x + 19}" y="{legend_y}" font-size="12" '
            f'font-family="sans-serif">{escape(method)}</text>'
        )
    return _svg_document(width, height, "".join(body))


def confusion_matrices_svg(
    metrics: list[dict[str, Any]],
    *,
    title: str,
) -> str:
    width, height = 900, 330
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-size="18" font-family="sans-serif">{escape(title)}</text>'
    ]
    cell = 85
    for method_index, row in enumerate(metrics):
        origin_x = 75 + method_index * 290
        origin_y = 100
        matrix = row["confusion_matrix"]
        maximum = max(max(values) for values in matrix) or 1
        body.append(
            f'<text x="{origin_x + cell}" y="65" text-anchor="middle" '
            f'font-size="15" font-family="sans-serif" '
            f'fill="{COLORS[row["method"]]}">{escape(row["method"])}</text>'
        )
        for actual in range(2):
            for predicted in range(2):
                value = int(matrix[actual][predicted])
                opacity = 0.15 + 0.75 * value / maximum
                x = origin_x + predicted * cell
                y = origin_y + actual * cell
                body.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{COLORS[row["method"]]}" fill-opacity="{opacity:.3f}" '
                    'stroke="white"/>'
                )
                body.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 6}" '
                    f'text-anchor="middle" font-size="18" font-family="sans-serif">'
                    f"{value}</text>"
                )
        body.append(
            f'<text x="{origin_x + cell}" y="{origin_y + 2 * cell + 25}" '
            'text-anchor="middle" font-size="11" font-family="sans-serif">'
            "predicted allow / block</text>"
        )
        body.append(
            f'<text x="{origin_x - 18}" y="{origin_y + cell}" '
            'text-anchor="middle" font-size="11" font-family="sans-serif" '
            f'transform="rotate(-90 {origin_x - 18} {origin_y + cell})">'
            "actual allow / block</text>"
        )
    return _svg_document(width, height, "".join(body))


def category_score_chart_svg(
    rows: list[dict[str, Any]],
    *,
    score_field: str,
    title: str,
) -> str:
    """Mean security score by method/category for pilot diagnostics."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["category"])].append(float(row[score_field]))
    methods = ("jlens", "logit_lens", "model_final")
    categories = ("clean", "hard_negative", "injection")
    means = {
        key: sum(values) / len(values)
        for key, values in grouped.items()
    }
    values = list(means.values())
    low, high = min(values), max(values)
    if high == low:
        high = low + 1.0

    width, height = 850, 420
    left, right, top, bottom = 80, 25, 55, 95
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(methods)
    bar_width = 42
    body = [
        f'<text x="{width / 2}" y="28" text-anchor="middle" '
        f'font-size="18" font-family="sans-serif">{escape(title)}</text>'
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = top + plot_height * (1 - tick / 5)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" '
            'y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        body.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="11" font-family="sans-serif">{value:.2f}</text>'
        )
    for method_index, method in enumerate(methods):
        center = left + group_width * (method_index + 0.5)
        start_x = center - bar_width * len(categories) / 2
        for category_index, category in enumerate(categories):
            value = means[(method, category)]
            normalized = (value - low) / (high - low)
            bar_height = max(2, plot_height * normalized)
            x = start_x + category_index * bar_width
            y = top + plot_height - bar_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 4}" '
                f'height="{bar_height:.1f}" fill="{COLORS[category]}"/>'
            )
        body.append(
            f'<text x="{center:.1f}" y="{height - bottom + 24}" '
            'text-anchor="middle" font-size="12" font-family="sans-serif">'
            f"{escape(method)}</text>"
        )
    for index, category in enumerate(categories):
        x = left + index * 190
        body.append(
            f'<rect x="{x}" y="{height - 26}" width="13" height="13" '
            f'fill="{COLORS[category]}"/>'
        )
        body.append(
            f'<text x="{x + 19}" y="{height - 15}" font-size="12" '
            f'font-family="sans-serif">{escape(category)}</text>'
        )
    return _svg_document(width, height, "".join(body))
