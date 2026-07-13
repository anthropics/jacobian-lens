import argparse
import json
from pathlib import Path

from jlens.security import (
    calibrate_all_methods,
    evaluate_all_methods,
    readout_score_rows,
    write_split_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate/evaluate comparative security readouts."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("results/gpu/multilayer_security_readouts.json"),
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        help="Optional separate calibration readout; recommended for validation/test.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/gpu/summary"),
    )
    parser.add_argument("--score-field", choices=("max_logit", "rank_score"), default="max_logit")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    return parser.parse_args()


def load_readouts(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Readout not found: {path.resolve()}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    readouts = load_readouts(args.input_path)
    calibration_readouts = (
        load_readouts(args.calibration_path)
        if args.calibration_path
        else readouts
    )
    if args.calibration_path is None:
        print(
            "WARNING: calibrating and evaluating on the same split. "
            "Use --calibration-path for locked-threshold evaluation."
        )
    rows = readout_score_rows(readouts)
    calibration_rows = readout_score_rows(calibration_readouts)
    calibration = calibrate_all_methods(
        calibration_rows,
        score_field=args.score_field,
        max_fpr=args.max_fpr,
    )
    metrics = evaluate_all_methods(rows, calibration)
    split = readouts["samples"][0].get("split") or "evaluation"
    paths = write_split_evaluation(
        args.output_dir,
        split=split,
        rows=rows,
        metrics=metrics,
        calibration=calibration,
    )

    print(
        f"{'Method':<14}{'ROC-AUC':>10}{'PR-AUC':>10}{'Precision':>11}"
        f"{'Recall':>9}{'F1':>9}{'FPR':>9}"
    )
    for row in metrics:
        print(
            f"{row['method']:<14}{row['roc_auc']:>10.4f}{row['pr_auc']:>10.4f}"
            f"{row['precision']:>11.4f}{row['recall']:>9.4f}"
            f"{row['f1']:>9.4f}{row['fpr']:>9.4f}"
        )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
