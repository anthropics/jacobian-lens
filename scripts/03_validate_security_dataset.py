import argparse
from pathlib import Path

from jlens.security import SECURITY_SPLITS, validate_dataset_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate all security datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/security"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "smoke": args.data_dir / "smoke_prompts.jsonl",
        "pilot": args.data_dir / "pilot_prompts.jsonl",
        "calibration": args.data_dir / "calibration.jsonl",
        "validation": args.data_dir / "validation.jsonl",
        "test": args.data_dir / "test.jsonl",
    }
    _, reports = validate_dataset_collection(paths)
    print(
        f"{'Split':<14}{'Samples':>10}{'Clean':>10}{'Hard neg':>12}"
        f"{'Injection':>12}{'Carriers':>10}"
    )
    for report in reports:
        print(
            f"{report['split']:<14}{report['n_samples']:>10}"
            f"{report['n_clean']:>10}{report['n_hard_negative']:>12}"
            f"{report['n_injection']:>12}{report['n_carriers']:>10}"
        )
    assert [report["split"] for report in reports] == list(SECURITY_SPLITS)
    print("All five security datasets are valid and cross-split IDs are disjoint.")


if __name__ == "__main__":
    main()
